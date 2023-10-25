import os
import sys
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import argbind
import audiotools as at
import torch
import torch.nn as nn
from audiotools import AudioSignal
from audiotools.data import transforms
from einops import rearrange
from rich import pretty
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter

import vampnet
from vampnet.modules.transformer import VampNet
from vampnet.util import codebook_unflatten, codebook_flatten
from vampnet import mask as pmask
# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC

from audiotools.ml.decorators import (
    timer, Tracker, when
)

import loralib as lora

import torch._dynamo
torch._dynamo.config.verbose=True


# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Install to make things look nice
warnings.filterwarnings("ignore", category=UserWarning)
pretty.install()
install()

# optim
Accelerator = argbind.bind(at.ml.Accelerator, without_prefix=True)
CrossEntropyLoss = argbind.bind(nn.CrossEntropyLoss)
AdamW = argbind.bind(torch.optim.AdamW)
NoamScheduler = argbind.bind(vampnet.scheduler.NoamScheduler)

# transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# model
VampNet = argbind.bind(VampNet)


# data
AudioLoader = argbind.bind(at.datasets.AudioLoader)
AudioDataset = argbind.bind(at.datasets.AudioDataset, "train", "val")

IGNORE_INDEX = -100


@argbind.bind("train", "val", without_prefix=True)
def build_transform():
    transform = transforms.Compose(
        tfm.VolumeNorm(("const", -24)),
        # tfm.PitchShift(),
        tfm.RescaleAudio(),
    )
    return transform


@torch.no_grad()
def apply_transform(transform_fn, batch):
    sig: AudioSignal = batch["signal"]
    kwargs = batch["transform_args"]

    sig: AudioSignal = transform_fn(sig.clone(), **kwargs)
    return sig


def build_datasets(args, sample_rate: int):
    with argbind.scope(args, "train"):
        train_data = AudioDataset(
            AudioLoader(), sample_rate, transform=build_transform()
        )
    with argbind.scope(args, "val"):
        val_data = AudioDataset(AudioLoader(), sample_rate, transform=build_transform())
    return train_data, val_data


def rand_float(shape, low, high, rng):
    return rng.draw(shape)[:, 0] * (high - low) + low


def flip_coin(shape, p, rng):
    return rng.draw(shape)[:, 0] < p


def num_params_hook(o, p):
    return o + f" {p/1e6:<.3f}M params."


def add_num_params_repr_hook(model):
    import numpy as np
    from functools import partial

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])

        setattr(m, "extra_repr", partial(num_params_hook, o=o, p=p))


def accuracy(
    preds: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 1,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    # Flatten the predictions and targets to be of shape (batch_size * sequence_length, n_class)
    preds = rearrange(preds, "b p s -> (b s) p")
    target = rearrange(target, "b s -> (b s)")

    # return torchmetrics.functional.accuracy(preds, target, task='multiclass', top_k=topk, num_classes=preds.shape[-1], ignore_index=ignore_index)
    if ignore_index is not None:
        # Create a mask for the ignored index
        mask = target != ignore_index
        # Apply the mask to the target and predictions
        preds = preds[mask]
        target = target[mask]

    # Get the top-k predicted classes and their indices
    _, pred_indices = torch.topk(preds, k=top_k, dim=-1)

    # Determine if the true target is in the top-k predicted classes
    correct = torch.sum(torch.eq(pred_indices, target.unsqueeze(1)), dim=1)

    # Calculate the accuracy
    accuracy = torch.mean(correct.float())

    return accuracy

def _metrics(z_hat, r, target, flat_mask, output):
    for r_range in [(0, 0.5), (0.5, 1.0)]:
        unmasked_target = target.masked_fill(flat_mask.bool(), IGNORE_INDEX)
        masked_target = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)

        assert target.shape[0] == r.shape[0]
        # grab the indices of the r values that are in the range
        r_idx = (r >= r_range[0]) & (r < r_range[1])

        # grab the target and z_hat values that are in the range
        r_unmasked_target = unmasked_target[r_idx]
        r_masked_target = masked_target[r_idx]
        r_z_hat = z_hat[r_idx]

        for topk in (1, 25):
            s, e = r_range
            tag = f"accuracy-{s}-{e}/top{topk}"

            output[f"{tag}/unmasked"] = accuracy(
                preds=r_z_hat,
                target=r_unmasked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )
            output[f"{tag}/masked"] = accuracy(
                preds=r_z_hat,
                target=r_masked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )


@dataclass
class State:
    model: VampNet
    codec: DAC

    optimizer: AdamW
    scheduler: NoamScheduler
    criterion: CrossEntropyLoss
    grad_clip_val: float

    rng: torch.quasirandom.SobolEngine

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


@timer()
def train_loop(state: State, batch: dict, accel: Accelerator):
    state.model.train()
    batch = at.util.prepare_batch(batch, accel.device)
    signal = apply_transform(state.train_data.transform, batch)

    output = {}
    vn = accel.unwrap(state.model)
    with accel.autocast():
        with torch.inference_mode():
            state.codec.to(accel.device)
            z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
            z = z[:, : vn.n_codebooks, :]

        n_batch = z.shape[0]
        r = state.rng.draw(n_batch)[:, 0].to(accel.device)

        mask = pmask.random(z, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
        z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)
        
        z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

        dtype = torch.bfloat16 if accel.amp else None
        with accel.autocast(dtype=dtype):
            z_hat = state.model(z_mask_latent)

        target = codebook_flatten(
            z[:, vn.n_conditioning_codebooks :, :],
        )

        flat_mask = codebook_flatten(
            mask[:, vn.n_conditioning_codebooks :, :],
        )

        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
        output["loss"] = state.criterion(z_hat, t_masked)

        _metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=flat_mask,
            output=output,
        )

    
    accel.backward(output["loss"])

    output["other/learning_rate"] = state.optimizer.param_groups[0]["lr"]
    output["other/batch_size"] = z.shape[0]


    accel.scaler.unscale_(state.optimizer)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), state.grad_clip_val
    )

    accel.step(state.optimizer)
    state.optimizer.zero_grad()

    state.scheduler.step()
    accel.update()


    return {k: v for k, v in sorted(output.items())}


@timer()
@torch.no_grad()
def val_loop(state: State, batch: dict, accel: Accelerator):
    state.model.eval()
    state.codec.eval()
    batch = at.util.prepare_batch(batch, accel.device)
    signal = apply_transform(state.val_data.transform, batch)

    vn = accel.unwrap(state.model)
    z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
    z = z[:, : vn.n_codebooks, :]

    n_batch = z.shape[0]
    r = state.rng.draw(n_batch)[:, 0].to(accel.device)

    mask = pmask.random(z, r)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

    z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

    z_hat = state.model(z_mask_latent)

    target = codebook_flatten(
        z[:, vn.n_conditioning_codebooks :, :],
    )

    flat_mask = codebook_flatten(
        mask[:, vn.n_conditioning_codebooks :, :]
    )

    output = {}
    # replace target with ignore index for masked tokens
    t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
    output["loss"] = state.criterion(z_hat, t_masked)

    _metrics(
        r=r,
        z_hat=z_hat,
        target=target,
        flat_mask=flat_mask,
        output=output,
    )

    return output


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(state, batch, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
    return output


def checkpoint(state, save_iters, save_path, fine_tune):
    if accel.local_rank != 0:
        state.tracker.print(f"ERROR:Skipping checkpoint on rank {accel.local_rank}")
        return

    metadata = {"logs": dict(state.tracker.history)}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")

    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    if state.tracker.is_best("val", "loss"):
        state.tracker.print(f"Best model so far")
        tags.append("best")

    if fine_tune:
        for tag in tags: 
            # save the lora model 
            (Path(save_path) / tag).mkdir(parents=True, exist_ok=True)
            torch.save(
                lora.lora_state_dict(accel.unwrap(state.model)), 
                f"{save_path}/{tag}/lora.pth"
            )

    for tag in tags:
        model_extra = {
            "optimizer.pth": state.optimizer.state_dict(),
            "scheduler.pth": state.scheduler.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }

        accel.unwrap(state.model).metadata = metadata
        accel.unwrap(state.model).save_to_folder(
            f"{save_path}/{tag}", model_extra, package=False
        )


def save_sampled(state, z, writer):
    num_samples = z.shape[0]

    for i in range(num_samples):
        sampled = accel.unwrap(state.model).generate(
            codec=state.codec,
            time_steps=z.shape[-1],
            start_tokens=z[i : i + 1],
        )
        sampled.cpu().write_audio_to_tb(
            f"sampled/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )


def save_imputation(state, z, val_idx, writer):
    n_prefix = int(z.shape[-1] * 0.25)
    n_suffix = int(z.shape[-1] *  0.25)

    vn = accel.unwrap(state.model)

    mask = pmask.inpaint(z, n_prefix, n_suffix)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

    imputed_noisy = vn.to_signal(z_mask, state.codec)
    imputed_true = vn.to_signal(z, state.codec)

    imputed = []
    for i in range(len(z)):
        imputed.append(
            vn.generate(
                codec=state.codec,
                time_steps=z.shape[-1],
                start_tokens=z[i][None, ...],
                mask=mask[i][None, ...],
            )   
        )   
    imputed = AudioSignal.batch(imputed)

    for i in range(len(val_idx)):
        imputed_noisy[i].cpu().write_audio_to_tb(
            f"inpainted_prompt/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )
        imputed[i].cpu().write_audio_to_tb(
            f"inpainted_middle/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )
        imputed_true[i].cpu().write_audio_to_tb(
            f"reconstructed/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )


@torch.no_grad()
def save_samples(state: State, val_idx: int, writer: SummaryWriter):
    state.model.eval()
    state.codec.eval()
    vn = accel.unwrap(state.model)

    batch = [state.val_data[i] for i in val_idx]
    batch = at.util.prepare_batch(state.val_data.collate(batch), accel.device)

    signal = apply_transform(state.val_data.transform, batch)

    z = state.codec.encode(signal.samples, signal.sample_rate)["codes"]
    z = z[:, : vn.n_codebooks, :]

    r = torch.linspace(0.1, 0.95, len(val_idx)).to(accel.device)


    mask = pmask.random(z, r)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.mask_token)

    z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

    z_hat = state.model(z_mask_latent)

    z_pred = torch.softmax(z_hat, dim=1).argmax(dim=1)
    z_pred = codebook_unflatten(z_pred, n_c=vn.n_predict_codebooks)
    z_pred = torch.cat([z[:, : vn.n_conditioning_codebooks, :], z_pred], dim=1)

    generated = vn.to_signal(z_pred, state.codec)
    reconstructed = vn.to_signal(z, state.codec)
    masked = vn.to_signal(z_mask.squeeze(1), state.codec)

    for i in range(generated.batch_size):
        audio_dict = {
            "original": signal[i],
            "masked": masked[i],
            "generated": generated[i],
            "reconstructed": reconstructed[i],
        }
        for k, v in audio_dict.items():
            v.cpu().write_audio_to_tb(
                f"onestep/_{i}.r={r[i]:0.2f}/{k}",
                writer,
                step=state.tracker.step,
                plot_fn=None,
            )

    save_sampled(state=state, z=z, writer=writer)
    save_imputation(state=state, z=z, val_idx=val_idx, writer=writer)



@argbind.bind(without_prefix=True)
def load(
    args,
    accel: at.ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    fine_tune_checkpoint: Optional[str] = None,
    grad_clip_val: float = 5.0,
) -> State:
    codec = DAC.load(args["codec_ckpt"], map_location="cpu")
    codec.eval()

    model, v_extra = None, {}

    if args["fine_tune"]:
        assert fine_tune_checkpoint is not None, "Must provide a fine-tune checkpoint"
        model = torch.compile(
            VampNet.load(location=Path(fine_tune_checkpoint), 
                         map_location="cpu", 
            )
        )
        
    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": False,
        }
        tracker.print(f"Loading checkpoint from {kwargs['folder']}")
        if (Path(kwargs["folder"]) / "vampnet").exists():
            model, v_extra = VampNet.load_from_folder(**kwargs)
        else:
            raise ValueError(
                f"Could not find a VampNet checkpoint in {kwargs['folder']}"
            )




    model = torch.compile(VampNet()) if model is None else model
    model = accel.prepare_model(model)

    # assert accel.unwrap(model).n_codebooks == codec.quantizer.n_codebooks
    assert (
        accel.unwrap(model).vocab_size == codec.quantizer.quantizers[0].codebook_size
    )

    optimizer = AdamW(model.parameters(), use_zero=accel.use_ddp)
    scheduler = NoamScheduler(optimizer, d_model=accel.unwrap(model).embedding_dim)
    scheduler.step()

    if "optimizer.pth" in v_extra:
        optimizer.load_state_dict(v_extra["optimizer.pth"])
        scheduler.load_state_dict(v_extra["scheduler.pth"])
    if "tracker.pth" in v_extra:
        tracker.load_state_dict(v_extra["tracker.pth"])
    
    criterion = CrossEntropyLoss()

    sample_rate = codec.sample_rate

    # a better rng for sampling from our schedule
    rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=args["seed"])  

    # log a model summary w/ num params
    if accel.local_rank == 0:
        add_num_params_repr_hook(accel.unwrap(model))
        with open(f"{save_path}/model.txt", "w") as f:
            f.write(repr(accel.unwrap(model)))

    # load the datasets
    train_data, val_data = build_datasets(args, sample_rate)

    return State(
        tracker=tracker,
        model=model,
        codec=codec,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        rng=rng,
        train_data=train_data,
        val_data=val_data,
        grad_clip_val=grad_clip_val,
    )


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: at.ml.Accelerator,
    seed: int = 0,
    codec_ckpt: str = None,
    save_path: str = "ckpt",
    num_iters: int = int(1000e6),
    save_iters: list = [10000, 50000, 100000, 300000, 500000,],
    sample_freq: int = 10000, 
    val_freq: int = 1000,
    batch_size: int = 12,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    num_workers: int = 10,
    fine_tune: bool = False, 
):
    assert codec_ckpt is not None, "codec_ckpt is required"

    seed = seed + accel.local_rank
    at.util.seed(seed)
    writer = None

    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir=f"{save_path}/logs/")
        argbind.dump_args(args, f"{save_path}/args.yml")

        tracker = Tracker(
            writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
        )

    # load the codec model
    state: State = load(
        args=args, 
        accel=accel, 
        tracker=tracker, 
        save_path=save_path)
    print("initialized state.")

    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=num_workers > 0,
    )
    print("initialized dataloader.")

    

    if fine_tune:
        lora.mark_only_lora_as_trainable(state.model)
        print("marked only lora as trainable.")

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint

    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    print("starting training loop.")
    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )

            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer)

            if tracker.step % val_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(
                    state=state, 
                    save_iters=save_iters, 
                    save_path=save_path, 
                    fine_tune=fine_tune)

                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
