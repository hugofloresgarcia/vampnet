import os
import sys
import warnings
import yaml
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import time

import audiotools as at
import torch
import torch.nn as nn
from audiotools import AudioSignal
from dac.model.dac import DAC
import pandas as pd

from einops import rearrange
from rich import pretty
from rich.traceback import install
from torch.distributed.elastic.multiprocessing.errors import record

from torch.utils.tensorboard import SummaryWriter

import vampnet
from vampnet import mask as pmask
from vampnet.model.scheduler import NoamScheduler
from vampnet.model.transformer import VampNet
from vampnet.util import codebook_flatten
from vampnet.util import codebook_unflatten
import vampnet.util

from audiotools.ml.decorators import (
    timer, Tracker, when
)
                                                                  
import loralib as lora

import torch._dynamo

torch._dynamo.config.suppress_errors = True     
torch._dynamo.config.verbose=True
torch._dynamo.skip_nnmodule_hook_guards=False

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


from vampnet.db.data import build_datasets, VampNetDataset



@dataclass
class State:
    model: VampNet
    codec: DAC

    optimizer: torch.optim.AdamW
    scheduler: NoamScheduler
    criterion: nn.CrossEntropyLoss
    grad_clip_val: float

    rng: torch.quasirandom.SobolEngine

    train_data: VampNetDataset
    val_data: VampNetDataset
    sample_data: VampNetDataset

    tracker: Tracker
    grad_acc_steps: int = 1
    compute_loss_on_masked_tokens_only: bool = True


def preprocess(state: State, accel, batch: dict, stage: str):
    z_in = batch["codes"]
    z_out = batch["codes"]
    z_cond = batch["ctrls"]
    ctx_mask = batch["ctx_mask"]

    # if z_cond is a list of Nones, let's make it a single None
    if z_cond is not None and all([z is None for z in z_cond]):
        z_cond = None

    z_in = z_in[:, :accel.unwrap(state.model).n_codebooks, :]
    z_out = z_out[:, :accel.unwrap(state.model).n_codebooks, :]
    return z_in, z_out, ctx_mask, z_cond


@timer()
def train_loop(state: State, batch: dict, accel: at.ml.Accelerator):
    state.model.train()
    batch = at.util.prepare_batch(batch, accel.device)

    z_in, z_out, ctx_mask, z_cond = preprocess(state, accel, batch, "train")

    output = {}
    vn = accel.unwrap(state.model)
    dtype = torch.float16 if accel.amp else None
        
    with accel.autocast(dtype=dtype):

        n_batch = z_in.shape[0]
        r = state.rng.draw(n_batch)[:, 0].to(accel.device)

        mask, ignore_indices_mask = pmask.hugo_random(z_in, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
        z_mask, mask = pmask.apply_mask(z_in, mask, vn.special_tokens["MASK"])
        
        z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

        # TODO: need to run z cond through and embedding model AND CLIP THE RANGE TO (0, 1) for the dataset
        z_hat = state.model(z_mask_latent, pad_mask=ctx_mask, cross_x=z_cond, cross_pad_mask=ctx_mask)

        target = codebook_flatten(
            z_out[:, vn.n_conditioning_codebooks :, :],
        )

        # ctx mask is 1 where there is real data, 0 where there is padding
        # mask is 1 where there is generated data, 0 where there is real data
        # we want the loss mask to be 1 where we infer and 0 where we condition
        # loss mask = ctx_mask & mask
        ctx_mask = ctx_mask.unsqueeze(1).repeat_interleave(vn.n_predict_codebooks, dim=1)
        if state.compute_loss_on_masked_tokens_only:
            loss_mask = codebook_flatten(
                torch.logical_and(
                    mask[:, vn.n_conditioning_codebooks :, :].bool(),
                    ctx_mask,
                )
            )
        else:
            loss_mask = codebook_flatten(ctx_mask)

        # add the ignore indices mask to the loss mask
        loss_mask = ~loss_mask
        loss_mask = loss_mask | codebook_flatten(ignore_indices_mask)

        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(loss_mask, vampnet.IGNORE_INDEX)
        output["loss"] = state.criterion(z_hat, t_masked)
        _metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=~loss_mask,
            output=output,
        )

    if state.tracker.step % state.grad_acc_steps == 0:
        state.optimizer.zero_grad()
        accel.backward(output["loss"])
        accel.scaler.unscale_(state.optimizer)
        output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), state.grad_clip_val
        )
        accel.step(state.optimizer)

        output["other/learning_rate"] = state.optimizer.param_groups[0]["lr"]
        output["other/batch_size"] = n_batch

        state.scheduler.step()
        accel.update()


    return {k: v for k, v in sorted(output.items())}


@timer()
@torch.no_grad()
def val_loop(state: State, batch: dict, accel: at.ml.Accelerator):
    state.model.eval()
    state.codec.eval()

    batch = at.util.prepare_batch(batch, accel.device)
    z_in, z_out, ctx_mask, z_cond = preprocess(state, accel, batch, "val")

    vn = accel.unwrap(state.model)
    
    output = {}

    with accel.autocast(dtype=torch.float16):
        n_batch = z_in.shape[0]
        r = state.rng.draw(n_batch)[:, 0].to(accel.device)

        mask, ignore_indices_mask = pmask.hugo_random(z_in, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
        z_mask, mask = pmask.apply_mask(z_in, mask, vn.special_tokens["MASK"])

        z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

        z_hat = state.model(z_mask_latent, pad_mask=ctx_mask, cross_x=z_cond, cross_pad_mask=ctx_mask)

        target = codebook_flatten(
            z_out[:, vn.n_conditioning_codebooks :, :],
        )

        # ctx mask is 1 where there is real data, 0 where there is padding
        # mask is 1 where there is generated data, 0 where there is real data
        # we want the loss mask to be 1 where we infer and 0 where we condition
        # loss mask = ctx_mask & mask
        ctx_mask = ctx_mask.unsqueeze(1).repeat_interleave(vn.n_predict_codebooks, dim=1)
        if state.compute_loss_on_masked_tokens_only:
            loss_mask = codebook_flatten(
                torch.logical_and(
                    mask[:, vn.n_conditioning_codebooks :, :].bool(),
                    ctx_mask,
                )
            )
        else:
            loss_mask = codebook_flatten(ctx_mask)

        # add the ignore indices mask to the loss mask
        loss_mask = ~loss_mask
        loss_mask = loss_mask | codebook_flatten(ignore_indices_mask)

        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(loss_mask, vampnet.IGNORE_INDEX)
        output["loss"] = state.criterion(z_hat, t_masked)

        _metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=~loss_mask,
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


def save_sampled(state, z, cross_x, writer):
    num_samples = z.shape[0]

    for i in range(num_samples):
        cx = cross_x[i][None, ...] if cross_x is not None else None
        sampled = state.interface.to_signal(
            state.interface.model.generate(
            codec=state.codec,
            cross_x=cx,
            time_steps=z.shape[-1],
            start_tokens=z[i : i + 1],
        ))
        sampled.cpu().write_audio_to_tb(
            f"sampled/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )


def save_inpainting(state, z, val_idx, cross_x, writer):
    n_prefix = int(z.shape[-1] * 0.25)
    n_suffix = int(z.shape[-1] *  0.25)

    vn = accel.unwrap(state.model)

    mask = pmask.inpaint(z, n_prefix, n_suffix)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z, mask, vn.special_tokens["MASK"])

    inpainted_prompt = state.interface.to_signal(z_mask,silence_mask=True)
    inpainted_gnd_truth = state.interface.to_signal(z, )

    inpainted = []
    for i in range(len(z)):
        cx = cross_x[i][None, ...] if cross_x is not None else None
        inpainted.append(
            state.interface.to_signal(
                state.interface.model.generate(
                codec=state.codec,
                cross_x=cx,
                time_steps=z.shape[-1],
                start_tokens=z[i][None, ...],
                mask=mask[i][None, ...],
            ))   
        )   
    inpainted = AudioSignal.batch(inpainted)

    for i in range(len(val_idx)):
        inpainted_prompt[i].cpu().write_audio_to_tb(
            f"inpainted_prompt/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )
        inpainted[i].cpu().write_audio_to_tb(
            f"inpainted/{i}",
            writer,
            step=state.tracker.step,
            plot_fn=None,
        )
        inpainted_gnd_truth[i].cpu().write_audio_to_tb(
            f"inpainted_gnd_truth/{i}",
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

    z_in, z_out, ctx_mask, z_cond = preprocess(state,accel,  batch, "sample")
    z_in = z_in[:, : vn.n_codebooks, :]

    r = torch.linspace(0.1, 0.95, len(val_idx)).to(accel.device)

    mask, ignore_indices_mask = pmask.hugo_random(z_in, r)
    mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
    z_mask, mask = pmask.apply_mask(z_in, mask, vn.special_tokens["MASK"])

    z_mask_latent = vn.embedding.from_codes(z_mask, state.codec)

    z_hat = state.model(z_mask_latent, cross_x=z_cond)

    z_pred = torch.softmax(z_hat, dim=1).argmax(dim=1)
    z_pred = codebook_unflatten(z_pred, n_c=vn.n_predict_codebooks)
    z_pred = torch.cat([z_in[:, : vn.n_conditioning_codebooks, :], z_pred], dim=1)
    z_pred, _ = pmask.apply_mask(z_pred, (~mask.bool()).long(), z_mask)

    state.interface = vampnet.interface.Interface(state.codec, accel.unwrap(state.model))

    generated = state.interface.to_signal(z_pred, )
    reconstructed = state.interface.to_signal(z_out, )
    masked = state.interface.to_signal(z_mask, silence_mask=False)

    for i in range(generated.batch_size):
        audio_dict = {
            # "original": signal[i],
            "masked": masked[i],
            "generated": generated[i],
            "reconstructed": reconstructed[i],
        }
        for k, v in audio_dict.items():
            v.cpu().write_audio_to_tb(
                f"samples/_{i}.r={r[i]:0.2f}/{k}",
                writer,
                step=state.tracker.step,
                plot_fn=None,
            )

    save_sampled(state=state, z=z_in, cross_x=z_cond, writer=writer)
    save_inpainting(state=state, z=z_in, val_idx=val_idx, cross_x=z_cond, writer=writer)


def load(
    accel: at.ml.Accelerator,
    tracker: Tracker,
    seed: int, 
    resume: bool,
    tag: str = "latest",
    dataset: str = vampnet.DATASET,
    save_path: str = vampnet.RUNS_DIR,
    grad_clip_val: float = vampnet.GRAD_CLIP_VAL,
    grad_acc_steps: int = vampnet.GRAD_ACC_STEPS,
    compile: bool = vampnet.COMPILE,
    fine_tune: bool = False,
    model_name: str = None, 
) -> State:
    
    # load the datasets
    train_data, val_data, sample_data = build_datasets(dataset, split=not fine_tune)
    # classlist = train_data.classlist if train_data.class_cond == True else []

    # load the codec
    from vampnet.controls.codec import DACControl, load_codec
    codec = load_codec()
    codec.eval()
    codec.to(accel.device)

    # load model
    model, v_extra = None, {}
    
    if fine_tune:
        print(f"FINE TUNING!")
        print(f"loading model from {model_name}")
        assert model_name is not None, "need to specify a model name when fine tuning"
        model = vampnet.load_model(model_name)
    elif not fine_tune and model_name is not None:
        assert False, "cannot specify a model name when not fine tuning"


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
        

    model = VampNet() if model is None else model
    if compile: 
        print(f"Compiling model")
        model = torch.compile(model)
        print(f"Finished compiling model")

    local_rank = os.getenv("LOCAL_RANK", None)
    use_ddp = torch.cuda.device_count() > 1 and local_rank is not None
    if use_ddp:
        model = accel.prepare_model(model, find_unused_parameters=True if fine_tune else False)
    else:
        model = accel.prepare_model(model)

    assert accel.unwrap(model).vocab_size == codec.quantizer.quantizers[0].codebook_size

    if use_ddp:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=vampnet.LR
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=vampnet.LR)
    scheduler = NoamScheduler(optimizer, d_model=accel.unwrap(model).embedding_dim)
    scheduler.step()

    if "optimizer.pth" in v_extra:
        optimizer.load_state_dict(v_extra["optimizer.pth"])
        scheduler.load_state_dict(v_extra["scheduler.pth"])
    if "tracker.pth" in v_extra:
        tracker.load_state_dict(v_extra["tracker.pth"])
    
    criterion = nn.CrossEntropyLoss()

    # a better rng for sampling from our schedule
    rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed)  

    # log a model summary w/ num params
    if accel.local_rank == 0:
        vampnet.util.add_num_params_repr_hook(accel.unwrap(model))
        with open(f"{save_path}/model.txt", "w") as f:
            f.write(repr(accel.unwrap(model)))

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
        sample_data=sample_data,
        grad_clip_val=grad_clip_val,
        grad_acc_steps=grad_acc_steps,
    )


@record
def train(accel: at.ml.Accelerator,
    resume: bool = vampnet.RESUME, 
    seed: int = vampnet.SEED, 
    save_path: str = vampnet.RUNS_DIR,
    num_iters: int = vampnet.NUM_ITERS,
    save_iters: list = vampnet.SAVE_ITERS,
    sample_freq: int = vampnet.SAMPLE_FREQ, 
    dataset: str = vampnet.DATASET,
    val_freq: int = vampnet.VAL_FREQ,
    batch_size: int = vampnet.BATCH_SIZE,
    val_batch_size: int = vampnet.VAL_BATCH_SIZE,
    val_idx: list = vampnet.VAL_IDX,
    num_workers: int = vampnet.NUM_WORKERS,
    fine_tune: bool = False, 
    model_name: str = None, 
    cli: bool = False, 
):


    # Enable cudnn autotuner to speed up training
    # (can be altered by the funcs.seed function)
    torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
    # Uncomment to trade memory for speed.

    # Install to make things look nice
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pretty.install()
    install()

    if seed is None:
        seed = time.time_ns() % 2**32

    print(f"seed: {seed}")
    seed = seed + accel.local_rank
    at.util.seed(seed)
    writer = None

    Path(save_path).mkdir(parents=True, exist_ok=True)

    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir=f"{save_path}/logs/")


    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/tracker.txt", 
        rank=accel.local_rank
    )

    # load the codec model
    state: State = load(
        accel=accel, 
        tracker=tracker,
        seed=seed,
        dataset=dataset,
        resume=resume, 
        save_path=save_path,
        fine_tune=fine_tune,
        model_name=model_name)
    print("initialized state.")

    # need to throw if the batch size is bigger than 
    # the dataset size
    # assert len(state.train_data) > batch_size, "Batch size is larger than the dataset size"
    # trim batch size to the dataset size
    batch_size = min(batch_size, len(state.train_data))
    print(f"trimmed batch size: {batch_size}")
    val_batch_size = min(val_batch_size, len(state.val_data))
    print(f"trimmed val batch size: {val_batch_size}")

    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        prefetch_factor=2 if num_workers > 0 else None,
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
    # first_iter = True
    with tracker.live:
        print(f"tracker opened")
        done = False

        while not done:
            for tracker.step, batch in enumerate(train_dataloader, start=tracker.step): 
                train_loop(state, batch, accel)
                    
                last_iter = (
                    tracker.step == num_iters - 1 if num_iters is not None else False
                )

                # if tracker.step == 0 or first_iter:
                #     first_iter = False
                #     continue

                if tracker.step % val_freq == 0 or last_iter:
                    tracker.print(f"Validating at iteration {tracker.step}")
                    validate(state, val_dataloader, accel)

                    print(f"Saving checkpoint at iteration {tracker.step}")
                    checkpoint(
                        state=state, 
                        save_iters=save_iters,
                        save_path=save_path, 
                        fine_tune=fine_tune)
                    print(f"checkpoint done")

                if tracker.step % sample_freq == 0 or last_iter:
                    tracker.print(f"Saving samples at iteration {tracker.step}")
                    save_samples(state, val_idx, writer)

                    # Reset validation progress bar, print summary since last validation.
                    tracker.done("val", f"Iteration {tracker.step}")

                if last_iter:
                    print(f"Finished training at iteration {tracker.step}")
                    done = True
                    break
        
    # return an interface with the codec and model
    return state.interface

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
        unmasked_target = target.masked_fill(flat_mask.bool(), vampnet.IGNORE_INDEX)
        masked_target = target.masked_fill(~flat_mask.bool(), vampnet.IGNORE_INDEX)

        assert target.shape[0] == r.shape[0]
        # grab the indices of the r values that are in the range
        r_idx = (r >= r_range[0]) & (r < r_range[1])

        # grab the target and z_hat values that are in the range
        r_unmasked_target = unmasked_target[r_idx]
        r_masked_target = masked_target[r_idx]
        r_z_hat = z_hat[r_idx]

        for topk in (25,):
            s, e = r_range
            tag = f"accuracy-{s}-{e}/top{topk}"

            output[f"{tag}/unmasked"] = accuracy(
                preds=r_z_hat,
                target=r_unmasked_target,
                ignore_index=vampnet.IGNORE_INDEX,
                top_k=topk,
            )
            output[f"{tag}/masked"] = accuracy(
                preds=r_z_hat,
                target=r_masked_target,
                ignore_index=vampnet.IGNORE_INDEX,
                top_k=topk,
            )


if __name__ == "__main__":
    import yapecs 

    parser = yapecs.ArgumentParser()
    # add a --resume flag
    parser.add_argument("--resume", action="store_true", help="resume training from the latest checkpoint")
    args = parser.parse_args()

    with at.ml.Accelerator(amp=vampnet.AMP) as accel:
        if accel.local_rank != 0:
            sys.tracebacklimit = 0
        train(accel, resume=args.resume, cli=True)


