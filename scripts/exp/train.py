import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Optional

import argbind
import audiotools as at
import torch
import torch.nn as nn
from audiotools import AudioSignal
from audiotools.data import transforms
from einops import rearrange
from rich import pretty
from rich.traceback import install
from tensorboardX import SummaryWriter

import vampnet
from vampnet.modules.transformer import VampNet
from lac.model.lac import LAC


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
        tfm.VolumeNorm(("uniform", -32, -14)),
        tfm.VolumeChange(("uniform", -6, 3)),
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
    with argbind.scope(args, "test"):
        test_data = AudioDataset(
            AudioLoader(), sample_rate, transform=build_transform()
        )
    return train_data, val_data, test_data


def rand_float(shape, low, high, rng):
    return rng.draw(shape)[:, 0] * (high - low) + low


def flip_coin(shape, p, rng):
    return rng.draw(shape)[:, 0] < p


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: at.ml.Accelerator,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    model, v_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        if (Path(kwargs["folder"]) / "vampnet").exists():
            model, v_extra = VampNet.load_from_folder(**kwargs)

    codec = LAC.load(args["codec_ckpt"], map_location="cpu")
    codec.eval()
    model = VampNet() if model is None else model

    model = accel.prepare_model(model)

    # assert accel.unwrap(model).n_codebooks == codec.quantizer.n_codebooks
    assert (
        accel.unwrap(model).vocab_size == codec.quantizer.quantizers[0].codebook_size
    )

    optimizer = AdamW(model.parameters(), use_zero=accel.use_ddp)
    scheduler = NoamScheduler(optimizer, d_model=accel.unwrap(model).embedding_dim)
    scheduler.step()

    trainer_state = {"state_dict": None, "start_idx": 0}

    if "optimizer.pth" in v_extra:
        optimizer.load_state_dict(v_extra["optimizer.pth"])
    if "scheduler.pth" in v_extra:
        scheduler.load_state_dict(v_extra["scheduler.pth"])
    if "trainer.pth" in v_extra:
        trainer_state = v_extra["trainer.pth"]

    return {
        "model": model,
        "codec": codec,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer_state": trainer_state,
    }


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    gpu_memory_map = {f"gpu/{k}": v / 1024 for k, v in gpu_memory_map.items()}
    return gpu_memory_map


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
    **kwargs,
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

def sample_prefix_suffix_amt(
        n_batch, 
        prefix_amt, 
        suffix_amt, 
        prefix_dropout, 
        suffix_dropout, 
        rng 
    ):
    """
    Sample the number of prefix and suffix tokens to drop.
    """
    if prefix_amt > 0.0:
        prefix_mask = flip_coin(n_batch, 1 - prefix_dropout, rng)
        n_prefix = int(prefix_amt * z.shape[-1]) * prefix_mask
    else:
        n_prefix = None
    if suffix_amt > 0.0:
        suffix_mask = flip_coin(n_batch, 1 - suffix_dropout, rng)
        n_suffix = int(suffix_amt * z.shape[-1]) * suffix_mask
    else:
        n_suffix = None
    return n_prefix, n_suffix


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: at.ml.Accelerator,
    codec_ckpt: str = None,
    seed: int = 0,
    save_path: str = "ckpt",
    max_epochs: int = int(100e3),
    epoch_length: int = 1000,
    save_audio_epochs: int = 10,
    batch_size: int = 48,
    grad_acc_steps: int = 1,
    val_idx: list = [0, 1, 2, 3, 4],
    num_workers: int = 20,
    detect_anomaly: bool = False,
    grad_clip_val: float = 5.0,
    prefix_amt: float = 0.0,
    suffix_amt: float = 0.0,
    prefix_dropout: float = 0.1,
    suffix_dropout: float = 0.1,
    quiet: bool = False,
):
    assert codec_ckpt is not None, "codec_ckpt is required"

    at.util.seed(seed)
    writer = None

    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir=f"{save_path}/logs/")
        argbind.dump_args(args, f"{save_path}/args.yml")

    # load the codec model
    loaded = load(args, accel, save_path)
    model = loaded["model"]
    codec = loaded["codec"]
    optimizer = loaded["optimizer"]
    scheduler = loaded["scheduler"]
    trainer_state = loaded["trainer_state"]

    sample_rate = codec.sample_rate

    # a better rng for sampling from our schedule
    rng = torch.quasirandom.SobolEngine(1, scramble=True)  

    # log a model summary w/ num params
    if accel.local_rank == 0:
        add_num_params_repr_hook(accel.unwrap(model))
        with open(f"{save_path}/model.txt", "w") as f:
            f.write(repr(accel.unwrap(model)))

    # load the datasets
    train_data, val_data, _ = build_datasets(args, sample_rate)
    train_dataloader = accel.prepare_dataloader(
        train_data,
        start_idx=trainer_state["start_idx"],
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=train_data.collate,
    )
    val_dataloader = accel.prepare_dataloader(
        val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=val_data.collate,
    )

    criterion = CrossEntropyLoss()

    class Trainer(at.ml.BaseTrainer):
        _last_grad_norm = 0.0

        def _metrics(self, vn, z_hat, r, target, flat_mask, output):
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
                        task="multiclass",
                        num_classes=vn.vocab_size,
                    )
                    output[f"{tag}/masked"] = accuracy(
                        preds=r_z_hat,
                        target=r_masked_target,
                        ignore_index=IGNORE_INDEX,
                        top_k=topk,
                        task="multiclass",
                        num_classes=vn.vocab_size,
                    )

        def train_loop(self, engine, batch):
            model.train()
            batch = at.util.prepare_batch(batch, accel.device)
            signal = apply_transform(train_data.transform, batch)

            output = {}
            vn = accel.unwrap(model)
            with accel.autocast():
                with torch.inference_mode():
                    codec.to(accel.device)
                    z = codec.encode(signal.samples, signal.sample_rate)["codes"]
                    z = z[:, : vn.n_codebooks, :]

                n_batch = z.shape[0]
                r = rng.draw(n_batch)[:, 0].to(accel.device)

                n_prefix, n_suffix = sample_prefix_suffix_amt(
                    n_batch=n_batch, prefix_amt=prefix_amt, suffix_amt=suffix_amt,
                    prefix_dropout=prefix_dropout, suffix_dropout=suffix_dropout,
                    rng=rng
                )

                z_mask, mask = vn.add_noise(
                    z, r, n_prefix=n_prefix, n_suffix=n_suffix
                )
                z_mask_latent = vn.embedding.from_codes(z_mask, codec)

                dtype = torch.bfloat16 if accel.amp else None
                with accel.autocast(dtype=dtype):
                    z_hat = model(z_mask_latent, r)
                    # for mask mode
                    z_hat = vn.add_truth_to_logits(z, z_hat, mask)

                target = vn.embedding.flatten(
                    z[:, vn.n_conditioning_codebooks :, :],
                    n_codebooks=vn.n_predict_codebooks,
                )

                flat_mask = vn.embedding.flatten(
                    mask[:, vn.n_conditioning_codebooks :, :],
                    n_codebooks=vn.n_predict_codebooks,
                )

                if vn.noise_mode == "mask":
                    # replace target with ignore index for masked tokens
                    t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
                    output["loss"] = criterion(z_hat, t_masked)
                else:
                    output["loss"] = criterion(z_hat, target)

                self._metrics(
                    vn=vn,
                    r=r,
                    z_hat=z_hat,
                    target=target,
                    flat_mask=flat_mask,
                    output=output,
                )

            
            accel.backward(output["loss"] / grad_acc_steps)

            output["other/learning_rate"] = optimizer.param_groups[0]["lr"]
            output["other/batch_size"] = z.shape[0]

            output.update(get_gpu_memory_map())

            if (
                (engine.state.iteration % grad_acc_steps == 0)
                or (engine.state.iteration % epoch_length == 0)
                or (engine.state.iteration % epoch_length == 1)
            ):  # (or we reached the end of the epoch)
                accel.scaler.unscale_(optimizer)
                output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_val
                )
                self._last_grad_norm = output["other/grad_norm"]

                accel.step(optimizer)
                optimizer.zero_grad()

                scheduler.step()
                accel.update()
            else:
                output["other/grad_norm"] = self._last_grad_norm

            return {k: v for k, v in sorted(output.items())}

        @torch.no_grad()
        def val_loop(self, engine, batch):
            model.eval()
            codec.eval()
            batch = at.util.prepare_batch(batch, accel.device)
            signal = apply_transform(val_data.transform, batch)

            vn = accel.unwrap(model)
            z = codec.encode(signal.samples, signal.sample_rate)["codes"]
            z = z[:, : vn.n_codebooks, :]

            n_batch = z.shape[0]
            r = rng.draw(n_batch)[:, 0].to(accel.device)

            n_prefix, n_suffix = sample_prefix_suffix_amt(
                n_batch=n_batch, prefix_amt=prefix_amt, suffix_amt=suffix_amt,
                prefix_dropout=prefix_dropout, suffix_dropout=suffix_dropout,
                rng=rng
            )

            z_mask, mask = vn.add_noise(z, r, n_prefix=n_prefix, n_suffix=n_suffix)
            z_mask_latent = vn.embedding.from_codes(z_mask, codec)

            z_hat = model(z_mask_latent, r)
            # for mask mode
            z_hat = vn.add_truth_to_logits(z, z_hat, mask)

            target = vn.embedding.flatten(
                z[:, vn.n_conditioning_codebooks :, :],
                n_codebooks=vn.n_predict_codebooks,
            )

            flat_mask = vn.embedding.flatten(
                mask[:, vn.n_conditioning_codebooks :, :],
                n_codebooks=vn.n_predict_codebooks,
            )

            output = {}
            if vn.noise_mode == "mask":
                # replace target with ignore index for masked tokens
                t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
                output["loss"] = criterion(z_hat, t_masked)
            else:
                output["loss"] = criterion(z_hat, target)

            self._metrics(
                vn=vn,
                r=r,
                z_hat=z_hat,
                target=target,
                flat_mask=flat_mask,
                output=output,
            )

            return output

        def checkpoint(self, engine):
            if accel.local_rank != 0:
                print(f"ERROR:Skipping checkpoint on rank {accel.local_rank}")
                return

            metadata = {"logs": dict(engine.state.logs["epoch"])}

            if self.state.epoch % save_audio_epochs == 0:
                self.save_samples()

            tags = ["latest"]
            loss_key = "loss/val" if "loss/val" in metadata["logs"] else "loss/train"
            self.print(f"Saving to {str(Path('.').absolute())}")

            if self.is_best(engine, loss_key):
                self.print(f"Best model so far")
                tags.append("best")

            for tag in tags:
                model_extra = {
                    "optimizer.pth": optimizer.state_dict(),
                    "scheduler.pth": scheduler.state_dict(),
                    "trainer.pth": {
                        "start_idx": self.state.iteration * batch_size,
                        "state_dict": self.state_dict(),
                    },
                    "metadata.pth": metadata,
                }

                accel.unwrap(model).metadata = metadata
                accel.unwrap(model).save_to_folder(
                    f"{save_path}/{tag}", model_extra
                )

        def save_sampled(self, z):
            num_samples = z.shape[0]

            for i in range(num_samples):
                sampled = accel.unwrap(model).sample(
                    codec=codec,
                    time_steps=z.shape[-1],
                    start_tokens=z[i : i + 1],
                )
                sampled.cpu().write_audio_to_tb(
                    f"sampled/{i}",
                    self.writer,
                    step=self.state.epoch,
                    plot_fn=None,
                )

        def save_imputation(self, z: torch.Tensor):
            # imputations
            mask_begin = z.shape[-1] // 4
            mask_end = (z.shape[-1] * 3) // 4

            imp_mask = torch.zeros(z.shape[0], z.shape[-1]).to(accel.device).int()
            imp_mask[:, mask_begin:mask_end] = 1

            imp_noisy = (
                z * (1 - imp_mask[:, None, :])
                + torch.randint_like(z, 0, accel.unwrap(model).vocab_size)
                * imp_mask[:, None, :]
            )
            imputed_noisy = accel.unwrap(model).to_signal(imp_noisy, codec)
            imputed_true = accel.unwrap(model).to_signal(z, codec)

            imputed = []
            for i in range(len(z)):
                imputed.append(
                    accel.unwrap(model).sample(
                        codec=codec,
                        time_steps=z.shape[-1],
                        start_tokens=z[i][None, ...],
                        mask=imp_mask[i][None, ...],
                    )
                )
            imputed = AudioSignal.batch(imputed)

            for i in range(len(val_idx)):
                imputed_noisy[i].cpu().write_audio_to_tb(
                    f"imputed_noisy/{i}",
                    self.writer,
                    step=self.state.epoch,
                    plot_fn=None,
                )
                imputed[i].cpu().write_audio_to_tb(
                    f"imputed/{i}",
                    self.writer,
                    step=self.state.epoch,
                    plot_fn=None,
                )
                imputed_true[i].cpu().write_audio_to_tb(
                    f"imputed_true/{i}",
                    self.writer,
                    step=self.state.epoch,
                    plot_fn=None,
                )

        @torch.no_grad()
        def save_samples(self):
            model.eval()
            codec.eval()
            vn = accel.unwrap(model)

            batch = [val_data[i] for i in val_idx]
            batch = at.util.prepare_batch(val_data.collate(batch), accel.device)

            signal = apply_transform(val_data.transform, batch)

            z = codec.encode(signal.samples, signal.sample_rate)["codes"]
            z = z[:, : vn.n_codebooks, :]

            r = torch.linspace(0.1, 0.95, len(val_idx)).to(accel.device)

            n_batch = z.shape[0]

            n_prefix, n_suffix = sample_prefix_suffix_amt(
                n_batch=n_batch, prefix_amt=prefix_amt, suffix_amt=suffix_amt,
                prefix_dropout=prefix_dropout, suffix_dropout=suffix_dropout,
                rng=rng
            )

            z_mask, mask = vn.add_noise(z, r, n_prefix=n_prefix, n_suffix=n_suffix)
            z_mask_latent = vn.embedding.from_codes(z_mask, codec)

            z_hat = model(z_mask_latent, r)
            # for mask mode
            z_hat = vn.add_truth_to_logits(z, z_hat, mask)

            z_pred = torch.softmax(z_hat, dim=1).argmax(dim=1)
            z_pred = vn.embedding.unflatten(z_pred, n_codebooks=vn.n_predict_codebooks)
            z_pred = torch.cat([z[:, : vn.n_conditioning_codebooks, :], z_pred], dim=1)

            print("z_mask", z_mask.shape)
            generated = vn.to_signal(z_pred, codec)
            reconstructed = vn.to_signal(z, codec)
            masked = vn.to_signal(z_mask.squeeze(1), codec)

            for i in range(generated.batch_size):
                audio_dict = {
                    "original": signal[i],
                    "masked": masked[i],
                    "generated": generated[i],
                    "reconstructed": reconstructed[i],
                }
                for k, v in audio_dict.items():
                    v.cpu().write_audio_to_tb(
                        f"samples/_{i}.r={r[i]:0.2f}/{k}",
                        self.writer,
                        step=self.state.epoch,
                        plot_fn=None,
                    )

            self.save_sampled(z)
            self.save_imputation(z)

    trainer = Trainer(writer=writer, quiet=quiet)

    if trainer_state["state_dict"] is not None:
        trainer.load_state_dict(trainer_state["state_dict"])
    if hasattr(train_dataloader.sampler, "set_epoch"):
        train_dataloader.sampler.set_epoch(trainer.trainer.state.epoch)

    trainer.run(
        train_dataloader,
        val_dataloader,
        num_epochs=max_epochs,
        epoch_length=epoch_length,
        detect_anomaly=detect_anomaly,
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            train(args, accel)
