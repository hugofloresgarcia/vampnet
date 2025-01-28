import os
import warnings
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import ConcatDataset
from audiotools.data.datasets import AudioLoader
from rich import pretty
from rich.traceback import install
from torch.utils.tensorboard import SummaryWriter

from vampnet.lac

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Install to make things look nice
warnings.filterwarnings("ignore", category=UserWarning)
pretty.install()
install()

# Optimizers
AdamW = argbind.bind(lac.nn.optimizer.AdamW, "generator", "discriminator")
ExponentialLR = argbind.bind(lac.nn.optimizer.ExponentialLR, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

# Models
LAC = argbind.bind(lac.model.LAC)
Discriminator = argbind.bind(lac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(lac.nn.loss, filter_fn=filter_fn)


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@torch.no_grad()
def apply_transform(transform_fn, batch):
    clean: AudioSignal = batch["signal"]
    kwargs = batch["transform_args"]
    noisy: AudioSignal = transform_fn(clean.clone(), **kwargs)
    with transform_fn.filter("preprocess", "postprocess"):
        clean = transform_fn(clean.clone(), **kwargs)
    return clean, noisy


@argbind.bind("train", "val")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


def build_datasets(args, sample_rate: int):
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)
    return train_data, val_data


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        if (Path(kwargs["folder"]) / "generator").exists():
            generator, g_extra = LAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = LAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    trainer_state = {"state_dict": None, "start_idx": 0}

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "trainer.pth" in g_extra:
        trainer_state = g_extra["trainer.pth"]

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    return {
        "generator": generator,
        "discriminator": discriminator,
        "optimizer_g": optimizer_g,
        "scheduler_g": scheduler_g,
        "optimizer_d": optimizer_d,
        "scheduler_d": scheduler_d,
        "trainer_state": trainer_state,
    }


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_epochs: int = 3000,
    save_epochs: list = [10, 50, 100, 250, 500],
    epoch_length: int = 1000,
    batch_size: int = 70,
    val_batch_size: int = 100,
    num_workers: int = 40,
    detect_anomaly: bool = False,
    save_audio_epochs: int = 10,
    val_idx: list = [0, 1, 2, 3, 4],
    quiet: bool = False,
    record_memory: bool = False,
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    util.seed(seed)
    writer = None

    if accel.local_rank == 0:
        writer = SummaryWriter(log_dir=f"{save_path}/logs")
        argbind.dump_args(args, f"{save_path}/args.yml")

    loaded = load(args, accel, save_path)
    generator = loaded["generator"]
    discriminator = loaded["discriminator"]
    optimizer_g = loaded["optimizer_g"]
    scheduler_g = loaded["scheduler_g"]
    optimizer_d = loaded["optimizer_d"]
    scheduler_d = loaded["scheduler_d"]
    trainer_state = loaded["trainer_state"]

    sample_rate = accel.unwrap(generator).sample_rate

    train_data, val_data = build_datasets(args, sample_rate)
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
        batch_size=val_batch_size,
        collate_fn=val_data.collate,
    )

    l1_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    class Trainer(ml.BaseTrainer):
        def train_loop(self, engine, batch):
            generator.train()
            discriminator.train()
            output = {}
            batch = util.prepare_batch(batch, accel.device)
            clean, noisy = apply_transform(train_data.transform, batch)

            with accel.autocast():
                out = generator(noisy.audio_data, noisy.sample_rate)
                # Pop keys that are not losses, if they are there
                for k in ["z", "codes", "latents"]:
                    out.pop(k, None)
                enhanced = AudioSignal(out.pop("audio"), sample_rate)

            with accel.autocast():
                output["adv/disc_loss"] = gan_loss.discriminator_loss(enhanced, clean)

            optimizer_d.zero_grad()
            accel.backward(output["adv/disc_loss"])
            accel.scaler.unscale_(optimizer_d)
            output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(), 10.0
            )
            accel.step(optimizer_d)
            scheduler_d.step()

            with accel.autocast():
                output["stft/loss"] = stft_loss(enhanced, clean)
                output["mel/loss"] = mel_loss(enhanced, clean)
                output["waveform/loss"] = l1_loss(enhanced, clean)
                (
                    output["adv/gen_loss"],
                    output["adv/feat_loss"],
                ) = gan_loss.generator_loss(enhanced, clean)
                output.update(out)
                output["loss"] = sum(
                    [v * output[k] for k, v in lambdas.items() if k in output]
                )

            optimizer_g.zero_grad()
            accel.backward(output["loss"])
            accel.scaler.unscale_(optimizer_g)
            output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
                generator.parameters(), 1e3
            )
            accel.step(optimizer_g)
            scheduler_g.step()
            accel.update()

            output["stft/nz"] = stft_loss(noisy, clean)
            output["stft/imp"] = output["stft/nz"] - output["stft/loss"]
            output["other/learning_rate"] = optimizer_g.param_groups[0]["lr"]
            output["other/batch_size"] = noisy.batch_size * accel.world_size

            return {k: v for k, v in sorted(output.items())}

        @torch.no_grad()
        def val_loop(self, engine, batch):
            generator.eval()
            batch = util.prepare_batch(batch, accel.device)
            clean, noisy = apply_transform(val_data.transform, batch)

            enhanced = generator(noisy.audio_data, noisy.sample_rate)["audio"]
            enhanced = AudioSignal(enhanced, sample_rate)
            stft_loss_val = stft_loss(enhanced, clean)
            stft_nz = stft_loss(noisy, clean)
            mel_loss_val = mel_loss(enhanced, clean)

            return {
                "loss": mel_loss_val,
                "mel/loss": mel_loss_val,
                "stft/imp": stft_nz - stft_loss_val,
                "stft/loss": stft_loss_val,
                "stft/nz": stft_nz,
                "waveform/loss": l1_loss(enhanced, clean),
            }

        def checkpoint(self, engine):
            metadata = {"logs": dict(engine.state.logs["epoch"])}

            if self.state.epoch % save_audio_epochs == 0 or self.state.epoch == 1:
                self.save_samples()

            tags = ["latest"]
            loss_key = "loss/val" if "loss/val" in metadata["logs"] else "loss/train"
            self.print(f"Saving to {str(Path('.').absolute())}")
            if self.is_best(engine, loss_key):
                self.print(f"Best generator so far")
                tags.append("best")
            if self.state.epoch in save_epochs:
                tags.append(f"{self.state.epoch}k")

            for tag in tags:
                generator_extra = {
                    "optimizer.pth": optimizer_g.state_dict(),
                    "scheduler.pth": scheduler_g.state_dict(),
                    "trainer.pth": {
                        "state_dict": self.state_dict(),
                        "start_idx": self.state.iteration * batch_size,
                    },
                    "metadata.pth": metadata,
                }
                accel.unwrap(generator).metadata = metadata
                accel.unwrap(generator).save_to_folder(
                    f"{save_path}/{tag}", generator_extra
                )
                discriminator_extra = {
                    "optimizer.pth": optimizer_d.state_dict(),
                    "scheduler.pth": scheduler_d.state_dict(),
                }
                accel.unwrap(discriminator).save_to_folder(
                    f"{save_path}/{tag}", discriminator_extra
                )

        def after_epoch(self, engine):
            # Consolidate optimizers to rank 0 if using ZeroRedundancyOptimizer.
            if hasattr(optimizer_g, "consolidate_state_dict"):
                optimizer_g.consolidate_state_dict()
                optimizer_d.consolidate_state_dict()

        @torch.no_grad()
        def save_samples(self):
            self.print("Saving audio samples to tensorboard")
            generator.eval()

            samples = [val_data[idx] for idx in val_idx]
            batch = val_data.collate(samples)
            batch = util.prepare_batch(batch, accel.device)
            clean, noisy = apply_transform(val_data.transform, batch)

            enhanced = generator(noisy.audio_data, noisy.sample_rate)["audio"]
            enhanced = AudioSignal(enhanced, noisy.sample_rate)

            audio_dict = {"generated": enhanced}
            if self.state.epoch == 1:
                audio_dict["clean"] = clean
                audio_dict["noisy"] = noisy

            for k, v in audio_dict.items():
                for nb in range(v.batch_size):
                    v[nb].cpu().write_audio_to_tb(
                        f"{k}/sample_{nb}.wav",
                        self.writer,
                        step=self.state.epoch,
                    )

    trainer = Trainer(
        writer=writer,
        rank=accel.local_rank,
        quiet=quiet,
        log_file=f"{save_path}/log.txt",
        record_memory=record_memory,
    )

    if trainer_state["state_dict"] is not None:
        trainer.load_state_dict(trainer_state["state_dict"])
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(trainer.trainer.state.epoch)

    trainer.run(
        train_dataloader,
        val_dataloader,
        num_epochs=num_epochs,
        epoch_length=epoch_length,
        detect_anomaly=detect_anomaly,
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            train(args, accel)
