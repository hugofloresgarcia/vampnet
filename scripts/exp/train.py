import os
import sys
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import argbind
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import loralib as lora
import torch._dynamo
torch._dynamo.config.verbose=True
import lightning as L

import vampnet
from vampnet.modules.transformer import VampNet
from vampnet.util import codebook_unflatten, codebook_flatten
from vampnet import mask as pmask
from vampnet.dac.model.dac import DAC
import vampnet.signal as sn

import soundmaterial as sm

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# optim
CrossEntropyLoss = argbind.bind(nn.CrossEntropyLoss)
AdamW = argbind.bind(torch.optim.AdamW)
NoamScheduler = argbind.bind(vampnet.scheduler.NoamScheduler)

# model
VampNet = argbind.bind(VampNet)

# Data
Dataset = argbind.bind(sm.dataset.Dataset, "train", "val")

IGNORE_INDEX = -100

def build_transform():
    def transform(sig):
        sig = sn.normalize(sig, -24.0)
        return sig
    return transform


@argbind.bind(without_prefix=True)
def get_checkpoint_path(resume_ckpt: str = None):
    print("~~~~")
    print(f"resuming from {resume_ckpt}" if resume_ckpt else "~~starting from scratch!!")
    print("~~~~")
    return resume_ckpt


def build_datasets(
    sample_rate: int,
    db_path: str = None, 
    query: str = "SELECT * from audio_file", 
):    
    assert db_path is not None, "db_path is required"
    train_tfm = build_transform()    
    val_tfm = build_transform()   

    import pandas as pd
    conn = sm.connect(db_path)
    print(f"loading data from {db_path}")

    df = pd.read_sql(query, conn)
    tdf, vdf = sm.dataset.train_test_split(df, test_size=0.1, seed=42)

    train_data = Dataset(
        tdf, sample_rate=sample_rate, transform=train_tfm
    )
    val_data = Dataset(
        vdf, sample_rate=sample_rate, transform=val_tfm, max_examples=2000
    )
    return train_data, val_data

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

from lightning.pytorch.callbacks import Callback
class AudioSampleLoggingCallback(Callback):

    def __init__(self, dataset: Dataset, num_samples: int = 10):
        self.dataset = dataset
        self.num_samples = num_samples


    @torch.inference_mode()
    def _save_onesteps(self, module):
        rs = torch.linspace(0, 1, self.num_samples+1)[1:]
        for i in range(self.num_samples):
            sig = self.dataset[i]["sig"].to(module.device)
            z = module.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : module.model.n_codebooks, :]

            n_batch = z.shape[0]
            r = rs[i] * torch.ones(n_batch).to(z.device)

            mask = pmask.random(z, r)
            mask = pmask.codebook_unmask(mask, module.model.n_conditioning_codebooks)
            z_mask = pmask.apply_mask(z, mask, module.model.mask_token)

            z_hat = module.model(z_mask)
            # argmax sample
            z_hat = z_hat.argmax(dim=-2)
            z_hat = codebook_unflatten(z_hat, module.model.n_codebooks)
            # replace masked with original
            z_hat = torch.where(~mask.bool(), z, z_hat)

            outwav = module.codec.decode(
                module.codec.quantizer.from_codes(z_hat)[0]
            )
            recons = module.codec.decode(
                module.codec.quantizer.from_codes(z)[0]
            )
            trainer.logger.experiment.add_audio(
                f"orig/{i}",
                sig.wav[0][0],
                global_step=trainer.global_step,
                sample_rate=sig.sr,
            )

            trainer.logger.experiment.add_audio(
                f"recons/{i}-r={r[0]:.2f}",
                recons[0][0],
                global_step=trainer.global_step,
                sample_rate=sig.sr,
            )

            trainer.logger.experiment.add_audio(
                f"sampled/{i}-r={r[0]:.2f}",
                outwav[0][0],
                global_step=trainer.global_step,
                sample_rate=sig.sr,
            )


            
    @torch.inference_mode()
    def _save_generations(self, module):
        for i in range(self.num_samples):
            sig = self.dataset[i]["sig"].to(module.device)

            z = module.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : module.model.n_codebooks, :]

            mask = pmask.full_mask(z)
            z_mask = pmask.apply_mask(z, mask, module.model.mask_token)

            z_hat = module.model.generate(z_mask)

            outwav = module.codec.decode(
                module.codec.quantizer.from_codes(z_hat)[0]
            )
            trainer.logger.experiment.add_audio(
                f"generated/{i}",
                outwav[0][0],
                global_step=trainer.global_step,
                sample_rate=sig.sr,
            )

    @torch.inference_mode()
    def _save_periodic_prompt(self, module):
        periods = [3, 5, 7, 11, 13, 21]
        for i in range(self.num_samples):
            sig = self.dataset[i]["sig"].to(module.device)

            period = periods[i % len(periods)]

            z = module.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : module.model.n_codebooks, :]

            mask = pmask.periodic_mask(z, period, 1, random_roll=True)
            z_mask = pmask.apply_mask(z, mask, module.model.mask_token)

            z_hat = module.model.generate(z_mask)

            outwav = module.codec.decode(
                module.codec.quantizer.from_codes(z_hat)[0]
            )
            trainer.logger.experiment.add_audio(
                f"periodic/{i}-p={period}",
                outwav[0][0],
                global_step=trainer.global_step,
                sample_rate=sig.sr,
            )

    def on_validation_epoch_end(self, trainer, module):
        print(f"saving samples at step {trainer.global_step}")
        module.eval()
        self._save_onesteps(module)
        self._save_generations(module)
        self._save_periodic_prompt(module)


class VampNetTrainer(L.LightningModule):

    def __init__(self, codec_ckpt: str):
        super().__init__()
        self.codec = DAC.load(codec_ckpt, map_location="cpu")
        self.codec.eval()
        self.codec = torch.compile(self.codec)

        self.model = VampNet()
        # TODOO: remove, use embedding instead
        # self.model.embedding.quantizer = self.codec.quantizer
        # self.model = torch.compile(self.model)

        self.criterion = CrossEntropyLoss()
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=42)

        assert (
            self.model.vocab_size == self.codec.quantizer.quantizers[0].codebook_size
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters())
        scheduler = NoamScheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def training_step(self, batch, batch_idx):
        self.model.train()
        batch = sn.prepare_batch(batch, self.device)
        sig = batch["sig"]

        output = {}
        vn = self.model
        with torch.inference_mode():
            self.codec.to(self.device)
            z = self.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : vn.n_codebooks, :]

        n_batch = z.shape[0]
        r = self.rng.draw(n_batch)[:, 0].to(self.device)

        mask = pmask.random(z, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
        z_mask = pmask.apply_mask(z, mask, vn.mask_token)
        
        # TODOO: use embedding instead
        # z_mask_latent = vn.embedding.from_codes(z_mask)
        z_hat = self.model(z_mask)

        target = codebook_flatten(
            z[:, vn.n_conditioning_codebooks :, :],
        )

        flat_mask = codebook_flatten(
            mask[:, vn.n_conditioning_codebooks :, :],
        )

        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
        output["loss"] = self.criterion(z_hat, t_masked)

        _metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=flat_mask,
            output=output,
        )

        self.log("loss/train", output["loss"], on_step=True, prog_bar=True)

        return output["loss"]

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.codec.eval()
        batch = sn.prepare_batch(batch, self.device)
        sig = batch["sig"]

        vn = self.model
        z = self.codec.encode(sig.wav, sig.sr)["codes"]
        z = z[:, : vn.n_codebooks, :]

        n_batch = z.shape[0]
        r = self.rng.draw(n_batch)[:, 0].to(self.device)

        mask = pmask.random(z, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)
        z_mask = pmask.apply_mask(z, mask, vn.mask_token)

        # TODOO: use embedding instead
        # z_mask_latent = vn.embedding.from_codes(z_mask)

        z_hat = self.model(z_mask)

        target = codebook_flatten(
            z[:, vn.n_conditioning_codebooks :, :],
        )

        flat_mask = codebook_flatten(
            mask[:, vn.n_conditioning_codebooks :, :]
        )

        output = {}
        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
        output["loss"] = self.criterion(z_hat, t_masked)

        _metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=flat_mask,
            output=output,
        )

        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True)
        
        return output["loss"]

@argbind.bind(without_prefix=True)
def prepare_dataloaders(train_data, val_data, batch_size=16, num_workers=4):
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=Dataset.collate)
    val_dataloader = torch.utils.data.DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=Dataset.collate)

    return train_dataloader, val_dataloader

if __name__ == "__main__":
    build_datasets = argbind.bind(build_datasets)

    args = argbind.parse_args()
    with argbind.scope(args):
        model = VampNetTrainer(args["codec_ckpt"])

        train_data, val_data = build_datasets(model.codec.sample_rate)
        train_dataloader, val_dataloader = prepare_dataloaders(train_data, val_data)

        callbacks = []
        callbacks.append(AudioSampleLoggingCallback(val_data))

        # todo setup datasets and dataloaders
        trainer = L.Trainer(
            devices=1,
            default_root_dir="runs/debug",
            max_epochs=-1,
            limit_val_batches=20, 
            gradient_clip_val=5.0,
            val_check_interval=1000,
            callbacks=callbacks,
            # precision="bf16"
        )

        trainer.fit(
            model=model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader, 
            ckpt_path=get_checkpoint_path()
        )