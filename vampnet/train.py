import os
import sys
import warnings
from pathlib import Path
from typing import Optional
import random
from dataclasses import dataclass
import copy
import os

import argbind
import torch
import torch.nn as nn
from einops import rearrange
import pandas as pd
from huggingface_hub import PyTorchModelHubMixin

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import Callback

import soundmaterial as sm
from soundmaterial.dataset import Dataset

import vampnet
from vampnet.control import Sketch2SoundController
from vampnet.modules.transformer import VampNet
from vampnet.util import codebook_unflatten, codebook_flatten, flip_coin
from vampnet import mask as pmask
from vampnet.dac.model.dac import DAC
import vampnet.dsp.signal as sn


# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

IGNORE_INDEX = -100
CODEC_CKPT = "~/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"
SEED = 1

DEFAULT_QUERY = """
    SELECT af.path, chunk.offset, chunk.duration, af.duration as total_duration, dataset.name 
    FROM chunk 
    JOIN audio_file as af ON chunk.audio_file_id = af.id 
    JOIN dataset ON af.dataset_id = dataset.id
"""

class VampNetTrainer(L.LightningModule, PyTorchModelHubMixin):

    def __init__(self,
        prefix_tag: str = "",
        # ~~~ codec ~~~
        codec_ckpt: str = CODEC_CKPT,
        # ~~~ model ~~~
        mode: str = "stemgen",
        ctrl_keys: tuple[str] = ("rms", ),
        n_heads: int = 12, 
        n_layers: int = 12, 
        embedding_dim: int = 1026,
        # ~~~ training // behavior ~~~
        outpaint_prob: float = 0.0, 
        prefix_min: float = 0.1, 
        prefix_max: float = 0.25,
        lr: float = 0.001, 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.prefix_tag = prefix_tag    

        codec_ckpt = Path(codec_ckpt).expanduser().resolve()
        
        # the codec
        self.codec = DAC.load(codec_ckpt, map_location="cpu")
        self.codec = torch.compile(self.codec)

        # the controller:
        ctrl_keys = ctrl_keys if ctrl_keys is not None else []
        self.controller = Sketch2SoundController(
            ctrl_keys=ctrl_keys, 
            hop_length=self.codec.hop_length,
            sample_rate=self.codec.sample_rate, 
        )

        # the vampnet
        self.model = VampNet(
            n_heads=n_heads,
            n_layers=n_layers,
            embedding_dim=embedding_dim,
            latent_dim=self.codec.quantizer.quantizers[0].codebook_dim,
            mode=mode,
            ctrl_dims=self.controller.ctrl_dims, 
            vocab_size=self.codec.quantizer.quantizers[0].codebook_size,
        )

        # to speed up the first steps of training, 
        # initialize VampNet's embedding layers with the codec's quantizers
        self.model.embedding.quantizer.load_state_dict(
            self.codec.quantizer.state_dict()
        ) 
        
        self.codec.eval()
        self.codec.requires_grad_(False)

        self.outpaint_prob = outpaint_prob
        self.prefix_min = prefix_min
        self.prefix_max = prefix_max

        # trainer things
        self.criterion = nn.CrossEntropyLoss()
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=1)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        scheduler = vampnet.scheduler.NoamScheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def get_controls(self, sig: sn.Signal):
        # get controls
        n_batch = sig.wav.shape[0]  
        if self.controller is not None:
            ctrls = self.controller.extract(sig)
            # draw control masks
            ctrl_masks = self.controller.random_mask(
                ctrls, 
                r=self.rng.draw(n_batch)[:, 0].to(self.device)
            )
        else:
            ctrls = None
            ctrl_masks = None
        
        return ctrls, ctrl_masks

    def generate_z_mask(self, z, vn, n_batch, ctrl_masks=None):
        r = self.rng.draw(n_batch)[:, 0].to(self.device)

        mask, ii = self.model.random_mask(z, r)
        mask = pmask.codebook_unmask(mask, vn.n_conditioning_codebooks)

        # outpaint? 
        if self.outpaint_prob > 0:
            if flip_coin(self.outpaint_prob):
                mask, ctrl_masks = self.build_tria_mask(mask, ctrl_masks)
        z_mask = pmask.apply_mask(z, mask, vn.mask_token)
        
        return z_mask, mask, ii, r, ctrl_masks

    def build_tria_mask(self, mask, ctrl_masks):
        tmask = vampnet.mask.tria_mask(mask, self.prefix_min, self.prefix_max)

        # this enables the codes where there's prefix
        mask = vampnet.mask.mask_and(mask, tmask)

        for ck, ctrlmask in ctrl_masks.items():
            # this disables the control where there's prefix
            # which is what we want
            ctrl_masks[ck] = vampnet.mask.mask_and(ctrlmask, tmask[:, 0, :])

        return mask, ctrl_masks

    def training_step(self, batch, batch_idx):
        self.model.train()
        batch = sn.prepare_batch(batch, self.device)
        sig = batch["sig"]
        sig.wav = sn.cut_to_hop_length(sig.wav, self.codec.hop_length)
        n_batch = sig.wav.shape[0]

        ctrls, ctrl_masks = self.get_controls(sig)

        output = {}
        vn = self.model
        with torch.inference_mode():
            self.codec.to(self.device)
            z = self.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : vn.n_codebooks, :]


        z_mask, mask, ii, r, ctrl_masks = self.generate_z_mask(z, vn, n_batch, ctrl_masks)
        
        z_hat = self.model(z_mask, ctrls=ctrls, ctrl_masks=ctrl_masks)

        target = codebook_flatten(
            z[:, vn.n_conditioning_codebooks :, :],
        )

        flat_mask = codebook_flatten(
            mask[:, vn.n_conditioning_codebooks :, :],
        )

        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
        
        # add the ignore indices from the mask generator
        ii = codebook_flatten(ii[:, vn.n_conditioning_codebooks :, :])
        t_masked = t_masked.masked_fill(ii.bool(), IGNORE_INDEX)

        output["loss"] = self.criterion(z_hat, t_masked)

        self.log("loss/train", output["loss"], on_step=True, prog_bar=True, sync_dist=True)

        return output["loss"]

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.codec.eval()
        batch = sn.prepare_batch(batch, self.device)
        sig = batch["sig"]
        sig.wav = sn.cut_to_hop_length(sig.wav, self.codec.hop_length)
        n_batch = sig.wav.shape[0]

        ctrls, ctrl_masks = self.get_controls(sig)

        vn = self.model
        z = self.codec.encode(sig.wav, sig.sr)["codes"]
        z = z[:, : vn.n_codebooks, :]

        z_mask, mask, ii, r, ctrl_masks = self.generate_z_mask(z, vn, n_batch, ctrl_masks)

        z_hat = self.model(z_mask, ctrls=ctrls, ctrl_masks=ctrl_masks)

        target = codebook_flatten(
            z[:, vn.n_conditioning_codebooks :, :],
        )

        flat_mask = codebook_flatten(
            mask[:, vn.n_conditioning_codebooks :, :]
        )

        output = {}
        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), IGNORE_INDEX)
        # # add the ignore indices from the mask generator
        ii = codebook_flatten(ii[:, vn.n_conditioning_codebooks :, :])
        t_masked = t_masked.masked_fill(ii.bool(), IGNORE_INDEX)
        output["loss/val"] = self.criterion(z_hat, t_masked)

        # update flat mask for metrics
        flat_mask = flat_mask.masked_fill(ii.bool(), 0)

        # TODO: add ctrl adherence metrics
        log_accuracy_metrics(
            r=r,
            z_hat=z_hat,
            target=target,
            flat_mask=flat_mask,
            dict_to_update=output,
        )

        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return output


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ datasets // transforms ~~~~ 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AUGMENT = False
if AUGMENT:
    from torch_pitch_shift import get_fast_shifts, pitch_shift
    SHIFTS = get_fast_shifts(44100, lambda x : x != 1 and x <1.5 and x > 0.5)

def build_transform():
    def transform(sig):
        # TODO: maybe figure out a nicer way to do these augment probs
        if AUGMENT:
            if flip_coin(0.5):
                # sig = sn.pitch_shift(sig, random.randint(-7, 7))
                #pick a shift
                shift = random.choice(SHIFTS)
                sig.wav = pitch_shift(sig.wav, shift, sig.sr)
            if flip_coin(0.3):
                sig = sn.low_pass(sig, random.randint(1000, 16000))
            if flip_coin(0.3):
                sig = sn.high_pass(sig, random.randint(40, 200))

        sig = sn.normalize(sig, -16.0)
        return sig
    return transform


class VampNetDataModule(L.LightningDataModule):

    def __init__(self,
        db_path: str = "sm.db", 
        query: str = DEFAULT_QUERY, 
        sample_rate: int = 44100,
        n_samples: int = 132096,
        use_chunk_table: bool = True,
        batch_size: int = 16,
        num_workers: int = 12, 
        augment: bool = False
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.db_path = db_path
        self.query = query
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_samples = n_samples
        self.use_chunk_table = use_chunk_table

        self.train_data, self.val_data = None, None

        self.train_tfm = build_transform() if augment else None
        self.val_tfm = build_transform() if augment else None

    def setup(self, stage: Optional[str] = None):
        sample_rate = self.sample_rate
        n_samples = self.n_samples

        conn = sm.connect(self.db_path)
        print(f"loading data from {self.db_path}")

        df = pd.read_sql(self.query, conn)
        df = df.sample(frac=1, random_state=SEED)
        tdf, vdf = sm.dataset.train_test_split(df, test_size=0.1, seed=SEED)

        vdf = vdf.sample(frac=1, random_state=SEED)

        self.train_data = Dataset(
            tdf, sample_rate=sample_rate, n_samples=n_samples,
            transform=self.train_tfm, use_chunk_table=self.use_chunk_table, 
            seed=SEED
        )
        self.val_data = Dataset(
            vdf, sample_rate=sample_rate, n_samples=n_samples,
            transform=self.val_tfm, use_chunk_table=self.use_chunk_table,
            max_examples=2000, 
            seed=SEED
        )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=Dataset.collate
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            collate_fn=Dataset.collate
        )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~ metrics ~~~~ 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    acc = torch.mean(correct.float())

    return acc

def log_accuracy_metrics(z_hat, r, target, flat_mask, dict_to_update):
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

            dict_to_update[f"{tag}/unmasked"] = accuracy(
                preds=r_z_hat,
                target=r_unmasked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )
            dict_to_update[f"{tag}/masked"] = accuracy(
                preds=r_z_hat,
                target=r_masked_target,
                ignore_index=IGNORE_INDEX,
                top_k=topk,
            )

class AudioSampleLoggingCallback(Callback):

    def __init__(self, dataset: Dataset, num_samples: int = 10):
        self.dataset = dataset
        self.num_samples = num_samples


    @torch.inference_mode()
    def _save_onesteps(self, module):
        rs = torch.linspace(0, 1, self.num_samples+1)[1:]
        for i in range(self.num_samples):
            sig = self.dataset[i]["sig"].to(module.device)
            sig.wav = sn.cut_to_hop_length(sig.wav, module.codec.hop_length)

            ctrls, ctrl_masks = module.get_controls(sig)

            z = module.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : module.model.n_codebooks, :]

            n_batch = z.shape[0]
            r = rs[i] * torch.ones(n_batch).to(z.device)

            mask, ii = module.model.random_mask(z, r)
            mask = pmask.codebook_unmask(mask, module.model.n_conditioning_codebooks)
            z_mask = pmask.apply_mask(z, mask, module.model.mask_token)

            z_hat = module.model(z_mask, ctrls=ctrls, ctrl_masks=ctrl_masks)
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
            sig.wav = sn.cut_to_hop_length(sig.wav, module.codec.hop_length)

            ctrls, ctrl_masks = module.get_controls(sig)

            z = module.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : module.model.n_codebooks, :]

            mask = pmask.full_mask(z)
            if module.outpaint_prob > 0:
                if flip_coin(module.outpaint_prob):
                    mask, ctrl_masks = module.build_tria_mask(mask, ctrl_masks)
            
            z_mask = pmask.apply_mask(z, mask, module.model.mask_token)

            z_hat = module.model.generate(z_mask, ctrls=ctrls, ctrl_masks=ctrl_masks)

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
            sig.wav = sn.cut_to_hop_length(sig.wav, module.codec.hop_length)

            ctrls, ctrl_masks = module.get_controls(sig)

            period = periods[i % len(periods)]

            z = module.codec.encode(sig.wav, sig.sr)["codes"]
            z = z[:, : module.model.n_codebooks, :]

            mask = pmask.periodic_mask(z, period, 1, random_roll=False)
            z_mask = pmask.apply_mask(z, mask, module.model.mask_token)

            z_hat = module.model.generate(
                z_mask, 
                ctrls=ctrls, 
                ctrl_masks=ctrl_masks
            )

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ ~~~~ training setup ~~~~ ~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_model_tag(model):
    return model.prefix_tag + f"{'tria-' if model.outpaint_prob > 0.1 else ''}d{model.model.embedding_dim}-l{model.model.n_layers}-h{model.model.n_heads}-mode-{model.hparams.mode}_{'-'.join(model.hparams.ctrl_keys)}"


@argbind.bind(without_prefix=True)
def get_checkpoint_path(resume_ckpt: str = None):
    print("~~~~")
    print(f"resuming from {resume_ckpt}" if resume_ckpt else "~~starting from scratch!!")
    print("~~~~")
    return resume_ckpt


@argbind.bind(without_prefix=True)
def is_fine_tuning(fine_tune: bool = False):
    return fine_tune

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~ ~~~~ the recipe ~~~~ ~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0

    VampNetTrainer = argbind.bind(VampNetTrainer)
    VampNetDataModule = argbind.bind(VampNetDataModule)

    with argbind.scope(args):
        # ~~~~ resume from checkpoint? ~~~~
        resume_ckpt = get_checkpoint_path()
        if resume_ckpt is not None:
            assert resume_ckpt.endswith(".ckpt"), f"checkpoint path must end with .ckpt, got {resume_ckpt}"

        # if we're fine-tuning, we don't want to load the checkpoint
        # for the trainer, just the weights
        are_we_fine_tuning = is_fine_tuning()

        # ~~~~ set up model ~~~~~
        model = (
            VampNetTrainer.load_from_checkpoint(checkpoint_path=resume_ckpt)
                if resume_ckpt is not None
                else VampNetTrainer()
        )
        # make sure the the tag comes from the config
        if "VampNetTrainer.prefix_tag" in args:
            model.prefix_tag = args["VampNetTrainer.prefix_tag"]


        # ~~~~ set up data ~~~~
        dm = VampNetDataModule(sample_rate=model.codec.sample_rate)
        dm.prepare_data()
        dm.setup()
        train_dataloader = dm.train_dataloader()
        val_dataloader = dm.val_dataloader()

        # ~~~~~ add datamodule metadata to model ~~~~~

        # ~~~~ callbacks ~~~~~
        callbacks = []
        callbacks.append(AudioSampleLoggingCallback(dm.val_data))
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        callbacks.append(ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best",
        ))
        
        # figure out how many gpus we have
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
        n_gpus = len(cuda_visible_devices.split(","))
        print(f"using {n_gpus} gpus")

        trainer = L.Trainer(
            devices=n_gpus,
            default_root_dir=f"runs/{get_model_tag(model)}",
            max_epochs=-1,
            limit_val_batches=20,
            gradient_clip_val=1.0,
            # val_check_interval=100,
            val_check_interval=1000,
            callbacks=callbacks,
            precision="bf16-mixed", 
            strategy="ddp_find_unused_parameters_true", 
            detect_anomaly=True
        )

        dm.train_data.seed = SEED + trainer.local_rank
        model.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=SEED + trainer.local_rank)

        trainer.fit(
            model=model, 
            train_dataloaders=dm.train_dataloader(), 
            val_dataloaders=dm.val_dataloader(), 
            ckpt_path=resume_ckpt if not are_we_fine_tuning else None
        )