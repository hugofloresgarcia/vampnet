import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools.metrics.distance import L1Loss
from audiotools.metrics.spectral import MelSpectrogramLoss
from audiotools.metrics.spectral import MultiScaleSTFTLoss
from audiotools.metrics.spectral import PhaseLoss
from torch import nn


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


if __name__ == "__main__":
    import wav2wav

    discriminator = wav2wav.modules.Discriminator()
    gan_loss = GANLoss(discriminator)
    fake = AudioSignal(torch.randn(10, 1, 44100), 44100)
    real = AudioSignal(torch.randn(10, 1, 44100), 44100)

    print(gan_loss.discriminator_loss(fake, real))
    print(gan_loss.generator_loss(fake, real))
