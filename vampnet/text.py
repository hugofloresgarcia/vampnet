import torch
from torch import nn


class CLAPTextConditioner(nn.Module):

    def __init__(self, ):
        super().__init__()
        try:
            from msclap import CLAP
        except ImportError:
            raise ImportError("Please install the msclap package: `pip install msclap`")
        self.clap_model = CLAP(version = '2023', use_cuda=True)
        # self.clap_model.eval()
        self.clap = self.clap_model.clap
        self.clap.requires_grad = False

        # remove the audio head
        del self.clap.audio_encoder

        _ex_output = self.clap_model.get_text_embeddings(["test"])
        self.output_dim = _ex_output.shape[-1]

    @torch.inference_mode()
    def forward(self, caps: list[str]):
        # Extract text embeddings
        if isinstance(caps, str):
            caps = [caps]
        emb = self.clap_model.get_text_embeddings(caps)
        emb = emb.unsqueeze(1) # add a singleton channel dim for cfg compat
        return emb

    @property
    def dim(self):
        return self.output_dim
