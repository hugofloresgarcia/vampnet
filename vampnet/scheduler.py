import copy
from typing import List

import torch

class NoamScheduler:
    """OG scheduler from transformer paper: https://arxiv.org/pdf/1706.03762.pdf
    Implementation from Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        factor: float = 1.0,
        warmup: int = 4000,
    ):
        # Store hparams
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model

        # Initialize variables `lr` and `steps`
        self.lr = None
        self.steps = 0

        # Store the optimizer
        self.optimizer = optimizer

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self.steps += 1
        self.lr = self.factor * (
            self.d_model ** (-0.5)
            * min(self.steps ** (-0.5), self.steps * self.warmup ** (-1.5))
        )

        for p in self.optimizer.param_groups:
            p["lr"] = self.lr

