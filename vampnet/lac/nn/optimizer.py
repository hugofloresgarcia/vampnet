import torch
from torch.distributed.optim import ZeroRedundancyOptimizer


def AdamW(
    parameters,
    lr: float = 1e-3,
    betas: tuple = (0.8, 0.99),
    eps: float = 1e-9,
    use_zero: bool = False,
):
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            parameters,
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            betas=betas,
            eps=eps,
        )
    else:
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, eps=eps)
    return optimizer


def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
