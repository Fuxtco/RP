# Minimal PyTorch implementation of LARC (Layer-wise Adaptive Rate Control)
# inspired by NVIDIA Apex LARC and open-source reimplementations.

from typing import Iterable
import torch
from torch.optim import Optimizer


class LARC(Optimizer):
    """
    Layer-wise Adaptive Rate Control (LARC).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        trust_coefficient: float = 0.001,
        clip: bool = True,
        eps: float = 1e-8,
        always_adapt: bool = False,
    ):
        if not isinstance(optimizer, Optimizer):
            raise ValueError("optimizer must be an instance of torch.optim.Optimizer")
        
        if trust_coefficient <= 0:
            raise ValueError("trust_coefficient should be > 0")
        
        self.optimizer = optimizer
        self.trust_coefficient = trust_coefficient
        self.clip = clip
        self.eps = eps
        self.always_adapt = always_adapt
        defaults = {}
        super().__init__(optimizer.param_groups, defaults)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.optimizer.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                p_data = p.data
                g_data = p.grad.data

                # norms in fp32 for stability
                param_norm = torch.norm(p_data.float())

                # include weight decay in the norm (LARC standard behavior)
                wd = group.get("weight_decay", 0.0)
                if wd != 0.0:
                    g_for_norm = g_data.float().add(p_data.float(), alpha=wd)  # g + wd*w
                else:
                    g_for_norm = g_data.float()

                grad_norm = torch.norm(g_for_norm)

                # skip if non-finite or too small
                if not torch.isfinite(param_norm) or not torch.isfinite(grad_norm):
                    continue
                pn = param_norm.item()
                gn = grad_norm.item()
                if pn == 0.0 or gn == 0.0:
                    continue

                # local lr as python float
                local_lr = self.trust_coefficient * pn / (gn + self.eps)

                # clipping mode (recommended for stability)
                actual_lr = min(local_lr, lr) if self.clip else local_lr

                # scale the original gradient in-place
                g_data.mul_(actual_lr / (lr + self.eps))

        # w <- w - lr * g_scaled
        self.optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
