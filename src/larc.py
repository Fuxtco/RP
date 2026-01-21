# Minimal PyTorch implementation of LARC (Layer-wise Adaptive Rate Control)
# inspired by NVIDIA Apex LARC and open-source reimplementations. :contentReference[oaicite:2]{index=2}

from typing import Iterable
import torch
from torch.optim import Optimizer


class LARC(Optimizer):
    """
    Layer-wise Adaptive Rate Control (LARC).

    This wraps an existing optimizer (typically SGD) and rescales gradients
    for each parameter tensor based on ||w|| / ||g||.

    Args:
        optimizer: an existing torch.optim.Optimizer instance (e.g. SGD).
        trust_coefficient: the "eta" / trust ratio coefficient.
        clip: if True, uses min(adaptive_lr, base_lr) (clipping mode);
              if False, uses pure scaling.
        eps: small epsilon to avoid division by zero.
        always_adapt: if True, adapt even when weight_decay == 0.
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

        # 调用父类构造函数：传入同样的 param_groups，
        # 这样 GradScaler.unscale_(optimizer) 等检查 isinstance(Optimizer) 时是合法的。
        defaults = {}
        super().__init__(optimizer.param_groups, defaults)

    # 让外部访问 optimizer.param_groups 时，直接用内部优化器的
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    def step(self, closure=None):
        """执行一次带 LARC 的更新：先缩放梯度，再调用内部 optimizer.step()"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 对每个 param_group、每个参数张量应用 LARC
        for group in self.optimizer.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                p_data = p.data
                g_data = p.grad.data

                # ================================================
                '''
                # 计算参数范数和梯度范数
                param_norm = torch.norm(p_data)
                grad_norm = torch.norm(g_data)

                # 避免 0 / 0
                # if param_norm == 0 or grad_norm == 0:
                #     continue
                # ========================================
                # skip if non-finite or too small 更稳健
                if not torch.isfinite(param_norm) or not torch.isfinite(grad_norm):
                    continue
                if param_norm.item() == 0.0 or grad_norm.item() == 0.0:
                    continue

                # 按论文公式计算 local_lr
                # local_lr = self.trust_coefficient * param_norm / (grad_norm + self.eps)
                # local_lr 是 Tensor，lr 是 float。Python 的 min() 在这种混合类型下会触发 Tensor 的比较逻辑，有时会报错或产生不确定行为。
                local_lr = (self.trust_coefficient * param_norm / (grad_norm + self.eps)).item()

                # 是否只截断不放大
                if self.clip:
                    actual_lr = min(local_lr, lr)
                else:
                    actual_lr = local_lr

                # 通过缩放梯度来“等效改变学习率”
                g_data.mul_(actual_lr / (lr + self.eps))
                '''

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

        # 交给内部优化器做真正的参数更新
        self.optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """清空梯度，直接调用内部优化器的 zero_grad"""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        直接复用内部 optimizer 的 state_dict，这样 checkpoint
        的格式和普通优化器完全一样。
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """加载内部 optimizer 的状态"""
        self.optimizer.load_state_dict(state_dict)
