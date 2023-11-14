import torch
import torch.nn as nn
from setup import device
from jaxtyping import Array, Float
from typing import Callable

loss_fn = torch.nn.MSELoss()

class DSMLoss():

    def __init__(self, alpha: float, diff_weight: bool):
        """
        Initialize the DSM Loss.

        Args:
            alpha: regularization weight
            diff_weight: scale loss by square of diffusion

        Returns:
            None
        """
        self.alpha = alpha
        self.diff_weight = diff_weight

    def __call__(self,
                 t: Array,
                 x: Array,
                 model: Callable[[Array], Array],
                 s: Array,
                 diff_sq: Float):
        """
        Args:
            t: uniformly sampled time period
            x: samples after t diffusion
            model: score prediction function s(x,t)
            s: ground truth score

        Returns:
            loss: average loss value
        """
        pred = model(t.to(device), x.to(device))
        loss = loss_fn(pred, s.to(device)) 
        if self.diff_weight:
            loss *= diff_sq
        return loss


class ISMLoss():
    """
    Implicit Score Matching Loss
    """

    def __init__(self):
        pass

    def __call__(self, t, x, model):
        """
        Args:
            t: uniformly sampled time period
            x: samples after t diffusion
            model: score prediction function s(x,t)

        Returns:
            loss: average loss value
        """
        t, x = t.to(device), x.to(device)
        x.requires_grad = True
        s = model(t, x)
        # div = torch.autograd.grad(s, x, grad_outputs=torch.ones_like(s), create_graph=True)[0]
        div = 0
        for i in range(s.shape[-1]):
            div += torch.autograd.grad(s[..., i], x, grad_outputs=torch.ones_like(s[..., i]), create_graph=True)[0][..., i:i+1]
        loss = (.5*torch.sum(s**2, dim=-1) + div).mean()
        return loss
