import abc
import torch
from setup import device
import numpy as np
from jaxtyping import Array
# from torch import Tensor as Array

class SDE(abc.ABC):
    def __init__(self, N: int, T: int):
        super().__init__()
        self.N = N         # number of discretization steps
        self.T = T         # terminal time
        self.dt = T / N
        self.is_reverse = False
        self.is_bridge = False

    @abc.abstractmethod
    def sde_coeff(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def marginal_prob(self, t, x):
        return NotImplemented

    @abc.abstractmethod
    def predict_fn(self, x):
        return NotImplemented

    @abc.abstractmethod
    def correct_fn(self, t, x):
        return NotImplemented

    def dw(self, x, dt=None):
        """
        Return the differential of Brownian motion

        Args:
            x: input data

        Returns:
            dw (same shape as x)
        """
        dt = self.dt if dt is None else dt
        dw = torch.randn_like(x) * (dt**0.5)
        return dw

    def prior_sampling(self, x: Array):
        """
        Sampling from prior distribution. Default to unit gaussian.

        Args:
            x: input data

        Returns:
            z: random variable with same shape as x
        """
        return torch.randn_like(x)

    def predict_fn(self,
                   t: Array,
                   x: Array,
                   dt: float=None):
        """
        Perform single step diffusion.

        Args:
            t: current diffusion time
            x: input with noise level at time t
            dt: the discrete time step. Default to T/N

        Returns:
            x: input at time t+dt
        """
        dt = self.dt if dt is None else dt
        f, g = self.sde_coeff(t, x)
        dx = f * dt + g * self.dw(x, dt)
        pred = x + dx
        return pred

    def correct_fn(self, t: Array, x: Array):
        return None

    def reverse(self, model):
        N = self.N
        T = self.T
        forward_sde_coeff = self.sde_coeff

        class RSDE(self.__class__):
            def __init__(self, score_fn):
                super().__init__(N, T)
                self.score_fn = score_fn
                self.is_reverse = True
                self.forward_sde_coeff = forward_sde_coeff

            def sde_coeff(self, t: Array, x: Array):
                """
                Return the reverse drift and diffusion terms.

                Args:
                    t: current diffusion time
                    x: current input at time t

                Returns:
                    reverse_f: reverse drift term
                    g: reverse diffusion term
                """
                forward_f, g = self.forward_sde_coeff(self.T - t, x)
                reverse_f = forward_f - g**2 * self.score_fn(self.T - t, x)
                return reverse_f, g

            def predict_fn(self,
                           t: Array,
                           x,
                           dt=None,
                           ode=False):
                """
                Perform single step reverse diffusion

                """
                dt = self.dt if dt is None else dt
                reverse_f, g = self.sde_coeff(t, x)
                dx = -reverse_f * dt + g * self.dw(x, dt)
                pred = x + dx
                return pred

        return RSDE(model)

class OU(SDE):
    def __init__(self, N=1000, T=1):
        super().__init__(N, T)

    def sde_coeff(self, t, x):
        t, x = t.to(device), x.to(device)
        f, g = -.5 * x, torch.ones(x.shape)
        return f.to(device), g.to(device)

    def marginal_prob(self, t, x):
        t, x = t.to(device), x.to(device)
        mean, std = x * torch.exp(-.5*t).unsqueeze(-1).to(device), \
            torch.ones_like(x, device=device) * \
            torch.sqrt(1 - torch.exp(-t)).unsqueeze(-1).to(device)
        return mean, std

class VESDE(SDE):
    def __init__(self, N=1000, T=1, sigma_min=0.01, sigma_max=50):
        super().__init__(N, T)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ratio = sigma_max / sigma_min

    def sde_coeff(self, t, x):
        t, x = t.to(device), x.to(device)
        f, g = torch.zeros_like(x, device=device), torch.ones_like(x, device=device) *\
            self.sigma_min * self.ratio**t.unsqueeze(-1) *\
            np.sqrt(2*np.log(self.ratio))
        return f, g

    def marginal_prob(self, t, x):
        t, x = t.to(device), x.to(device)
        mean, std = x, torch.ones_like(x, device=device) *\
            (self.sigma_min * self.ratio**(t)).unsqueeze(-1).to(device)
        return mean, std


class VPSDE(SDE):
    def __init__(self, N=1000, T=1, beta_min=0.1, beta_max=20):
        super().__init__(N, T)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def sde_coeff(self, t, x):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        beta = beta.unsqueeze(-1)
        f, g = -.5 * beta * x, \
            torch.ones_like(x, device=device) * torch.sqrt(beta)
        return f, g

    def marginal_prob(self, t, x):
        t, x = t.to(device), x.to(device)
        quadratic = -.5 * t**2 * (self.beta_max - self.beta_min) -\
            t * self.beta_min
        mean, std = x * torch.exp(.5*quadratic).unsqueeze(-1).to(device), \
            1-torch.ones_like(x, device=device) * \
            torch.exp(quadratic).unsqueeze(-1).to(device)
        return mean, std
