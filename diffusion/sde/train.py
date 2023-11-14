import torch
from setup import device, m
from tqdm import tqdm
from itertools import repeat
import matplotlib.pyplot as plt
from loss import ISMLoss, DSMLoss



def get_sde_step_fn(model, ema, opt, loss_fn, sde):
    
    # print(f'isinstance(loss_fn, DSMLoss): {isinstance(loss_fn, DSMLoss)}')
    def step_fn(batch):
        # uniformly sample time step
        t = sde.T*torch.rand(batch.shape[0])

        # TODO forward diffusion
        mean, std = sde.marginal_prob(t, batch)
        xt = mean + std * torch.randn_like(batch)

        # get loss
        if isinstance(loss_fn, DSMLoss):
            logp_grad = - 1 / (std**2) * (xt - mean)
            diff_sq = 1/((logp_grad**2).sum(dim=-1).mean())
            loss = loss_fn(t, xt.float(), model, logp_grad, diff_sq)
        elif isinstance(loss_fn, ISMLoss):
            loss = loss_fn(t, xt.float(), model)
        else:
            # print(type(loss_fn))
            raise Exception("undefined loss")

        # optimize model
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ema is not None:
            # update theta_ema <- m*theta_ema + (1-m)*theta
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.data = m*p_ema.data + (1-m)*p.data
            
            loss_ema = loss_fn(t, xt.float(), ema, logp_grad, diff_sq)
            return loss_ema.item()

        return loss.item()

    return step_fn


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def train_diffusion(dataloader, step_fn, N_steps, plot=False):
    pbar = tqdm(range(N_steps), bar_format="{desc}{bar}{r_bar}", mininterval=1)
    loader = iter(repeater(dataloader))

    log_freq = 200
    loss_history = torch.zeros(N_steps//log_freq)
    for i, step in enumerate(pbar):
        batch = next(loader)
        batch = batch.to(device)
        loss = step_fn(batch)

        if step % log_freq == 0:
            loss_history[i//log_freq] = loss
            pbar.set_description("Loss: {:.3f}".format(loss))

    if plot:
        plt.plot(range(len(loss_history)), loss_history)
        plt.show()
