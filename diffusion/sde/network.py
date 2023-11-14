import torch
import torch.nn as nn
import torch.nn.functional as F
from setup import device
import numpy as np
from jaxtyping import Array, Int, Float
# from jaxtyping import Int, Float
# from torch import Tensor as Array


class PositionalEncoding(nn.Module):

    def __init__(self, t_channel: Int):
        """
        Initialize positional encoding network

        Args:
            t_channel: number of modulation channel
        """
        super().__init__()
        self.t_channel = t_channel

    def forward(self, t: Array):
        """
        Return the positional encoding of

        Args:
            t: input time

        Returns:
            emb: time embedding
        """
        # create embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        emb = [t]
        for i in range(self.t_channel):
            emb.append(torch.sin(2**i * t))
            emb.append(torch.cos(2**i * t))
        emb = torch.cat(emb, dim=-1)
        return emb


class MLP(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 out_dim: Int,
                 hid_shapes: Int[Array, '...']):
        '''
        Build simple MLP

        Args:
            in_dim: input dimension
            out_dim: output dimension
            hid_shapes: array of hidden layers' dimension
        '''
        super().__init__()
        all_shapes = [in_dim] + hid_shapes
        self.model = nn.Sequential(
            *[nn.Sequential(nn.Linear(all_shapes[i], all_shapes[i+1]),
                            nn.ReLU()) for i in range(len(all_shapes)-1)],
            nn.Linear(all_shapes[-1], out_dim)
        )

    def forward(self, x: Array):
        return self.model(x)



class SimpleNet(nn.Module):

    def __init__(self,
                 in_dim: Int,
                 enc_shapes: Int[Array, '...'],
                 dec_shapes: Int[Array, '...'],
                 z_dim: Int):
        super().__init__()
        '''
        Build Score Estimation network.
        You are free to modify this function signature.
        You can design whatever architecture.

        hint: it's recommended to first encode the time and x to get
        time and x embeddings then concatenate them before feeding it
        to the decoder.

        Args:
            in_dim: dimension of input
            enc_shapes: array of dimensions of encoder
            dec_shapes: array of dimensions of decoder
            z_dim: output dimension of encoder
        '''
        
        t_channel = 10
        self.t_encoder = nn.Sequential(
            PositionalEncoding(t_channel),
            MLP(t_channel*2+1, z_dim, enc_shapes),
            nn.ReLU()
        )   
        self.x_encoder = nn.Sequential(
            MLP(in_dim, z_dim, enc_shapes),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            MLP(z_dim*2, in_dim, dec_shapes)
        )


    def forward(self, t: Array, x: Array):
        '''
        Implement the forward pass. This should output
        the score s of the noisy input x.

        hint: you are free

        Args:
            t: the time that the forward diffusion has been running
            x: the noisy data after t period diffusion
        '''
        t, x = t.to(device), x.to(device)
        # print(f'x requires grad: {x.requires_grad}')
        # encode time
        t_emb = self.t_encoder(t)
        # feed x to encoder
        z = self.x_encoder(x)
        # concatenate z and t_emb
        z = torch.cat([z, t_emb], dim=-1)
        # feed to decoder
        s = self.decoder(z)

        return s
