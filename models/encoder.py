'''
The encoder of VQVAE. 
Rewritten by XHZ. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of half-size convolution layers to stack
    - n_res_layers : number of residual layers to stack

    """

    def __init__(self, in_dim, h_dim, n_half_conv_layers, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        lst = []
        for i in range(n_half_conv_layers-1):
            din = h_dim//2**(n_half_conv_layers-i-1)
            dout = h_dim//2**(n_half_conv_layers-i-2)
            lst.extend([
                nn.Conv2d(din, dout, kernel_size=kernel,
                          stride=stride, padding=1), # width = (width+2-4)/2+1 = width/2
                nn.ReLU(),
                ResidualStack(dout, dout, res_h_dim, n_res_layers),
            ])
            
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim//2**(n_half_conv_layers-1), kernel_size=kernel,
                      stride=stride, padding=1), # width = (width+2-4)/2+1 = width/2
            nn.ReLU(),
            nn.Sequential(*lst), 
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1), # width = (width+2-3)/1+1 = width
        ) # width = width/4

    def forward(self, x):
        return self.conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 2, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
