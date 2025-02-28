'''
The VQVAE model. 
Created by the original author and updated (really a lot) by XHZ. 
'''
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer_elegant import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim: int, res_h_dim: int, n_half_conv_layers: int, n_res_layers: int, 
                 n_embeddings: int, embedding_dim: int, beta: float, 
                 save_img_embedding_map: bool = False):
        super(VQVAE, self).__init__()
        
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_half_conv_layers, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantizer = VectorQuantizer(embedding_dim, n_embeddings, beta)
        
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_half_conv_layers, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x: torch.Tensor, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        z_q, embedding_loss, encoding_indices, perplexity = self.vector_quantizer(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, encoding_indices, perplexity
