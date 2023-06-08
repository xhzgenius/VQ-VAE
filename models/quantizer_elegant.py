import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Code reference: https://zhuanlan.zhihu.com/p/463043201
Type annotation by XHZ
'''

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, beta: float):
        """
        VQ-VAE layer: Input any tensor to be quantized. 
        
        Args:
        ---
        embedding_dim (int): the dimensionality of the tensors in the quantized space. 
        Inputs to the modules must be in this format as well.
        
        num_embeddings (int): the number of vectors in the quantized space.
        
        beta (float): scalar which controls the weighting of the loss terms 
        (see equation 4 in the paper - this variable is Beta).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        
        # initialize embeddings
        self.embedding_vectors = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, z_encoded: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        # [B, C, H, W] -> [B, H, W, C]
        z_encoded = z_encoded.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = z_encoded.reshape(-1, self.embedding_dim)
        
        encoding_indices = self.get_code_indices(flat_x)
        z_quantized = self.quantize(encoding_indices).view_as(z_encoded) # [B, H, W, C]
        
        # embedding loss: move the embeddings towards the encoder's output
        z_q_loss = F.mse_loss(z_quantized, z_encoded.detach())
        # commitment loss
        z_e_loss = F.mse_loss(z_encoded, z_quantized.detach())
        loss = z_q_loss + self.beta * z_e_loss

        # Straight Through Estimator
        z_quantized = z_encoded + (z_quantized - z_encoded).detach()
        
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # perplexity: e^-\Sigma[p(x)*log p(x)]
        embedding_distribution = torch.mean(F.one_hot(encoding_indices, num_classes=self.num_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(
            embedding_distribution*torch.log(embedding_distribution+1e-10)
            ))
        
        return z_quantized, loss, perplexity
    
    def get_code_indices(self, flat_x: torch.Tensor) -> torch.Tensor:
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding_vectors.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embedding_vectors.weight.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """Returns embedding tensor for a batch of indices."""
        return self.embedding_vectors(encoding_indices) 