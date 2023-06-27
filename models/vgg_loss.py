
'''
The VGG loss function for training VQVAE. 
Purposed and created by XHZ. 

Code reference: https://github.com/bryandlee/stylegan2-encoder-pytorch/blob/master/train_encoder.py
Modified by XHZ. 
'''
import torch
import torch.nn as nn
import torchvision


class VGGLoss(nn.Module):
    def __init__(self, vgg_version: str = "vgg11"):
        super().__init__()

        if vgg_version=="vgg11":
            vgg = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features
            feature_layers = (0, 6, 11, 16)
            self.weights = (3.0, 1.0, 1.0, 1.0)
        elif vgg_version=="vgg19":
            vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features
            feature_layers = (2, 7, 12, 21, 30)
            self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        else:
            raise ValueError("VGG version not supported: %s"%vgg_version)
        # print(vgg)
        
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers)
            prev_layer = next_layer
        
        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()
        
    def forward(self, source: torch.Tensor, target: torch.Tensor):
        loss = 0 
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight*self.criterion(source, target)
            
        return loss