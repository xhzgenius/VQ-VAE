'''
The script to train the VQVAE model. 

Created by the original author and modified by XHZ. 
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from models.vgg_loss import VGGLoss
from focal_frequency_loss import FocalFrequencyLoss as FFL

timestamp = utils.readable_timestamp()


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_half_conv_layers", type=int, default=2)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset", type=str, default='CIFAR10')
parser.add_argument("--resume_dir", type=str, default=None)
parser.add_argument("--resume_epoch", type=int, default=0)

# # whether or not to save model
# parser.add_argument("-save", action="store_true")


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join("./results", args.dataset+" "+timestamp)
print("Results will be saved in", SAVE_PATH)


# Load data and define batch data loaders
training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)


# Set up VQ-VAE model with components defined in ./models/ folder
if args.resume_dir is None:
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens, args.n_half_conv_layers, 
                  args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
    results = {
        'epochs': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
    }
else:
    model, results, _ = utils.load_model(args.resume_dir+"/%d.pth"%args.resume_epoch)
print(model)

# Set up optimizer and training loop
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
vgg_loss = VGGLoss(vgg_version="vgg11").to(device)
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
# input shape: (B, C, H, W)
ffloss = FFL(loss_weight=1.0, alpha=1.0)  # Params same as https://github.com/EndlessSora/focal-frequency-loss. 

model.train()


def train():

    for i in range(1+args.resume_epoch, args.epochs+1+args.resume_epoch):
        (x, _) = next(iter(training_loader))
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, encoding_indices, perplexity = model(x)
        # recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        # recon_loss = l1_loss(x_hat, x)*50
        # recon_loss = vgg_loss(x_hat, x) + l1_loss(x_hat, x)*5
        recon_loss = vgg_loss(x_hat, x) + l2_loss(x_hat, x)*10
        # recon_loss = ffloss(x_hat, x)*2000 + l2_loss(x_hat, x)*10
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["epochs"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            hyperparameters = args.__dict__
            os.makedirs(SAVE_PATH, exist_ok=True)
            utils.save_model_and_results(
                model, results, hyperparameters, SAVE_PATH, i)

            print('Epoch #', i, 
                  'Recon Error:', np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


if __name__ == "__main__":
    train()
