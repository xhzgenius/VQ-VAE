'''
Some utility functions. 
Created by the original author and updated by XHZ. 
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np

from models.vqvae import VQVAE


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == "anime":
        transform=transforms.Compose([transforms.ToTensor(),
                                    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
        training_data = datasets.ImageFolder("./data/faces64x", transform)
        validation_data = datasets.ImageFolder("./data/faces64x_validation", transform)
        training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
        x_train_var = None
    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S")


def save_model_and_results(model, results, hyperparameters, save_dir, epoch):
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    old_checkpoing_path = os.path.join(save_dir, "%s.pth"%(epoch-5*hyperparameters["log_interval"]))
    if os.path.isfile(old_checkpoing_path):
        os.remove(old_checkpoing_path)
        # print("Deleted old checkpoint:", old_checkpoing_path)
    save_path = os.path.join(save_dir, "%s.pth"%epoch)
    torch.save(results_to_save, save_path)
    # print("Results saved to", save_path)
    

def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        data = torch.load(path)
    else:
        data = torch.load(path, map_location=lambda storage, loc: storage)
    
    hyperparameters = data["hyperparameters"]
    results = data["results"]
    
    model = VQVAE(hyperparameters['n_hiddens'], hyperparameters['n_residual_hiddens'],
                  hyperparameters['n_half_conv_layers'], 
                  hyperparameters['n_residual_layers'], 
                  hyperparameters['n_embeddings'], 
                  hyperparameters['embedding_dim'], hyperparameters['beta']).to(device)

    model.load_state_dict(data['model'])
    
    return model, results, hyperparameters

def chou_ka(p: float) -> bool:
    if np.random.random()<p:
        return True
    else:
        return False