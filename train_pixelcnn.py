'''
The script to train the Gated PixelCNN model. 

Created by XHZ. 

Code reference: 
The code is partly selected and modified from https://github.com/xiaohu2015/nngen/blob/main/models/vq_vae.ipynb. 
'''
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.pixelcnn import GatedPixelCNN
import utils

num_embeddings = 512

pixelcnn = GatedPixelCNN(num_embeddings, 128, num_embeddings)
pixelcnn = pixelcnn.cuda()

optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=1e-3)

train_indices = np.load("./data/encoding_indices/anime.npy")
train_indices = torch.as_tensor(train_indices)

class IndiceDataset(Dataset):
    def __init__(self, indices: torch.Tensor):
        self.indices: torch.Tensor = indices
        super(IndiceDataset, self).__init__()
    
    def __getitem__(self, index):
        return self.indices[index]
    
    def __len__(self):
        return len(self.indices)

indices_loader = DataLoader(IndiceDataset(train_indices), batch_size=1023)

timestamp = utils.readable_timestamp()
SAVE_PATH = os.path.join("./results", "pixelcnn "+timestamp)
os.makedirs(SAVE_PATH, exist_ok=True)
print("Results will be saved in", SAVE_PATH)

# train pixelcnn
total_epochs = 200
save_freq = 20
for epoch in tqdm(range(1, total_epochs+1)):
    print("Start training epoch {}".format(epoch,))
    for i, (indices) in enumerate(indices_loader):
        indices = indices.cuda()
        # print(indices)
        one_hot_indices = F.one_hot(indices, num_embeddings).float().view(-1, num_embeddings, 8, 8)#.permute(0, 3, 1, 2).contiguous()
        # print("One hot indices:", one_hot_indices)
        
        outputs = pixelcnn(one_hot_indices)
        outputs = outputs.view(-1, num_embeddings, 64)
        # print("Outputs of PixelCNN:", outputs.shape)
        
        loss = F.cross_entropy(outputs, indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i+1==len(indices_loader):
            print("[{}/{}]: loss {}".format(epoch, total_epochs, loss.item()))
    
    if epoch%save_freq==0: # Save the latest model and delete the oldest one. 
        old_checkpoing_path = os.path.join(SAVE_PATH, "%s.pth"%(epoch-5*save_freq))
        if os.path.isfile(old_checkpoing_path):
            os.remove(old_checkpoing_path)
            # print("Deleted old checkpoint:", old_checkpoing_path)
        save_path = os.path.join(SAVE_PATH, "%s.pth"%epoch)
        torch.save(pixelcnn, save_path)