'''

'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.vqvae import VQVAE
from models.pixelcnn import GatedPixelCNN
import utils

# Load models. 
model: VQVAE
model, results, hyperparameters = utils.load_model("./results/anime 2023-06-28 04.42.54/12000.pth")
pixelcnn: GatedPixelCNN = torch.load("./results/pixelcnn 2023-06-28 07.27.22/95.pth")

# Hyperparameters. 
num_embeddings = 512

# Create an empty array of priors.
n_samples = 32
prior_size = (8, 8) # h, w
real_priors: torch.Tensor = torch.as_tensor(np.load("./data/encoding_indices/anime.npy"))
print(real_priors.shape)
real_priors_count = torch.sum(F.one_hot(real_priors, num_embeddings), dim=0)
print(real_priors_count.shape)

# priors = torch.zeros((n_samples,) + prior_size, dtype=torch.long).cuda()
priors = real_priors[list(range(32))].view(n_samples, *prior_size).cuda()
# priors = torch.randint(0, 512, size=(n_samples,)+prior_size).cuda()
# priors = torch.multinomial(torch.as_tensor(real_priors_count, dtype=torch.float32), n_samples).view(8, 8, 32).permute(2, 0, 1).cuda()
print(priors.shape)
# raise

# use pixelcnn to generate priors
pixelcnn.eval()

# Iterate over the priors because generation has to be done sequentially pixel by pixel.
# for row in range(0, prior_size[0]):
#     for col in range(0, prior_size[1]):
for row in range(0, 8):
    for col in range(0, 8):
        update_prob = (row*8+col)/64/2
        update_prob = 0
        print("update probability:", update_prob)
        if utils.chou_ka(p=update_prob)==False:
            continue
        # Feed the whole array and retrieving the pixel value probabilities for the next pixel.
        with torch.inference_mode():
            one_hot_priors = F.one_hot(priors, num_embeddings).float().permute(0, 3, 1, 2).contiguous()
            logits = pixelcnn(one_hot_priors)
            probs = F.softmax(logits[:, :, row, col], dim=-1)
            # Use the probabilities to pick pixel values and append the values to the priors.
            priors[:, row, col] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
            # priors[:, row, col] = torch.argmax(probs, dim=-1).squeeze(dim=-1)
            # print("Current priors:", priors)
# Perform an embedding lookup and Generate new images
with torch.inference_mode():
    z = model.vector_quantizer.quantize(priors)
    z = z.permute(0, 3, 1, 2).contiguous()
    pred: torch.Tensor = model.decoder(z)
    # print("Generated new images:", pred.shape) # [n_samples, 3, 64, 64]

generated_samples = pred.permute(0, 2, 3, 1).cpu().numpy() # [n_samples, 64, 64, 3]

rows = int(n_samples/8+0.875)
fig = plt.figure(figsize=(1.25*min(8, n_samples), 1.25*rows), constrained_layout=True)
gs = fig.add_gridspec(rows, 8)
for n_row in range(int(n_samples/8+0.875)):
    for n_col in range(8):
        if n_row*8+n_col>=n_samples:
            break
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow(generated_samples[n_row*8+n_col])
        f_ax.axis("off")
plt.show()