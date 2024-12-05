"""
Latent Space Sampling, By Zibin Zhao started 04/12/2023
INPUT: Trained model to be used to evaluate
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from utility import onehot_to_seq, write_to_fasta
import time

# Set this parameters
MODEL_PATH = './save_models/CSr7.pt'
DATA_PATH = './data/input_data_CSr7.pt'
OUTPUT_FILE_HD = "./decoded_repeat/hd_CSr7.fasta"
OUTPUT_FILE_CC = "./decoded_repeat/cc_CSr7.fasta"

# Set global seed for reproducibility
SEED = 42
seed_everything(SEED)

# Hyperparameters (Fixed)
LAYER_1_SIZE = 256
LAYER_2_SIZE = 64
LATENT_SIZE = 64
DROPOUT_RATE = 0.5
INPUT_SIZE = 273
OUTPUT_SIZE = 273

#NOTE: For CS & TES is 39, for DHEA & CHO is 32
SEQ_LENGTH = 39 * 4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load input data
input_data = torch.load(DATA_PATH)

class MyDataset(Dataset):
    """ Custom dataset class for NGS data. """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class NGS_VAE(nn.Module):
    """ Variational Autoencoder for NGS data. """
    def __init__(self, layer1_size=LAYER_1_SIZE, latent_size=LATENT_SIZE, layer2_size=LAYER_2_SIZE, dropout_rate=DROPOUT_RATE):
        super(NGS_VAE, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(INPUT_SIZE, layer1_size)
        self.fc21 = nn.Linear(layer1_size, latent_size)  # mu layer
        self.fc22 = nn.Linear(layer1_size, latent_size)  # log_var layer
        # Decoder layers
        self.fc3 = nn.Linear(latent_size, layer2_size)
        self.fc4 = nn.Linear(layer2_size, OUTPUT_SIZE)
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return torch.sigmoid(self.fc4(F.relu(self.fc3(z))))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def main():
    """ Main evaluation and sampling loop. """
    
    # Prepare dataset
    print(input_data.shape)
    dataset = MyDataset(input_data)  # Use all samples
    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # Data loaders
    data_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    # Load the trained model
    model = NGS_VAE().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
    model.eval()

    # Extract latent variables
    latent_variables = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            mu, log_var = model.encode(data)
            z = model.reparameterize(mu, log_var)
            latent_variables.append(z.cpu().numpy())

    print("latent encoding finished")
    latent_variables = np.concatenate(latent_variables, axis=0)

    # Perform clustering in the original latent space
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(latent_variables)
    densities = kde.score_samples(latent_variables)
    high_density_points = latent_variables[densities > np.percentile(densities, 99.9)]  # Top 0.1% dense points
    print("high density points finished")
    
    kmeans = KMeans(n_clusters=10, random_state=SEED).fit(latent_variables)
    cluster_centers = kmeans.cluster_centers_
    print("cluster centers finished")
    
    # Decode points from high-density regions
    decoded_high_density = [model.decode(torch.from_numpy(point).float().to(device)).detach().cpu().numpy() for point in high_density_points]
    print("decoded high density finished")
    
    # Decode points from cluster centers
    decoded_cluster_centers = [model.decode(torch.from_numpy(center).float().to(device)).detach().cpu().numpy() for center in cluster_centers]
    print("decoded cluster centers finished")

    # Convert decoded sequences to ATGC format
    atgc_high_density = [onehot_to_seq(seq[:SEQ_LENGTH]) for seq in decoded_high_density]
    atgc_cluster_centers = [onehot_to_seq(seq[:SEQ_LENGTH]) for seq in decoded_cluster_centers]

    print("High Density Sequences:", atgc_high_density)
    print("Cluster Center Sequences:", atgc_cluster_centers)

    # Write to file
    write_to_fasta(atgc_high_density, OUTPUT_FILE_HD)
    write_to_fasta(atgc_cluster_centers, OUTPUT_FILE_CC)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Total processing time: %s seconds ---" % (time.time() - start_time))
