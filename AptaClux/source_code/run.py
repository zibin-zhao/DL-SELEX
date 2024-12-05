import os
import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from nupack import Model, mfe
from tqdm import tqdm
from utility import onehot_to_seq, write_to_fasta
from Bio import pairwise2
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# Set global seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Hyperparameters and file paths
LAYER_1_SIZE = 256
LATENT_SIZE = 64
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
TRAIN_SPLIT = 0.6
WEIGHT_DECAY = 1e-4
EARLY_LIMIT = 50

ENCODING_DICT = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
OUTPUT_FILE_HD = "hd_output.fasta"
OUTPUT_FILE_CC = "cc_output.fasta"
WRITER_PATH = "runs"
MODEL_NAME = 'trained_model.pt'


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='AptaClux Model Training and Sampling for aptamer candidates generation.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input FASTA file.')
    parser.add_argument('-o', '--output', type=str, default='output.txt', help='Path to the output text file for results.')
    parser.add_argument('-nb', '--num_devices', type=int, default=1, help='Number of GPUs (or CPUs if none) to use.')
    parser.add_argument('-max', '--max_epoch', type=int, default=2000, help='Maximum number of epochs for training.')
    parser.add_argument('-tp', '--temperature', type=float, default=25, help='Temperature in degrees Celsius for the model.')
    parser.add_argument('-ions', '--ions', type=float, default=0.147, help='Ion concentration, default is sodium=0.147M.')
    parser.add_argument('-oligos', '--oligos', type=str, default='dna', help='Oligonucleotide type, default is DNA.')
    parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()

def set_device(num_devices):
    """Set device based on the number of GPUs/CPUs specified."""
    if torch.cuda.is_available() and num_devices > 0:
        return torch.device(f"cuda:{num_devices-1}")
    return torch.device("cpu")


writer = SummaryWriter(WRITER_PATH)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class NGS_VAE(nn.Module):
    def __init__(self):
        super(NGS_VAE, self).__init__()
        self.fc1 = nn.Linear(273, LAYER_1_SIZE)
        self.fc21 = nn.Linear(LAYER_1_SIZE, LATENT_SIZE)
        self.fc22 = nn.Linear(LAYER_1_SIZE, LATENT_SIZE)
        self.fc3 = nn.Linear(LATENT_SIZE, 64)
        self.fc4 = nn.Linear(64, 273)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        return self.fc21(h1), self.fc22(h1)

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

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        for line in file:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence)
    return sequences

def calculate_dppn(sequences):
    dppn_data = []
    for seq in sequences:
        try:
            mfe_structure = mfe(strands=[seq], model=model)
            dppn_data.append(mfe_structure[0].structure.dotparensplus())
        except Exception as e:
            print(f"Error calculating dppn for sequence {seq}: {e}")
            print('-'*50)
            dppn_data.append("")
    return dppn_data

def one_hot_encode_seq(sequence):
    encoding = [ENCODING_DICT.get(nuc, [0, 0, 0, 0]) for nuc in sequence]
    return encoding + [[0, 0, 0, 0]] * (32 - len(encoding))

def one_hot_encode_2d(dot_bracket_string):
    mapping = {'.': [1, 0, 0], '(': [0, 1, 0], ')': [0, 0, 1]}
    encoded_string = [mapping.get(char, [0, 0, 0]) for char in dot_bracket_string]
    return encoded_string + [[0, 0, 0]] * (32 - len(encoded_string))

def preprocess(input_fasta):
    sequences = read_fasta(input_fasta)
    dppns = calculate_dppn(sequences)
    encoded_seqs = [one_hot_encode_seq(seq) for seq in sequences]
    encoded_dppns = [one_hot_encode_2d(dppn) for dppn in dppns]
    seq_tensor = torch.tensor(encoded_seqs, dtype=torch.float32).view(len(sequences), -1)
    dppn_tensor = torch.tensor(encoded_dppns, dtype=torch.float32).view(len(dppns), -1)
    return torch.cat((seq_tensor, dppn_tensor), dim=1)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for data in tqdm(train_loader, desc='Training'):
        data = data.float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.float().to(device)
            recon_batch, mu, logvar = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar).item()
    return val_loss / len(val_loader.dataset)


def msa_progressive(sequences, label):
    """Performs MSA and returns aligned sequences without directly writing to file."""
    print(f"Performing MSA for {label} sequences...")
    n = len(sequences)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            alignments = pairwise2.align.globalxx(sequences[i], sequences[j])
            distance_matrix[i, j] = distance_matrix[j, i] = 1 / (1 + alignments[0].score)

    condensed_dist_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist_matrix, method='average')
    clusters = fcluster(linkage_matrix, t=1.5, criterion='inconsistent')
    
    aligned_sequences = []
    for cluster in np.unique(clusters):
        cluster_seqs = [sequences[i] for i in range(n) if clusters[i] == cluster]
        alignment = pairwise2.align.globalms(cluster_seqs[0], cluster_seqs[1], 2, -1, -2, -1)
        aligned_sequences.append(alignment[0].seqA)
    
    return aligned_sequences

def sample_and_decode(model, latent_variables, device, output_file):
    print("Step 3: Starting sampling from latent space...")
    # High-density sampling
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(latent_variables)
    densities = kde.score_samples(latent_variables)
    high_density_points = latent_variables[densities > np.percentile(densities, 80)]
    hd_decoded = [model.decode(torch.from_numpy(point).float().to(device)).detach().cpu().numpy() for point in high_density_points]
    atgc_high_density = [onehot_to_seq(seq[:156]) for seq in hd_decoded]
    
    # Cluster center sampling
    kmeans = KMeans(n_clusters=10, random_state=SEED).fit(latent_variables)
    cluster_centers = kmeans.cluster_centers_
    cc_decoded = [model.decode(torch.from_numpy(center).float().to(device)).detach().cpu().numpy() for center in cluster_centers]
    atgc_cluster_centers = [onehot_to_seq(seq[:156]) for seq in cc_decoded]

    # MSA for High-Density and Cluster-Center sequences
    hd_aligned = msa_progressive(atgc_high_density, "High-Density")
    cc_aligned = msa_progressive(atgc_cluster_centers, "Cluster-Center")

    print("Sampling completed. Writing results to file...")

    # Write aligned sequences to file
    with open(output_file, 'w') as f:
        f.write("MSA Result for High-Density:\n")
        for seq in hd_aligned:
            f.write(seq + '\n')
        f.write("\nMSA Result for Cluster-Center:\n")
        for seq in cc_aligned:
            f.write(seq + '\n')
    print(f"MSA sequences written to {output_file}.")
    
# Updated train function without tqdm
def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for data in train_loader:  
        data = data.float().to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)



def main():
    args = parse_arguments()
    device = set_device(args.num_devices)

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize the NUPACK model with user-defined temperature, ion concentration, and oligos
    global model
    model = Model(material=args.oligos, celsius=args.temperature, sodium=args.ions)
    EPOCHS = args.max_epoch
    input_data = preprocess(args.input)

    print('-'*50)
    print('User-defined parameters:')
    print(f"Using random seed: {args.seed}")
    print(f"Temperature: {args.temperature}°C, Ion Concentration: Sodium={args.ions}M, Oligos: {args.oligos}")
    print(f"Max epochs: {args.max_epoch}")
    print('-'*50)
    print(f'Step 1: {args.input} successfully loaded.')
    print('-'*50)
    dataset = MyDataset(input_data)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NGS_VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    no_improve_count = 0
    print("Step 2: Starting training process...")
    print('-'*50)

    # Use tqdm for the entire training process over all epochs
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Progress"):
        train_loss = train(model, optimizer, train_loader, device)
        val_loss = validate(model, val_loader, device)
        
        writer.add_scalars('Loss', {'train': train_loss, 'validation': val_loss}, epoch)

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_NAME)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= EARLY_LIMIT:
                print("Early stopping due to no improvement.")
                break

    # Print final losses after training completes
    print(f"Final Train Loss: {train_loss:.4f}, Final Validation Loss: {val_loss:.4f}")
    print('-'*50)

    model.load_state_dict(torch.load(MODEL_NAME, weights_only=True))
    model.eval()
    latent_vars = [model.encode(data.float().to(device))[0].detach().cpu().numpy() for data in train_loader]
    latent_vars = np.concatenate(latent_vars, axis=0)
    sample_and_decode(model, latent_vars, device, args.output)

if __name__ == "__main__":
    main()