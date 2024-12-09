""" 
NGS Variational Autoencoder
Started 06/11/2023 by Zibin Zhao
This script defines and trains a Variational Autoencoder (VAE) for training on the 
NGS sequence data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.manifold import TSNE
import seaborn as sns

# Local application imports
from utility import compute_edit_distance #, onehot_to_seq , onehot_to_2d(have been processed in 2-NGS_preprocessing.py)

# Set global seed for reproducibility
SEED = 42
seed_everything(SEED)

# Hyperparameters
LAYER_1_SIZE = 256
LAYER_2_SIZE = 64
LATENT_SIZE = 64
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
TRAIN_SPLIT = 0.6
EPOCHS = 2000
WEIGHT_DECAY = 1e-4

# Constants
INPUT_SIZE = 273  # 39 * (3+4)  #* Please adjust accordingly to your cases
OUTPUT_SIZE = 273               #* Should be same as INPUT_SIZE
SEQ_LENGTH = 39 * 4
MAX_LENGTH = 39  
LOG_INTERVAL = 200
EARLY_LIMIT = 50

# Paths #* Please change here accordingly
WRITER_PATH = "runs/CSr3"
MODEL_NAME = './save_models/CSr3.pt'
DATA_PATH = './data/input_data_CSr3.pt'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize TensorBoard writer
writer = SummaryWriter(WRITER_PATH)

# Load input data
input_data = torch.load(DATA_PATH)
print(input_data.shape)

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


def loss_function(recon_x, x, mu, logvar):
    """ Custom loss function for VAE. """
    seq_loss = F.binary_cross_entropy(recon_x[:, :SEQ_LENGTH], x[:, :SEQ_LENGTH], reduction='none')
    structure_loss = F.binary_cross_entropy(recon_x[:, SEQ_LENGTH:], x[:, SEQ_LENGTH:], reduction='none')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return seq_loss.sum() + structure_loss.sum() + KLD


def train(model, optimizer, epoch, train_loader):
    """ Training loop for the model. """
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for data in progress_bar:
        data = data.float().to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data)
        loss = loss_function(recon_x, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item() / len(data))
    avg_train_loss = train_loss / len(train_loader.dataset)
    writer.add_scalar('training_loss', avg_train_loss, epoch)
    return avg_train_loss


def validate(model, epoch, val_loader):
    """ Validation loop for the model. """
    model.eval()
    val_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for data in progress_bar:
            data = data.float().to(device)
            recon_x, mu, logvar = model(data)
            val_loss += loss_function(recon_x, data, mu, logvar).item()
            progress_bar.set_postfix(loss=val_loss / len(val_loader.dataset))
    avg_val_loss = val_loss / len(val_loader.dataset)
    writer.add_scalar('validation_loss', avg_val_loss, epoch)
    return avg_val_loss


def test(epoch, model, test_loader, device):
    """ Test loop for the model. """
    model.eval()
    test_loss = 0
    total_seq_edit_distance = 0
    total_2d_edit_distance = 0
    total_seqs = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            recon_x, mu, logvar = model(data)
            test_loss += loss_function(recon_x, data, mu, logvar).item()
            comparison = [data.view(-1, OUTPUT_SIZE), recon_x.view(-1, OUTPUT_SIZE)]
            # Extract sequences
            original_seq = (comparison[0][0].detach().cpu().numpy()[:SEQ_LENGTH])
            original_2d = (comparison[0][0].detach().cpu().numpy()[SEQ_LENGTH:])
            reconstructed_seq = (comparison[1][0].detach().cpu().numpy()[:SEQ_LENGTH])
            reconstructed_2d = (comparison[1][0].detach().cpu().numpy()[SEQ_LENGTH:])
            
            # Compute the Edit distance
            total_seq_edit_distance += compute_edit_distance(original_seq, reconstructed_seq)
            total_2d_edit_distance += compute_edit_distance(original_2d, reconstructed_2d)
            total_seqs += 1
            print('-' * 89)
            print(f'SAMPLE {i+1}')
            print(f"Original sequence: {original_seq}")
            print(f"Reconstructed sequence: {reconstructed_seq}")
            print(f"Original 2D structure: {original_2d}")
            print(f"Reconstructed 2D structure: {reconstructed_2d}")
            
    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_seq_edit_distance = total_seq_edit_distance / total_seqs
    avg_2d_edit_distance = total_2d_edit_distance / total_seqs
    writer.add_scalar('test_loss', avg_test_loss, epoch)
    writer.add_scalar('edit_distance', avg_seq_edit_distance, epoch)
    writer.add_scalar('2d_edit_distance', avg_2d_edit_distance, epoch)
    print('-' * 89)
    print('====> Test set loss: {:.4f}'.format(avg_test_loss))
    print('====> Average Seq Edit distance: {:.4f}'.format(avg_seq_edit_distance))
    print('====> Average 2D Edit distance: {:.4f}'.format(avg_2d_edit_distance))
    return avg_test_loss, avg_seq_edit_distance, avg_2d_edit_distance


def main():
    """ Main training and evaluation loop. """
    # Prepare dataset
    dataset = MyDataset(input_data)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, and loss
    model = NGS_VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in tqdm(range(1, EPOCHS + 1), desc='Epochs'):
        train_loss = train(model, optimizer, epoch, train_loader)
        val_loss = validate(model, epoch, val_loader)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict()}, MODEL_NAME)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= EARLY_LIMIT:
            print("Stopping training early.")
            break

    # Testing
    test(epoch, model, test_loader, device)
    writer.close()

    '''t-SNE visualization of latent space (optinonal)'''
    # # Step 2: Extract Latent Variables
    # model.eval()
    # latent_variables = []
    # with torch.no_grad():
    #     for data in train_loader:
    #         data = data.to(device)
    #         mu, _ = model.encode(data)
    #         latent_variables.append(mu.cpu().numpy())

    # latent_variables = np.concatenate(latent_variables, axis=0)

    # # Step 3: Dimensionality Reduction
    # tsne = TSNE(n_components=2, random_state=SEED)
    # tsne_results = tsne.fit_transform(latent_variables)

    # # Step 4: Plotting
    # df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    # sns.scatterplot(x="TSNE1", y="TSNE2", data=df).set(title='Latent Space Visualization')
    # plt.show()

if __name__ == "__main__":
    main()
