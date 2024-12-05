import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything
from tqdm import tqdm
from torchsummary import summary
# from nltk.translate.bleu_score import sentence_bleu

# Local application imports
from utility import *


# Hyperparameter tuning
LAYER_1_SIZE = 256
LAYER_2_SIZE = 768      # match the coattention output shape
DROPOUT_RATE = 0.5
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
EPOCHS = 8000
WEIGHT_DECAY = 1e-4
LAMBDA = 0.5     # adjusted weight for co-attention loss

# Paths
WRITER_PATH = "runs/VAE_CoA/10072023/directcomparison2z_coatt02"
MODEL_NAME = './save_models/CoA_model/test_model.pt'

# Model parameters
INPUT_SIZE = 1850
OUTPUT_SIZE = 1850   # maximum length of sequence + SOS + EOS
LOG_INTERVAL = 1000
EARLY_LIMIT = 500
TRAIN_SPLIT = 0.6
#N_NEW_SEQ = 10
SEQ_LENGTH = 118 * 7 #826
MAX_LENGTH = 118    # maximum length of sequence + SOS + EOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
SEED = 42
seed_everything(SEED)

# Initialize TensorBoard writer
writer = SummaryWriter(WRITER_PATH)


DATA_PATH = './data/input_data.pt'
MASK_PATH = './data/input_mask.pt'   

# Load input data
input_data = torch.load(DATA_PATH)
masks = torch.load(MASK_PATH)   # Load precomputed masks
co_attention_output = torch.load('./model/co_attn_output.pt').squeeze(1).to(device) #shape: (194, 5, 768)
#print("input data shape: ", input_data.shape)

class MyDataset(Dataset):
    """Custom Dataset for loading the input data and the corresponding masks"""

    def __init__(self, data, masks, co_attention_output):
        self.data = data
        self.masks = masks
        self.co_att = co_attention_output

    def __getitem__(self, index):
        return self.data[index], self.masks[index], self.co_att[index]

    def __len__(self):
        return len(self.data)


class VAE(nn.Module):
    """Variational Autoencoder model"""

    def __init__(self, input_dim: int, layer1_dim: int, layer2_dim: int, output_dim: int):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer1_dim)
        self.fc21 = nn.Linear(layer1_dim, layer2_dim)
        self.fc22 = nn.Linear(layer1_dim, layer2_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc3 = nn.Linear(layer2_dim, layer1_dim)
        self.fc4 = nn.Linear(layer1_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def encode(self, x):
        h1 = self.dropout(self.leaky_relu(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.leaky_relu(self.fc3(z))
        h4 = self.leaky_relu(self.fc4(h3))
        return torch.sigmoid(h4)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, INPUT_SIZE))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def loss_function(recon_x, x, mu, logvar, mask, co_attention_output, z):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
    #print("BCE shape: ", BCE.shape)
    BCE = BCE * mask
    BCE = BCE.sum()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    #TODO: make sure the shape
    CoALoss = F.mse_loss(z, co_attention_output, reduction='sum')
    return BCE + KLD + LAMBDA * CoALoss

# def calculate_bleu(original_seq, reconstructed_seq):
#     # Tokenize sequences into words (assuming sequences are sentences)
#     original_tokens = original_seq.split()
#     reconstructed_tokens = reconstructed_seq.split()

#     # Compute BLEU score
#     bleu_score = sentence_bleu([original_tokens], reconstructed_tokens)
#     return bleu_score


def train(model, optimizer, epoch, train_loader):
    """Training function for one epoch"""
    model.train()
    train_loss = 0
    for batch_idx, (data, mask, co_att) in enumerate(train_loader):   
        data = data.to(device)
        mask = mask.to(device)
        co_att = co_att.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        #print("other shape: ", recon_batch.shape, data.shape, mu.shape, logvar.shape)
        loss = loss_function(recon_batch, data, mu, logvar, mask, co_att, z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        writer.add_scalar('training_loss', loss.item() / len(data), epoch * len(train_loader) + batch_idx)
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def validate(model, epoch, val_loader):
    """Validation function"""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, mask, co_att) in enumerate(val_loader):
            data = data.to(device)
            mask = mask.to(device)
            co_att = co_att.to(device)
            recon_batch, mu, logvar, z = model(data)
            val_loss += loss_function(recon_batch, data, mu, logvar, mask, co_att, z).item()
        val_loss /= len(val_loader.dataset)
    writer.add_scalar('validation_loss', val_loss, epoch)
    print('====> Validation set loss: {:.4f}'.format(val_loss))
    return val_loss


def test(epoch, model, test_loader, device):
    """Test function"""
    model.eval()
    test_loss = 0
    total_edit_distance = 0
    total_seqs = 0
    total_bleu_score = 0
    #print(len(test_loader))
    for i, (data, mask, co_att) in enumerate(test_loader):
        data = data.to(device)
        mask = mask.to(device)
        co_att = co_att.to(device)
        recon_batch, mu, logvar, z = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, mask, co_att, z).item()
        #print(data.shape)
        #print("len of testloader: ", len(test_loader))
        # n = data.size(0)
        # print(n)
        # 1102 output size
        comparison = [data.view(-1, OUTPUT_SIZE), recon_batch.view(-1, OUTPUT_SIZE)]   
        #print("len of comparison: ", len(comparison))
        # print("shape of comparison original seq: ", comparison[0].shape)
        # print("shape of comparison reconstructed seq: ", comparison[1].shape)
        
        original_seq = onehot_to_seq(comparison[0][0].detach().cpu().numpy()[:SEQ_LENGTH]) # get the first 580 elements for sequence
        reconstructed_seq = onehot_to_seq(comparison[1][0].detach().cpu().numpy()[:SEQ_LENGTH]) # get the first 580 elements for sequence
        
        # Compute the Edit distance and increment the total
        total_edit_distance += compute_edit_distance(original_seq, reconstructed_seq)
        total_seqs += 1
        
        print(f'Original sequence {i+1}: {original_seq}')
        print(f'Reconstructed sequence {i+1}: {reconstructed_seq}')
            
        # Compute the BLEU score and increment the total
        # bleu_score = calculate_bleu(original_seq, reconstructed_seq)
        # print(bleu_score)
        # total_bleu_score += bleu_score

    # average_bleu_score = total_bleu_score / total_seqs
    # print('====> Average BLEU score: {:.4f}'.format(average_bleu_score))
    
    test_loss /= len(test_loader.dataset)
    average_edit_distance = total_edit_distance / total_seqs
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Average Edit distance: {:.4f}'.format(average_edit_distance))
    writer.add_scalar('test_loss', test_loss, epoch)


def main():
    """Main function"""
    dataset = MyDataset(input_data, masks, co_attention_output)     # pass both data and masks to the dataset

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)   
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True) 

    
    model = VAE(INPUT_SIZE, LAYER_1_SIZE, LAYER_2_SIZE, OUTPUT_SIZE).to(device)
    
    # L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in tqdm(range(1, EPOCHS + 1)):
        
        # training model
        train(model, optimizer, epoch, train_loader)

        # validating model
        val_loss = validate(model, epoch, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving model...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_NAME)
            epochs_without_improvement = 0
        elif val_loss >= best_val_loss:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_LIMIT:
            print("Stopping training early.")
            break
        #print(epochs_without_improvement)
    
    # testing model
    test(epoch, model, test_loader, device)
    writer.close()



if __name__ == "__main__":
    # Main training loop
    main()
    
    # Model summary
    # model = VAE(INPUT_SIZE, LAYER_1_SIZE, LAYER_2_SIZE, OUTPUT_SIZE).to(device)
    # summary(model, input_size=(1, INPUT_SIZE))




