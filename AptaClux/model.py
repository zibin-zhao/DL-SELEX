import torch
import torch.nn as nn
import torch.nn.functional as F
    
LAYER_1_SIZE = 256
LATENT_SIZE = 64
LAYER_2_SIZE = 64
DROPOUT_RATE = 0.5
INPUT_SIZE = 273
OUTPUT_SIZE = 273

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