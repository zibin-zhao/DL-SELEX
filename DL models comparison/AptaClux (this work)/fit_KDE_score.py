# fit_kde_on_R3.py
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import joblib

SEED=42
torch.manual_seed(SEED) 
np.random.seed(SEED)
MODEL_PATH = './save_models/CSr3.pt'          # <- R3 model
DATA_R3    = './data/input_data_CSr3.pt'      # <- R3 tensor (N,273)
OUT_KDE    = './save_models/CSr3_latentKDE.joblib'

LAYER_1_SIZE = 256
LAYER_2_SIZE = 64  # Changed from 16
LATENT_SIZE = 16   # Changed from 64
DROPOUT_RATE = 0.5
INPUT_SIZE = 273    #*Please adjust accordingly (same as the INPUT_SIZE in 3-train.py)
OUTPUT_SIZE = 273 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
@torch.no_grad()
def encode_mu(model, X, batch=512):
    Z=[]; loader=DataLoader(TensorDataset(torch.tensor(X).float()), batch_size=batch, shuffle=False)
    for (xb,) in loader:
        mu,_=model.encode(xb.to(device))
        Z.append(mu.cpu().numpy())
    return np.vstack(Z)

# load
Xr3 = torch.load(DATA_R3)
m = NGS_VAE().to(device)
m.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
m.eval()

Zr3 = encode_mu(m, Xr3)

# bandwidth CV on R3 only (no leakage)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': np.linspace(0.25,1.25,9)}, cv=5)
grid.fit(Zr3)
kde = grid.best_estimator_
joblib.dump({'kde': kde, 'bandwidth': grid.best_params_['bandwidth']}, OUT_KDE)
print(f"Saved KDE → {OUT_KDE} (best bw={grid.best_params_['bandwidth']})")
