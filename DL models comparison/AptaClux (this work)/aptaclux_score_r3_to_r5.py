# aptaclux_score_r3_to_r5.py
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from pytorch_lightning import seed_everything

# ---- config ----
SEED = 42
MODEL_PATH = './save_models/CSr3.pt'          # train on R3
DATA_R3 = './data/input_data_CSr3.pt'         # score model fit
DATA_R5 = './data/input_data_CSr5.pt'         # score targets
OUT_SCORES = './scores_R5_CS_clux.csv'        # seq_id,score
INPUT_SIZE = OUTPUT_SIZE = 273
LAYER_1_SIZE, LAYER_2_SIZE, LATENT_SIZE = 256, 64, 64
DROPOUT_RATE = 0.5
BATCH = 512

seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NGS_VAE(nn.Module):
    def __init__(self, layer1_size=LAYER_1_SIZE, latent_size=LATENT_SIZE,
                 layer2_size=LAYER_2_SIZE, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, layer1_size)
        self.fc21 = nn.Linear(layer1_size, latent_size)  # mu
        self.fc22 = nn.Linear(layer1_size, latent_size)  # logvar
        self.fc3 = nn.Linear(latent_size, layer2_size)
        self.fc4 = nn.Linear(layer2_size, OUTPUT_SIZE)
        self.dropout = nn.Dropout(dropout_rate)
    def encode(self, x):
        h = F.relu(self.fc1(x)); h = self.dropout(h)
        return self.fc21(h), self.fc22(h)
    def forward(self, x):
        mu, logv = self.encode(x)
        return torch.sigmoid(self.fc4(F.relu(self.fc3(mu)))), mu, logv  # decode(mu) not needed here

@torch.no_grad()
def encode_mu(model, X, batch=BATCH):
    model.eval()
    Z = []
    loader = DataLoader(torch.tensor(X).float(), batch_size=batch, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        mu, _ = model.encode(xb)
        Z.append(mu.detach().cpu().numpy())
    return np.vstack(Z)

def main():
    # load data
    X_r3 = torch.load(DATA_R3)  # shape [N3, D]
    X_r5 = torch.load(DATA_R5)  # shape [N5, D]

    # load model
    model = NGS_VAE().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # latent means
    Z_r3 = encode_mu(model, X_r3)
    Z_r5 = encode_mu(model, X_r5)

    # KDE on R3 (bandwidth CV on R3 only; no R5 peeking)
    params = {'bandwidth': np.linspace(0.25, 1.25, 9)}
    gs = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=5)
    gs.fit(Z_r3)
    kde = gs.best_estimator_

    # log-density scores for R5
    scores = kde.score_samples(Z_r5)  # higher = better

    # save
    pd.DataFrame({'seq_id': np.arange(len(scores)), 'score': scores}).to_csv(OUT_SCORES, index=False)
    print(f"Saved {OUT_SCORES} with {len(scores)} rows. Best bw={gs.best_params_['bandwidth']:.2f}")

if __name__ == '__main__':
    main()
