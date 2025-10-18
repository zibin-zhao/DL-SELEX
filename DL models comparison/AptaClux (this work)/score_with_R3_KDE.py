# score_with_R3_KDE.py
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib

SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== EDIT THESE ====
MODEL_PATH = './save_models/CSr3.pt'           # same R3 model as before
DATA_EVAL  = './data/CS_panel_input.pt'        # or './data/input_data_CSr5.pt'
ID_CSV     = './CS_panel.csv'                  # or './R5_pool.csv'
ID_COL     = 'panel_id'                        # or 'seq_id'
KDE_PATH   = './save_models/CSr3_latentKDE.joblib'
OUT_SCORES = './scores_CS_panel_clux.csv'      # or 'scores_R5_CS_clux.csv'
INPUT_SIZE = 273
L1,L2,LAT,DR = 256,64,16,0.5
# =====================

class NGS_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(INPUT_SIZE, L1)
        self.fc21 = nn.Linear(L1, LAT)   # mu
        self.fc22 = nn.Linear(L1, LAT)   # log_var
        self.fc3  = nn.Linear(LAT, L2)   # (unused here)
        self.fc4  = nn.Linear(L2, INPUT_SIZE)
        self.drop = nn.Dropout(DR)
    def encode(self, x):
        h = F.relu(self.fc1(x)); h = self.drop(h)
        return self.fc21(h), self.fc22(h)

@torch.no_grad()
def encode_mu(model, X, batch=512):
    Z=[]; loader=DataLoader(TensorDataset(torch.tensor(X).float()), batch_size=batch, shuffle=False)
    for (xb,) in loader:
        mu,_=model.encode(xb.to(device))
        Z.append(mu.cpu().numpy())
    return np.vstack(Z)

# load eval features & ids
X = torch.load(DATA_EVAL)
ids = pd.read_csv(ID_CSV)[ID_COL].values
assert X.ndim==2 and X.shape[1]==INPUT_SIZE, f"Expected (*,{INPUT_SIZE}), got {X.shape}"
assert len(ids)==X.shape[0], "ID count must match rows in DATA_EVAL"

# load model
m = NGS_VAE().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
m.load_state_dict(state, strict=False); m.eval()

# encode eval to μ and score with KDE
Z = encode_mu(m, X)
bundle = joblib.load(KDE_PATH); kde = bundle['kde']
scores = kde.score_samples(Z)  # higher = better
pd.DataFrame({ID_COL: ids, 'score': -scores}).to_csv(OUT_SCORES, index=False)
print(f"Saved {OUT_SCORES} with {len(scores)} rows.")
