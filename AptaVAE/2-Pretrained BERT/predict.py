"""This script is created by Zibin Zhao 22/05/2023 for generating new sequences using the trained VAE model"""

# Standard library imports
import torch
import numpy as np

# Local application imports
from utility import *
from model import VAE


N_NEW_SEQ = 5000
SEQ_LENGTH = 826
LAYER_2_SIZE = 256
PATH = "./save_models/fine_tune_models/cortisol/256_256_0.9_bs128_0.0001_ver2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
model.load_state_dict(torch.load(PATH)['model_state_dict'])
model.eval()

decoded_seq = []
with torch.no_grad():
    sample = torch.randn(N_NEW_SEQ, LAYER_2_SIZE).to(device)
    sample = model.decode(sample).cpu()
    
    for i in range(N_NEW_SEQ):
        
        decoded_seq.append(np.array((onehot_to_seq(sample.numpy()[i, :SEQ_LENGTH]))))
        
        print(f'Decoded sample {i+1}:')
        print(onehot_to_seq(sample.numpy()[i, :SEQ_LENGTH])) # print decoded sample sequence
    # print(seq)
    # print(type(seq), len(seq), np.shape(seq))
    
#np.savetxt('decoded_seq_02062023_cortisol_refined.csv', decoded_seq, delimiter=',', fmt='%s')
