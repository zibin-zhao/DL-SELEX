'''Batch generated analysis (BGA) for 
    stable model sampling by Zibin Zhao'''


import torch
import numpy as np
import pandas as pd
from utility import *
from model_64 import VAE
import matplotlib.pyplot as plt

BATCH_SIZE = 200
N_BATCHES = 100
SEQ_LENGTH = 118 * 6
LAYER_2_SIZE = 256
THRESHOLD = 0.9

INPUT_SIZE = 1
layer1_dim = 256
layer2_dim = 256
OUTPUT_SIZE = 1741

PATH = "./save_models/refined_primer/best_model_8.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(INPUT_SIZE, layer1_dim, layer2_dim, OUTPUT_SIZE).to(device)
model.load_state_dict(torch.load(PATH)['model_state_dict'])
model.eval()

decoded_seqs = []
decode_classes = []
decoded_scores = []

mean_scores = []
std_scores = []
proportions_above_threshold = []

for batch in range(N_BATCHES):
    with torch.no_grad():
        sample = torch.randn(BATCH_SIZE, LAYER_2_SIZE).to(device)
        sample = model.decode(sample).cpu()

        for i in range(BATCH_SIZE):
            decoded_seq = np.array((onehot_to_seq(sample.numpy()[i, :SEQ_LENGTH])))
            decoded_mol_class = decode_class(sample.numpy()[i, 1732:1740])[0]
            decoded_score = sample.numpy()[i, -1]

            decoded_seqs.append(decoded_seq)
            decode_classes.append(decoded_mol_class)
            decoded_scores.append(decoded_score)

            print(f'Decoded sample {i+1} of batch {batch+1}:')
            print(f'Decoded sequence: {decoded_seq}')
            print(f'Decoded class: {decoded_mol_class}')
            print(f'Decoded score: {decoded_score}')

        # Calculate and store metrics after each batch
        mean_scores.append(np.mean(decoded_scores))
        std_scores.append(np.std(decoded_scores))
        proportions_above_threshold.append(sum(np.array(decoded_scores) > THRESHOLD) / len(decoded_scores))

        print(f'After batch {batch+1}:')
        print(f'Mean score: {mean_scores[-1]}')
        print(f'Standard deviation of scores: {std_scores[-1]}')
        print(f'Proportion of scores above {THRESHOLD}: {proportions_above_threshold[-1]}')

df = pd.DataFrame({
    'Decoded Sequences': decoded_seqs,
    'Decoded Classes': decode_classes,
    'Decoded Scores': decoded_scores
})
df.to_csv('BGA_decoded_results.csv', index=False)

metrics_df = pd.DataFrame({
    'Batch': list(range(1, N_BATCHES+1)),
    'Mean Score': mean_scores,
    'STD Score': std_scores,
    'Proportion Above Threshold': proportions_above_threshold
})
metrics_df.to_csv('BGA_metrics.csv', index=False)



# Plot Mean Score over batches
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(metrics_df['Batch'], metrics_df['Mean Score'])
plt.xlabel('Batch')
plt.ylabel('Mean Score')
plt.title('Mean Score Over Batches')

# Plot STD Score over batches
plt.subplot(1, 3, 2)
plt.plot(metrics_df['Batch'], metrics_df['STD Score'])
plt.xlabel('Batch')
plt.ylabel('STD Score')
plt.title('STD Score Over Batches')

# Plot Proportion Above Threshold over batches
plt.subplot(1, 3, 3)
plt.plot(metrics_df['Batch'], metrics_df['Proportion Above Threshold'])
plt.xlabel('Batch')
plt.ylabel('Proportion Above Threshold')
plt.title('Proportion Above Threshold Over Batches')

# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.show()
