import Levenshtein as lev
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

__all__ = ['compute_edit_distance', 
           'onehot_to_seq', 
           'decode_class', 
           'plot_t_sne', 
           'write_to_fasta', 
           'onehot_to_2d']


def write_to_fasta(sequences, file_name):
    with open(file_name, 'w') as file:
        for i, seq in enumerate(sequences):
            file.write(f">Sequence_{i}\n{seq}\n")

def compute_edit_distance(seq1, seq2):
    """
    This function computes the Edit distance between two sequences
    """
    return lev.distance(seq1, seq2)

def onehot_to_seq(onehot_seq):
    bases = ['A', 'T', 'C', 'G']
    seq = ''
    for i in range(0, len(onehot_seq), 4):      # as we flatten the vector, hence every 5 samples is a base
        try:
            #print(len(onehot_seq))
            seq += bases[np.argmax(onehot_seq[i:i+4])]
        except IndexError:
            print(f"Error at index {i} with onehot_seq length {len(onehot_seq)}")
            break
    return seq


def onehot_to_2d(onehot_2d):
    bases = ['.', '(', ')']
    seq_2d = ''
    for i in range(0, len(onehot_2d), 3):      # as we flatten the vector, hence every 5 samples is a base
        try:
            #print(len(onehot_seq))
            seq_2d += bases[np.argmax(onehot_2d[i:i+3])]
        except IndexError:
            print(f"Error at index {i} with onehot_seq length {len(onehot_2d)}")
            break
    return seq_2d

def decode_class(onehot_class):
    """Decode a one-hot encoded class back to its original class."""
    class_dict = {0: 'CS', 1: 'ALD', 2: 'BE', 3: 'DCA', 4: 'DIS', 5: 'DOG', 6: 'PRO', 7: 'TES'}
    onehot_class_tensor = torch.tensor(onehot_class)
    top2_indices = torch.topk(onehot_class_tensor, 2).indices  # Find the indices of the top 2 classes
    return [class_dict[index.item()] for index in top2_indices]  # Return the corresponding classes
 # def calculate_bleu(original_seq, reconstructed_seq):
#     # Tokenize sequences into words (assuming sequences are sentences)
#     original_tokens = original_seq.split()
#     reconstructed_tokens = reconstructed_seq.split()

#     # Compute BLEU score
#     bleu_score = sentence_bleu([original_tokens], reconstructed_tokens)
#     return bleu_score

def plot_t_sne(latent_space, epoch, output_dir):
    tsne = TSNE(n_components=2, random_state=42)
    latent_2D = tsne.fit_transform(latent_space.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_2D[:, 0], latent_2D[:, 1], cmap='viridis')
    plt.title(f'Latent space t-SNE plot at epoch {epoch}', fontsize=20)
    plt.xlabel('t-SNE Component 1', fontsize=15)
    plt.ylabel('t-SNE Component 2', fontsize=15)
    plt.savefig(os.path.join(output_dir, f'tsne_plot_epoch_{epoch}.png'))
    plt.close()