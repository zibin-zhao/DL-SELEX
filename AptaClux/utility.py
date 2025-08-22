import Levenshtein as lev
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE

__all__ = ['compute_edit_distance', 'onehot_to_seq', 'decode_class', 'plot_t_sne']

def compute_edit_distance(seq1, seq2):
    """
    This function computes the Edit distance between two sequences
    """
    return lev.distance(seq1, seq2)

def write_to_fasta(sequences, output_path):
    with open(output_path, 'w') as fasta_file:
        for i, sequence in enumerate(sequences):
            fasta_file.write(f">Sequence_{i + 1}\n")  # FASTA header
            fasta_file.write(sequence + "\n")         # Sequence content
    print(f"FASTA file written to {output_path}")

def onehot_to_seq(onehot_seq):
    bases = ['A', 'T', 'C', 'G']
    seq = ''
    for i in range(0, len(onehot_seq), 4):  # each base is represented by 4 elements
        seq += bases[np.argmax(onehot_seq[i:i+4])]
    return seq

# def decode_class(onehot_class):
#     """Decode a one-hot encoded class back to its original class."""
#     class_dict = {0: 'CS', 1: 'ALD', 2: 'BE', 3: 'DCA', 4: 'DIS', 5: 'DOG', 6: 'PRO', 7: 'TES'}
#     class_index = np.argmax(onehot_class)  # Find the index of the '1' in the one-hot encoded class
#     return class_dict[class_index]  # Return the corresponding class


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
