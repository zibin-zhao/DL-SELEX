'''Started 28/11/2023 by Zibin Zhao, encoding for seq + 2D with onehot
    input: combined.csv file with seq + 2d with max length
    output: encoding tensor'''

import pandas as pd
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Encoding dictionary and maximum length
ENCODING_DICT = {'A': [1, 0, 0, 0], 
                 'T': [0, 1, 0, 0], 
                 'C': [0, 0, 1, 0], 
                 'G': [0, 0, 0, 1]}


INPUT_PATH = "../raw_NGS/DHEA/r7/DHEAr7_combined.csv"
OUTPUT_NAME = "../data/input_data_DHEAr7.pt"
MAX_LENGTH = 32  # N32


def load_data(path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)


def one_hot_encode_seq(sequence, MAX_LENGTH):
    """One-hot encode a sequence."""
    encoding = [ENCODING_DICT[nucleotide] for nucleotide in sequence]
    return encoding

def one_hot_encode_2d(dot_bracket_string):
    # Define the mapping from characters to vectors
    mapping = {
        '.': [1, 0, 0],
        '(': [0, 1, 0],
        ')': [0, 0, 1],
    }
    
    # Encode the string
    #print(dot_bracket_string)
    encoded_string = [mapping[char] for char in dot_bracket_string]
    
    return encoded_string


def main(PATH):
    """Main function to load and preprocess the data."""
    df = load_data(PATH)
    sequences = df['Sequence'].values
    seq_2d = df['Sequence_2d'].values
    print(len(seq_2d[0]))
    
    # Compute one-hot encoding and masks for sequences
    sequences= [one_hot_encode_seq(sequence, MAX_LENGTH) for sequence in sequences]
    
    #print(f'sequences shape: {np.array(sequences).shape}, masks shape: {np.array(masks).shape}')
    
    sequences = torch.tensor(np.array(list(sequences)).astype(np.float32))
    print(sequences.shape)
    # Reshape the sequences and masks tensors
    sequences_tensor = sequences.view(sequences.shape[0], -1)

    # Compute Morgan fingerprints for targets
    #targets = df['Target_SMILES'].apply(morgan_fingerprint)
    #targets_tensor = torch.tensor(np.array(targets.tolist()).astype(np.float32))

    #score_tensor = torch.tensor(score.reshape(-1, 1).astype(np.float32))

    # encode 2d structure into onehot
    # List comprehension to encode each 2d structure
    seq_2d_encoded_list = [one_hot_encode_2d(structure) for structure in seq_2d]
    
    # Convert the list of lists to a tensor
    seq_2d_tensor = torch.tensor(seq_2d_encoded_list)
    seq_2d_tensor = seq_2d_tensor.view(seq_2d_tensor.shape[0], -1)
    print(seq_2d_tensor.shape)
    # Concatenate tensors for input data
    #print(sequences_tensor.shape, seq_2d_tensor.shape)
    input_data = torch.cat((sequences_tensor, seq_2d_tensor), dim=1)
    
    # Seq shape (n, 39, 4)
    # 2d shape (n, 39, 3)
    
    return input_data

if __name__ == "__main__":
    input_data = main(INPUT_PATH)
    # Should be the same shape for both
    print(f"input_data.shape: {input_data.shape}")
    
    # Save the tensors
    # torch.save(input_data, OUTPUT_NAME)
    # print("Saved input data!")
