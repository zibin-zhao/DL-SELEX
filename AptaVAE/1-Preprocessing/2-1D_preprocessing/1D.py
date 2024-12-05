'''1D preprocessing of the sequence and target by Zibin Zhao'''

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
ENCODING_DICT = {'<': [1, 0, 0, 0, 0, 0],  # encoding for SOS
                 'A': [0, 1, 0, 0, 0, 0], 
                 'T': [0, 0, 1, 0, 0, 0], 
                 'C': [0, 0, 0, 1, 0, 0], 
                 'G': [0, 0, 0, 0, 1, 0], 
                 '>': [0, 0, 0, 0, 0, 1]}  # encoding for EOS

'''DEFINE the maximum length of sequence + SOS + EOS'''
MAX_LENGTH = 100 + 2  # maximum length of sequence + SOS + EOS
PATH = "../data/refined_primer/train_dataset_refined_primer.csv"

def load_data(path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(path)

def morgan_fingerprint(smiles):
    """Generate Morgan fingerprints for a SMILES string."""
    # Convert the SMILES string to a molecule
    molecule = Chem.MolFromSmiles(smiles)
    
    # Generate the Morgan fingerprint for the molecule
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
    
    # Convert the fingerprint to a list of bits
    return np.array(fingerprint)

def one_hot_encode_seq(sequence):
    """One-hot encode a sequence and create a corresponding mask."""
    # Add the SOS token to the start of the sequence and the EOS token to the end
    sequence = '<' + sequence + '>'
    
    encoding = [ENCODING_DICT[nucleotide] for nucleotide in sequence]
    
    padding = MAX_LENGTH - len(sequence)
    
    encoding += [[0, 0, 0, 0, 0, 0]] * padding
    
    mask = [[1, 1, 1, 1, 1, 1]] * len(sequence) + [[0, 0, 0, 0, 0, 0]] * padding
    
    return encoding, mask

def one_hot_encode_2d(dot_bracket_string):
    # Define the mapping from characters to vectors
    mapping = {
        '.': [1, 0, 0],
        '(': [0, 1, 0],
        ')': [0, 0, 1],
    }
    
    # Encode the string
    encoded_string = [mapping[char] for char in dot_bracket_string]
    
    return encoded_string

def one_hot_encode_class(classes):
    """One-hot encode a categorical variable."""
    encoder = OneHotEncoder(sparse=False)
    classes_reshaped = classes.reshape(-1, 1)
    classes_one_hot = encoder.fit_transform(classes_reshaped)
    return classes_one_hot

def normalize_score(score):
    """Normalize a numerical variable to have zero mean and unit variance."""
    scaler = StandardScaler()
    score_normalized = scaler.fit_transform(score.reshape(-1, 1))
    return score_normalized

def reshape_mask(mask, target_size):
    """Reshape the mask to match the target size."""
    mask = mask.view(mask.shape[0], -1)  # flatten the mask
    padding_size = target_size - mask.size(1)
    zeros = torch.zeros(mask.size(0), padding_size, device=mask.device)
    mask = torch.cat([mask, zeros], dim=-1)
    return mask

def main(PATH):
    """Main function to load and preprocess the data."""
    df = load_data(PATH)
    sequences = df['modified seq'].values
    targets = df['Target_SMILES'].values
    score = df['Relative Score'].values
    mol_class = df['Class'].values


    # Compute one-hot encoding and masks for sequences
    sequences, masks = zip(*[one_hot_encode_seq(sequence) for sequence in sequences])
    sequences = torch.tensor(np.array(list(sequences)).astype(np.float32))
    masks = torch.tensor(np.array(list(masks)).astype(np.float32))

    # Reshape the sequences and masks tensors
    sequences_tensor = sequences.view(sequences.shape[0], -1)
    masks_tensor = masks.view(masks.shape[0], -1)

    # Compute Morgan fingerprints for targets
    targets = df['Target_SMILES'].apply(morgan_fingerprint)
    targets_tensor = torch.tensor(np.array(targets.tolist()).astype(np.float32))

    # Compute one-hot encoding for mol_class and normalize score
    mol_class_tensor = torch.tensor(one_hot_encode_class(mol_class).astype(np.float32))
    #print(mol_class_tensor, mol_class_tensor.shape)
    #score_tensor = torch.tensor(normalize_score(score).astype(np.float32))
    score_tensor = torch.tensor(score.reshape(-1, 1).astype(np.float32))

    # encode 2d structure into onehot
    #seq_2d_tensor = torch.tensor(one_hot_encode_2d(seq_2d))
    
    # Concatenate tensors for input data
    input_data = torch.cat((sequences_tensor, targets_tensor, mol_class_tensor, score_tensor), dim=1)
    
    # 118 max length, 7 one-hot encoding, 1024 morgan fingerprint, 8 mol_class, 1 score
    masks_tensor = reshape_mask(masks_tensor, target_size = MAX_LENGTH * 6 + 1024 + 8 + 1)   
    
    return input_data, masks_tensor

if __name__ == "__main__":
    input_data, masks_tensor = main(PATH)
    
    print(f"input_data: {input_data}")
    print(f"input_data.shape: {input_data.shape}")
    print(f"masks_tensor.shape: {masks_tensor.shape}")
    
    # Save the tensors
    torch.save(input_data, "../data/refined_primer/MF_primer_input.pt")
    torch.save(masks_tensor, "../data/refined_primer/MF_primer_mask.pt")
