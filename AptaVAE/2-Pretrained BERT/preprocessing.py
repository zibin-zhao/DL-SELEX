import pandas as pd
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import torch.nn as nn


ENCODING_DICT = {'<': [1, 0, 0, 0, 0, 0, 0],  # encoding for SOS
                 'A': [0, 1, 0, 0, 0, 0, 0], 
                 'T': [0, 0, 1, 0, 0, 0, 0], 
                 'C': [0, 0, 0, 1, 0, 0, 0], 
                 'G': [0, 0, 0, 0, 1, 0, 0], 
                 'N': [0, 0, 0, 0, 0, 1, 0],
                 '>': [0, 0, 0, 0, 0, 0, 1]}  # encoding for EOS
MAX_LENGTH = 118    # maximum length of sequence + SOS + EOS
PATH = "../data/our_dataset.csv"


# Generate Morgan fingerprints
def morgan_fingerprint(smiles):
    # Convert the SMILES string to a molecule
    molecule = Chem.MolFromSmiles(smiles)
    
    # Generate the Morgan fingerprint for the molecule
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024)
    
    # Convert the fingerprint to a list of bits
    return np.array(fingerprint)


def one_hot_encode(sequence):
    # Add the SOS token to the start of the sequence and the EOS token to the end
    sequence = '<' + sequence + '>'
    
    encoding = [ENCODING_DICT[nucleotide] for nucleotide in sequence]
    
    padding = MAX_LENGTH - len(sequence)
    
    encoding += [[0, 0, 0, 0, 0, 0, 0]] * padding
    
    mask = [[1, 1, 1, 1, 1, 1, 1]] * len(sequence) + [[0, 0, 0, 0, 0, 0, 0]] * padding
    
    return encoding, mask

# defien a mask for the input sequence with zero masking
def reshape_mask(mask, target_size):
    mask = mask.view(mask.shape[0], -1)  # flatten the mask
    padding_size = target_size - mask.size(1)
    zeros = torch.zeros(mask.size(0), padding_size, device=mask.device)
    mask = torch.cat([mask, zeros], dim=-1)
    return mask


def get_molecule_class(df):
    


def main(PATH):
    df = pd.read_csv(PATH)
    sequences = df['Sequence'].values
    targets = df['Target_SMILES'].values

    # compute one hot encoding and mask
    sequences, masks = zip(*[one_hot_encode(sequence) for sequence in sequences])

    sequences = torch.tensor(np.array(list(sequences)).astype(np.float32))
    masks = torch.tensor(np.array(list(masks)).astype(np.float32))
    #print(type(sequences), type(masks))

    # Reshape the tensors
    sequences_tensor = sequences.view(sequences.shape[0], -1)
    masks_tensor = masks.view(masks.shape[0], -1)
    #print(sequences_tensor.shape, masks_tensor.shape)

    # Compute Morgan fingerprints
    targets = df['Target_SMILES'].apply(morgan_fingerprint)
    targets_tensor = torch.tensor(np.array(targets.tolist()).astype(np.float32))
    #print(targets_tensor.shape)

    #Concatenation for input tensor
    input_data = torch.cat((sequences_tensor, targets_tensor), dim=1)
    #print("input size: ", input_data.shape)
    
    masks_tensor = reshape_mask(masks_tensor, target_size = 118 * 7 + 1024 + 2)
    #print("mask shape: ", masks_tensor.shape)
    
    return input_data, masks_tensor

if __name__ == "__main__":
    input_data, masks_tensor = main(PATH)
    print(input_data.shape,masks_tensor.shape)
    
    # save the tensors
    # torch.save(input_data, "../data/input_data.pt")
    # torch.save(masks_tensor, "../data/input_mask.pt")