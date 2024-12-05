'''3D preprocessing for target distance, adjacency matrix and 
    converting into torch tensor by Zibin Zhao'''


import pandas as pd
import numpy as np
import ast
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdmolops

MAX_SIZE = 100


def smiles_to_adjacency_matrix(smiles):
    """Convert a SMILES string to an adjacency matrix."""
    # Convert the SMILES string to a molecule.
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    
    # Get the adjacency matrix.
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    
    return adjacency_matrix


def smiles_to_distance_matrix(smiles):
    """Convert a SMILES string to a distance matrix."""
    # Convert the SMILES string to a molecule.
    molecule = Chem.MolFromSmiles(smiles)
    
    # Add Hydrogens to the molecule
    molecule = Chem.AddHs(molecule)
    
    # Generate a 3D conformation for the molecule
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    
    # Get the distance matrix.
    distance_matrix = Chem.rdmolops.Get3DDistanceMatrix(molecule)
    
    # Normalize the distance matrix.
    distance_matrix = distance_matrix / distance_matrix.max()
    
    return distance_matrix



# Pad and mask the target matrix to 85
def pad_and_mask_matrix(matrix, max_size=MAX_SIZE):
    """Pads the matrix with 0s and adds a mask where the original matrix is 1 and the padded area is 0."""
    
    # The mask should be 1 for the area of the original matrix.
    mask = np.ones_like(matrix, dtype=np.uint8)
    
    # Pad the original matrix with 0s to the max_size.
    padded_matrix = np.pad(matrix, ((0, max_size - matrix.shape[0]), (0, max_size - matrix.shape[1])), 'constant')
    
    # Pad the mask with 0s to the max_size.
    padded_mask = np.pad(mask, ((0, max_size - mask.shape[0]), (0, max_size - mask.shape[1])), 'constant')
    
    return padded_matrix, padded_mask


# Pad and mask sequence matrix to 85
def dot_bracket_to_matrix(dot_bracket, max_length=MAX_SIZE):
    length = len(dot_bracket)
    matrix = np.zeros((max_length, max_length))
    mask = np.zeros((max_length, max_length))
    
    stack = []
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                matrix[start][i] = 1
                matrix[i][start] = 1
                mask[start][i] = 1
                mask[i][start] = 1
                
    return matrix, mask


def main():
    df = pd.read_csv('../data/refined_primer/train_dataset_refined_primer.csv')

    # Get the adjacency and distance matrices for each molecule.
    df['AdjacencyMatrix'] = df['Target_SMILES'].apply(smiles_to_adjacency_matrix)
    df['DistanceMatrix'] = df['Target_SMILES'].apply(smiles_to_distance_matrix)
    
    # Pad and mask the adjacency and distance matrices for targets.
    df['AdjacencyMatrix'], df['AdjacencyMask'] = zip(*df['AdjacencyMatrix'].apply(pad_and_mask_matrix))
    df['DistanceMatrix'], df['DistanceMask'] = zip(*df['DistanceMatrix'].apply(pad_and_mask_matrix))
    
    # Pad and mask the adjacency and distance matrices for sequences.
    df['seq_matrix'], df['seq_mask'] = zip(*df['modified dppn'].map(dot_bracket_to_matrix))
    
    # Stack the matrices and masks.
    num_samples = len(df)
    combined_matrix = np.zeros((num_samples, 3, MAX_SIZE, MAX_SIZE))
    combined_mask = np.zeros((num_samples, 3, MAX_SIZE, MAX_SIZE))
    
    for i in range(num_samples):
        
        combined_matrix[i, 0] = df['AdjacencyMatrix'].iloc[i]
        combined_matrix[i, 1] = df['seq_matrix'].iloc[i]
        combined_matrix[i, 2] = df['DistanceMatrix'].iloc[i]
        
        combined_mask[i, 0] = df['AdjacencyMask'].iloc[i]
        combined_mask[i, 1] = df['seq_mask'].iloc[i]
        combined_mask[i, 2] = df['DistanceMask'].iloc[i]
    
    # Convert the matrices and masks to PyTorch tensors.
    combined_matrix = torch.from_numpy(combined_matrix)
    combined_mask = torch.from_numpy(combined_mask)
    
    print(df['AdjacencyMatrix'].head())
    print(df['seq_matrix'].head())
    print(df['DistanceMatrix'].head())
    
    print(combined_matrix.shape)  # Should be torch.Size([195, 3, 85, 85])
    print(combined_mask.shape)   # Should be torch.Size([195, 3, 85, 85])
    
    torch.save(combined_matrix, '../data/refined_primer/primer_3d_input.pt')
    torch.save(combined_mask, '../data/refined_primer/primer_3d_mask.pt')

if __name__=="__main__":
    main()