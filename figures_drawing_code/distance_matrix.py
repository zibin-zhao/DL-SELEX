'''Drwaing distance matrix for figure 2 by Zibin Zhao with CS as an exmaple'''

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# Define the function to convert SMILES to a distance matrix
def smiles_to_distance_matrix(smiles):
    """Convert a SMILES string to a distance matrix."""
    # Convert the SMILES string to a molecule.
    molecule = Chem.MolFromSmiles(smiles)
    
    # Add Hydrogens to the molecule
    molecule = Chem.AddHs(molecule)
    
    # Generate a 3D conformation for the molecule
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    
    # Get the distance matrix
    distance_matrix = Chem.rdmolops.Get3DDistanceMatrix(molecule)
    
    # Normalize the distance matrix.
    distance_matrix = distance_matrix / distance_matrix.max()
    
    return distance_matrix

# Create a custom colormap using the color #c42238
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ffffff', '#d98380'])

# Hydrocortisone SMILES string
smiles_hydrocortisone = "O=C1CC2C3C(C=C2C1C4CCC(C(C4)O)C(=O)C3)C5CC(CCC5O)C(=O)CO"

# Get the distance matrix for hydrocortisone
distance_matrix = smiles_to_distance_matrix(smiles_hydrocortisone)

# Plot the distance matrix as a heatmap using the custom colormap
plt.figure(figsize=(8, 6))
plt.imshow(distance_matrix, cmap=custom_cmap, interpolation='none')
plt.colorbar(label='Normalized Distance')
plt.title("Normalized Distance Matrix for Hydrocortisone")
plt.show()
