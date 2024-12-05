'''Drawing adjacency matrix for Figure 2 of cortisol as an example by Zibin Zhao'''

import matplotlib.pyplot as plt
import networkx as nx

# Define the color requested
custom_color = '#c42238'

# Create a graph using networkx for hydrocortisone
G = nx.Graph()

# Define the key functional groups of hydrocortisone
atoms = ["C3 Carbonyl", "C11 Hydroxyl", "C17 Hydroxyl", "C21 Hydroxyl", "C4-C5 Alkene"]

# Define connections (bonds) between the functional groups
connections = [
    ("C3 Carbonyl", "C4-C5 Alkene"),
    ("C4-C5 Alkene", "C11 Hydroxyl"),
    ("C11 Hydroxyl", "C17 Hydroxyl"),
    ("C17 Hydroxyl", "C21 Hydroxyl"),
]

# Add nodes for each key functional group or atom
for atom in atoms:
    G.add_node(atom)

# Add edges (bonds) based on the connections
for atom1, atom2 in connections:
    G.add_edge(atom1, atom2)

# Define a manual square layout
pos_square = {
    "C3 Carbonyl": (0, 1),
    "C4-C5 Alkene": (1, 1),
    "C11 Hydroxyl": (1, 0),
    "C17 Hydroxyl": (2, 0),
    "C21 Hydroxyl": (2, 1)
}

# Plot the graph using the custom color
plt.figure(figsize=(8, 6))
nx.draw(G, pos_square, with_labels=True, node_color=custom_color, node_size=1500, font_size=10, font_weight='bold', edge_color='gray')
nx.draw_networkx_edge_labels(G, pos_square, edge_labels={(atom1, atom2): 'bond' for atom1, atom2 in connections})

plt.title("Square Layout of Simplified Hydrocortisone with Custom Color")
plt.show()

from rdkit import Chem
import numpy as np

# Define the function to convert SMILES to an adjacency matrix
def smiles_to_adjacency_matrix(smiles):
    """Convert a SMILES string to an adjacency matrix."""
    # Convert the SMILES string to a molecule.
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    
    # Get the adjacency matrix.
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    
    return adjacency_matrix

# Hydrocortisone SMILES string
smiles_hydrocortisone = "O=C1CC2C3C(C=C2C1C4CCC(C(C4)O)C(=O)C3)C5CC(CCC5O)C(=O)CO"

# Get the adjacency matrix for hydrocortisone
adj_matrix = smiles_to_adjacency_matrix(smiles_hydrocortisone)

# Print the adjacency matrix with 2 decimal points
print("Adjacency Matrix for Hydrocortisone (formatted to 2 decimal places):")
print(np.array2string(np.array(adj_matrix, dtype=float), formatter={'float_kind':lambda x: f"{x:.0f}"}))


# Plot the adjacency matrix with a white background and custom colors for the 1's
import matplotlib.pyplot as plt
import numpy as np

# Example adjacency matrix similar to the user's image
adj_matrix_example = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0]
])
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(np.ones_like(adj_matrix_example), cmap='gray_r')  # White background

# Add annotations for the matrix values, using custom colors for '1' and white for '0'
for (i, j), val in np.ndenumerate(adj_matrix_example):
    if val == 1:
        ax.text(j, i, f'{val}', ha='center', va='center', color='#d98380', fontsize=14, fontweight='bold')
    else:
        ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=14, fontweight='bold')

# Remove x and y ticks
ax.set_xticks([])
ax.set_yticks([])
plt.show()