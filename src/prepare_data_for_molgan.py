import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

# Function to load dataset
def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

# Function to convert SMILES to graph representation
def smiles_to_graph(smiles):
    """
    Convert SMILES string to graph representation.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        adj = rdmolops.GetAdjacencyMatrix(mol)
        features = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        return adj, features
    else:
        return None, None

# Load the preprocessed dataset
file_path = 'data/preprocessed_dataset.csv'
data = load_data(file_path)

# Display the first few rows of the dataset
print(data.head())

# Prepare graph data
adj_list = []
features_list = []

for smiles in data['SMILES']:
    adj, features = smiles_to_graph(smiles)
    if adj is not None and features is not None:
        adj_list.append(adj)
        features_list.append(features)

# Convert lists to numpy arrays
adj_array = np.array(adj_list)
features_array = np.array(features_list)

# Save arrays for MolGAN
np.save('data/adj_array.npy', adj_array)
np.save('data/features_array.npy', features_array)

print("Graph data saved for MolGAN.")
