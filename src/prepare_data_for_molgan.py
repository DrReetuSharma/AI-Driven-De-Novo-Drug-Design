"""
Convert SMILES to graph and prepare adjacency matrix (in the form of array) using rdmolops and feature matrix 
(atomic numbers, molwt, logp, tsra, num_rotational bond, hydrogenbond donars and acceptors)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors

# Function to load dataset
def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)

# Function to convert SMILES to graph representation
def smiles_to_graph(smiles):
    """
    Convert SMILES string to graph representation with additional features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        adj = rdmolops.GetAdjacencyMatrix(mol)
        
        # Extract atomic numbers
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
        # Calculate additional features
        molwt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_h_donors = Descriptors.NumHDonors(mol)
        num_h_acceptors = Descriptors.NumHAcceptors(mol)
        
        # Combine features into one array
        additional_features = [molwt, logp, tpsa, num_rotatable_bonds, num_h_donors, num_h_acceptors]
        features = atomic_numbers + additional_features
        
        return adj, np.array(features)
    return None, None

# Function to pad lists to the same length
def pad_list(lst, length, pad_value=0):
    lst = list(lst)
    return lst + [pad_value] * (length - len(lst))

# Load the preprocessed dataset
# file_path = 'c:\\Users\\reetu\\Desktop\\predataset.csv'
file_path = 'data/predataset.csv'
data = load_data(file_path)

# Display the first few rows of the dataset
# print(data.head())

# Prepare graph data
adj_list = []
features_list = []

for smiles in data['SMILES']:
    adj, features = smiles_to_graph(smiles)
    if adj is not None and features is not None:
        adj_list.append(adj)
        features_list.append(features)

# Determine the maximum length of features
max_length = max(len(f) for f in features_list)

# Pad features lists to the same length
features_list_padded = [pad_list(f, max_length) for f in features_list]

# Convert lists to numpy arrays
adj_array = np.array(adj_list, dtype=object)
features_array = np.array(features_list_padded)

#print('adjacency_matrix\n', adj_array)
print('\n\nfeatures_array)\n', features_array)

# Save arrays for MolGAN
# np.save('c:\\Users\\reetu\\Desktop\\adj_array.npy', adj_array)
# np.save('c:\\Users\\reetu\\Desktop\\features_array.npy', features_array)

np.save('data/adj_array.npy', adj_array)
np.save('data/features_array.npy', features_array)
print("Graph data saved for MolGAN.")
