import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load the dataset.csv file
df = pd.read_csv('dataset.csv')

# Initialize lists to store molecular properties
molecular_properties = []

# Iterate over each SMILES string in the dataset
for smiles in df['SMILES']:
    # Convert SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Calculate molecular properties
    mol_weight = Descriptors.MolWt(mol) # Molecular weight
    num_atoms = mol.GetNumAtoms() # Number of atoms
    num_heavy_atoms = Descriptors.HeavyAtomCount(mol) # Number of heavy atoms
    num_hetero_atoms = Descriptors.NumHeteroatoms(mol) # Number of hetero atoms
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol) # Number of rotatable bonds
    num_rings = Descriptors.RingCount(mol) # Number of rings
    
    # Append calculated properties to the list
    molecular_properties.append([mol_weight, num_atoms, num_heavy_atoms, num_hetero_atoms, num_rotatable_bonds, num_rings])

# Create a DataFrame to store molecular properties
properties_df = pd.DataFrame(molecular_properties, columns=['Molecular_Weight', 'Num_Atoms', 'Num_Heavy_Atoms', 'Num_Hetero_Atoms', 'Num_Rotatable_Bonds', 'Num_Rings'])

# Concatenate the original DataFrame with the calculated properties DataFrame
result_df = pd.concat([df, properties_df], axis=1)

# Save the result to a new CSV file
result_df.to_csv('molecular_properties.csv', index=False)
