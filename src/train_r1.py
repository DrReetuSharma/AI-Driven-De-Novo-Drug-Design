import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rdkit import Chem
from src.models.molgan import Generator, Discriminator
from src.utils.data_processing import load_data, graph_to_smiles
from src.utils.chem_utils import calculate_reward

# Load the preprocessed dataset
file_path = 'data/preprocessed_dataset.csv'
data = load_data(file_path)

# Load the trained Generator and Discriminator models
generator = Generator()
discriminator = Discriminator()
generator.load_state_dict(torch.load('models/generator_final.pth'))
discriminator.load_state_dict(torch.load('models/discriminator_final.pth'))

# Set the models to evaluation mode
generator.eval()
discriminator.eval()

# Define the RL optimizer
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)

# Reinforcement Learning Training Loop
num_epochs = 1000
generated_smiles_list = []  # To store generated molecules
for epoch in range(num_epochs):
    optimizer_G.zero_grad()
    
    # Generate new molecules
    z = torch.randn(data.shape[0], 128)
    generated_data = generator(z)
    
    # Convert generated data to SMILES
    generated_smiles = []
    rewards = []
    for data in generated_data:
        adj_matrix = np.round(data[:64]).astype(int)  # Adjust based on actual size
        features = data[64:].astype(int)
        smiles = graph_to_smiles(adj_matrix, features)
        if smiles:
            generated_smiles.append(smiles)
            rewards.append(calculate_reward(smiles))
    
    # Append to the list for saving later
    generated_smiles_list.extend(generated_smiles)
    
    # Calculate the policy gradient loss
    rewards = torch.tensor(rewards, dtype=torch.float32)
    loss = -torch.mean(rewards)  # Policy gradient loss (negative reward)
    
    # Backpropagation and optimization
    loss.backward()
    optimizer_G.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the RL-trained Generator model
torch.save(generator.state_dict(), 'models/generator_rl_final.pth')

# Save the generated molecules to a CSV file
generated_molecules_df = pd.DataFrame(generated_smiles_list, columns=['SMILES'])
generated_molecules_df.to_csv('data/generated_molecules.csv', index=False)

print("Reinforcement Learning training completed and files saved.")
