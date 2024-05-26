import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.molgan import Generator, Discriminator
from src.utils.data_processing import load_data, smiles_to_graph

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

# Convert to torch tensors
adj_tensor = torch.tensor(adj_array, dtype=torch.float32)
features_tensor = torch.tensor(features_array, dtype=torch.float32)

# Training MolGAN
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

num_epochs = 1000
for epoch in range(num_epochs):
    # Train Discriminator
    optimizer_D.zero_grad()
    real_labels = torch.ones(features_tensor.size(0), 1)
    fake_labels = torch.zeros(features_tensor.size(0), 1)
    
    outputs = discriminator(features_tensor)
    d_loss_real = criterion(outputs, real_labels)
    
    z = torch.randn(features_tensor.size(0), 128)
    fake_data = generator(z)
    outputs = discriminator(fake_data.detach())
    d_loss_fake = criterion(outputs, fake_labels)
    
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()
    
    # Train Generator
    optimizer_G.zero_grad()
    outputs = discriminator(fake_data)
    g_loss = criterion(outputs, real_labels)
    g_loss.backward()
    optimizer_G.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

print("Training completed.")
