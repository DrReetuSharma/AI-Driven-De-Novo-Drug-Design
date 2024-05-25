import numpy as np

# Load adj_array.npy and features_array.npy
adj_array = np.load('data/adj_array.npy')
features_array = np.load('data/features_array.npy')

# Inspect shapes
print("adj_array shape:", adj_array.shape)
print("features_array shape:", features_array.shape)

# Print sample data
print("\nSample adj_array:")
print(adj_array[0])  # Print the first graph's adjacency matrix
print("\nSample features_array:")
print(features_array[0])  # Print the atomic features of the first graph
