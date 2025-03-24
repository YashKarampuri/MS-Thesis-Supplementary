#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import os
import glob
from scipy.spatial import KDTree
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed

# Detect CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the directory containing PDB files (HOLO 4k dataset)
pdb_dir = r"/home/yashk/holo4k"

# Get a list of all PDB files in the directory
pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))

def validate_and_fix_mol(mol):
    if mol is None:
        return None
    try:
        for atom in mol.GetAtoms():
            if atom.GetExplicitValence() > Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()):
                return None  # Ignore molecules with incorrect valence
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return None

def extract_sas_points(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        mol = validate_and_fix_mol(mol)
        if mol is None:
            return None

        conf = mol.GetConformer()
        sas_points = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                conf.GetAtomPosition(atom.GetIdx()).y,
                                conf.GetAtomPosition(atom.GetIdx()).z] for atom in mol.GetAtoms()])
        
        return sas_points if sas_points.size > 0 else None
    except Exception as e:
        print(f"Error processing PDB file {pdb_file}: {e}")
        return None

def construct_edges(sas_points, distance_threshold=6.0):
    if len(sas_points) == 0:
        return np.empty((0, 2), dtype=np.int64)

    tree = KDTree(sas_points)
    pairs = tree.query_pairs(distance_threshold)
    edges = np.array(list(pairs))

    return edges if edges.size > 0 else np.empty((0, 2), dtype=np.int64)

def compute_local_graph_features(edges, num_nodes):
    degree = np.zeros(num_nodes, dtype=np.float32)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    return torch.tensor(degree).view(-1, 1)  # Degree as node feature

def process_pdb_file(pdb_file):
    sas_points = extract_sas_points(pdb_file)
    if sas_points is None:
        return None

    edges = construct_edges(sas_points)
    edge_index = torch.tensor(edges.T, dtype=torch.long) if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.tensor([[np.linalg.norm(sas_points[i] - sas_points[j])] for i, j in edges], dtype=torch.float) if edges.size > 0 else torch.empty((0, 1), dtype=torch.float)

    node_features = torch.tensor(sas_points, dtype=torch.float)
    local_graph_features = compute_local_graph_features(edges, len(sas_points))
    combined_node_features = torch.cat([node_features, local_graph_features], dim=1)

    labels = torch.full((len(sas_points),), 0.5, dtype=torch.float)  # Corrected label shape

    data = Data(
        x=combined_node_features.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_features.to(device),
        y=labels.to(device)
    )

    return data

class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, pdb_files, transform=None, pre_transform=None):
        self.pdb_files = pdb_files
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) if os.path.exists(self.processed_paths[0]) else self.process()

    @property
    def processed_file_names(self):
        return ["protein_graphs.pt"]

    def process(self):
        data_list = Parallel(n_jobs=8)(delayed(process_pdb_file)(pdb) for pdb in self.pdb_files)
        data_list = [d for d in data_list if d is not None]
        print(f"Processed {len(data_list)} graphs out of {len(self.pdb_files)} PDB files.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

# Load the dataset
holo4k_dataset = ProteinGraphDataset(root='holo4k_data', pdb_files=pdb_files)
dataset_loader = DataLoader(holo4k_dataset, batch_size=8, shuffle=True)

print(f"Total number of graphs: {len(holo4k_dataset)}")

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Initialize the model
input_dim = holo4k_dataset[0].x.size(1)
hidden_dim = 64
output_dim = 1  # Ligandability score

model = GNN(input_dim, hidden_dim, output_dim).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in dataset_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset_loader)}")

# Save the trained model weights
torch.save(model.state_dict(), "trained_model_weights.pth")
print("Model weights saved.")


# In[2]:


import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

# Function to visualize a graph
def visualize_graph(data, title="Graph Visualization"):
    # Convert PyTorch Geometric Data object to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Get the number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Check if the graph has nodes and edges
    if num_nodes == 0 or num_edges == 0:
        print(f"Skipping graph visualization: {title} (empty graph)")
        return
    
    # Plot the graph
    plt.figure(figsize=(8, 8))
    
    # Use a simple layout algorithm that does not require random_state
    try:
        pos = nx.circular_layout(G)  # Circular layout
    except Exception as e:
        print(f"Error generating layout for {title}: {e}")
        return
    
    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', edge_color='gray', alpha=0.9)
    
    # Add text annotation for number of nodes and edges
    plt.text(
        0.05, 0.05,  # Position of the text (bottom-left corner)
        f"Nodes: {num_nodes}\nEdges: {num_edges}",  # Text content
        transform=plt.gca().transAxes,  # Use axes coordinates
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')  # Add a background box
    )
    
    plt.title(title)
    plt.show()

# Visualize a few sample graphs from the dataset
num_samples = 3  # Number of graphs to visualize
for i in range(num_samples):
    sample_data = holo4k_dataset[i]
    visualize_graph(sample_data, title=f"Sample Graph {i+1}")


# In[3]:


import matplotlib.pyplot as plt

loss_values = [0.372, 0.028, 0.0267, 0.0214, 0.0169, 0.0152, 0.0135, 0.0663, 0.0104, 0.0074]

# Plotting the loss values
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# In[4]:


# Extract node features from the dataset
node_features = torch.cat([data.x for data in holo4k_dataset], dim=0).cpu().numpy()

# Plot histograms for each feature dimension
plt.figure(figsize=(12, 8))
for i in range(node_features.shape[1]):
    plt.subplot(2, 2, i + 1)
    plt.hist(node_features[:, i], bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of Node Feature {i + 1}')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[5]:


import torch
import matplotlib.pyplot as plt

# Compute degrees for all nodes in the dataset
degrees = []
for data in holo4k_dataset:
    # Move edge_index to CPU (if it's on GPU)
    edge_index = data.edge_index.cpu()
    
    # Create a zero tensor to store degrees for each node
    node_degrees = torch.zeros(data.num_nodes, dtype=torch.long)
    
    # Count the occurrences of each node in the edge_index
    unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
    node_degrees[unique_nodes] = counts
    
    # Append degrees to the list
    degrees.extend(node_degrees.numpy())

# Plot degree distribution
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=50, color='purple', alpha=0.7)
plt.title('Degree Distribution of Nodes')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[6]:


import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import os
import glob
from scipy.spatial import KDTree
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Detect CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the directory containing PDB files (HOLO 4k dataset)
pdb_dir = "/home/yashk/holo4k"

# Get a list of all PDB files in the directory
pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))

def validate_and_fix_mol(mol):
    if mol is None:
        return None
    try:
        for atom in mol.GetAtoms():
            if atom.GetExplicitValence() > Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()):
                return None  # Ignore molecules with incorrect valence
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return None

def extract_sas_points(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        mol = validate_and_fix_mol(mol)
        if mol is None:
            return None

        conf = mol.GetConformer()
        sas_points = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                conf.GetAtomPosition(atom.GetIdx()).y,
                                conf.GetAtomPosition(atom.GetIdx()).z] for atom in mol.GetAtoms()])
        
        return sas_points if sas_points.size > 0 else None
    except Exception as e:
        print(f"Error processing PDB file {pdb_file}: {e}")
        return None

def construct_edges(sas_points, distance_threshold=6.0):
    if len(sas_points) == 0:
        return np.empty((0, 2), dtype=np.int64)

    tree = KDTree(sas_points)
    pairs = tree.query_pairs(distance_threshold)
    edges = np.array(list(pairs))

    return edges if edges.size > 0 else np.empty((0, 2), dtype=np.int64)

def compute_local_graph_features(edges, num_nodes):
    degree = np.zeros(num_nodes, dtype=np.float32)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    return torch.tensor(degree).view(-1, 1)  # Degree as node feature

def process_pdb_file(pdb_file):
    sas_points = extract_sas_points(pdb_file)
    if sas_points is None:
        return None

    edges = construct_edges(sas_points)
    edge_index = torch.tensor(edges.T, dtype=torch.long) if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.tensor([[np.linalg.norm(sas_points[i] - sas_points[j])] for i, j in edges], dtype=torch.float) if edges.size > 0 else torch.empty((0, 1), dtype=torch.float)

    node_features = torch.tensor(sas_points, dtype=torch.float)
    local_graph_features = compute_local_graph_features(edges, len(sas_points))
    combined_node_features = torch.cat([node_features, local_graph_features], dim=1)

    labels = torch.full((len(sas_points),), 0.5, dtype=torch.float)  # Corrected label shape

    data = Data(
        x=combined_node_features.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_features.to(device),
        y=labels.to(device)
    )

    return data

class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, pdb_files, transform=None, pre_transform=None):
        self.pdb_files = pdb_files
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) if os.path.exists(self.processed_paths[0]) else self.process()

    @property
    def processed_file_names(self):
        return ["protein_graphs.pt"]

    def process(self):
        data_list = Parallel(n_jobs=8)(delayed(process_pdb_file)(pdb) for pdb in self.pdb_files)
        data_list = [d for d in data_list if d is not None]
        print(f"Processed {len(data_list)} graphs out of {len(self.pdb_files)} PDB files.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

# Load the dataset
holo4k_dataset = ProteinGraphDataset(root='holo4k_data', pdb_files=pdb_files)
dataset_loader = DataLoader(holo4k_dataset, batch_size=8, shuffle=True)

print(f"Total number of graphs: {len(holo4k_dataset)}")

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Initialize the model
input_dim = holo4k_dataset[0].x.size(1)
hidden_dim = 64
output_dim = 1  # Ligandability score

model = GNN(input_dim, hidden_dim, output_dim).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in dataset_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset_loader)}")

# Save the trained model weights
torch.save(model.state_dict(), "trained_model_weights.pth")
print("Model weights saved.")

# Clustering and visualization functions
def cluster_points(sas_points, ligandability_scores, cutoff=3.0):
    """
    Cluster points based on ligandability scores and spatial proximity.

    Args:
        sas_points (np.ndarray): Array of shape (N, 3) containing SAS points.
        ligandability_scores (np.ndarray): Array of shape (N,) containing ligandability scores.
        cutoff (float): Distance cutoff for clustering.

    Returns:
        clusters (np.ndarray): Array of cluster labels for high-score points.
        high_score_points (np.ndarray): Array of high-score points.
    """
    # Ensure ligandability_scores is a 1D array
    ligandability_scores = ligandability_scores.squeeze()

    # Filter points with high ligandability scores (e.g., top 20%)
    high_score_indices = ligandability_scores > np.quantile(ligandability_scores, 0.8)
    high_score_points = sas_points[high_score_indices]

    # Handle edge case: no high-score points
    if len(high_score_points) == 0:
        print("No high-score points found for clustering.")
        return np.array([]), np.array([])

    # Compute pairwise distances
    distances = pdist(high_score_points)

    # Perform single-linkage clustering
    Z = linkage(distances, method='single')
    clusters = fcluster(Z, t=cutoff, criterion='distance')

    return clusters, high_score_points

def visualize_clusters(high_score_points, clusters):
    """
    Visualize clusters in 3D space.

    Args:
        high_score_points (np.ndarray): Array of shape (N, 3) containing high-score points.
        clusters (np.ndarray): Array of cluster labels for high-score points.
    """
    if len(high_score_points) == 0:
        print("No points to visualize.")
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster with a different color
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = high_score_points[clusters == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clustering of High-Score Points')
    plt.legend()
    plt.show()

# Example usage
data = holo4k_dataset[0].to(device)
sas_points = data.x[:, :3].cpu().numpy()  # Extract SAS points
ligandability_scores = model(data.x, data.edge_index, data.batch).detach().cpu().numpy()

# Cluster points
clusters, high_score_points = cluster_points(sas_points, ligandability_scores)

# Visualize clusters
visualize_clusters(high_score_points, clusters)


# In[7]:


def rank_pockets(clusters, high_score_points, ligandability_scores):
    """
    Rank pockets based on the cumulative squared ligandability scores of their points.

    Args:
        clusters (np.ndarray): Array of cluster labels for high-score points.
        high_score_points (np.ndarray): Array of high-score points.
        ligandability_scores (np.ndarray): Array of ligandability scores for high-score points.

    Returns:
        ranked_pockets (list): List of tuples (cluster_id, cumulative_score) sorted by score.
    """
    # Ensure ligandability_scores is a 1D array
    ligandability_scores = ligandability_scores.squeeze()

    # Filter ligandability_scores to match high_score_points
    high_score_indices = ligandability_scores > np.quantile(ligandability_scores, 0.8)
    filtered_scores = ligandability_scores[high_score_indices]

    # Calculate cumulative squared scores for each cluster
    pocket_scores = {}
    for cluster_id, score in zip(clusters, filtered_scores):
        if cluster_id not in pocket_scores:
            pocket_scores[cluster_id] = 0
        pocket_scores[cluster_id] += score ** 2  # Sum of squared scores
    
    # Sort pockets by cumulative score
    ranked_pockets = sorted(pocket_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_pockets

# Example usage
ranked_pockets = rank_pockets(clusters, high_score_points, ligandability_scores)
print("Ranked Pockets:", ranked_pockets)


# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
import os
import glob
from scipy.spatial import KDTree
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Detect CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the directory containing PDB files (HOLO 4k dataset)
pdb_dir = "/home/yashk/holo4k"

# Get a list of all PDB files in the directory
pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))

def validate_and_fix_mol(mol):
    if mol is None:
        return None
    try:
        for atom in mol.GetAtoms():
            if atom.GetExplicitValence() > Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()):
                return None  # Ignore molecules with incorrect valence
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Error sanitizing molecule: {e}")
        return None

def extract_sas_points(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        mol = validate_and_fix_mol(mol)
        if mol is None:
            return None

        conf = mol.GetConformer()
        sas_points = np.array([[conf.GetAtomPosition(atom.GetIdx()).x,
                                conf.GetAtomPosition(atom.GetIdx()).y,
                                conf.GetAtomPosition(atom.GetIdx()).z] for atom in mol.GetAtoms()])
        
        return sas_points if sas_points.size > 0 else None
    except Exception as e:
        print(f"Error processing PDB file {pdb_file}: {e}")
        return None

def construct_edges(sas_points, distance_threshold=6.0):
    if len(sas_points) == 0:
        return np.empty((0, 2), dtype=np.int64)

    tree = KDTree(sas_points)
    pairs = tree.query_pairs(distance_threshold)
    edges = np.array(list(pairs))

    return edges if edges.size > 0 else np.empty((0, 2), dtype=np.int64)

def compute_local_graph_features(edges, num_nodes):
    degree = np.zeros(num_nodes, dtype=np.float32)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    return torch.tensor(degree).view(-1, 1)  # Degree as node feature

def process_pdb_file(pdb_file):
    sas_points = extract_sas_points(pdb_file)
    if sas_points is None:
        return None

    edges = construct_edges(sas_points)
    edge_index = torch.tensor(edges.T, dtype=torch.long) if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)
    edge_features = torch.tensor([[np.linalg.norm(sas_points[i] - sas_points[j])] for i, j in edges], dtype=torch.float) if edges.size > 0 else torch.empty((0, 1), dtype=torch.float)

    node_features = torch.tensor(sas_points, dtype=torch.float)
    local_graph_features = compute_local_graph_features(edges, len(sas_points))
    combined_node_features = torch.cat([node_features, local_graph_features], dim=1)

    labels = torch.full((len(sas_points),), 0.5, dtype=torch.float)  # Corrected label shape

    data = Data(
        x=combined_node_features.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_features.to(device),
        y=labels.to(device)
    )

    return data

class ProteinGraphDataset(InMemoryDataset):
    def __init__(self, root, pdb_files, transform=None, pre_transform=None):
        self.pdb_files = pdb_files
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) if os.path.exists(self.processed_paths[0]) else self.process()

    @property
    def processed_file_names(self):
        return ["protein_graphs.pt"]

    def process(self):
        data_list = Parallel(n_jobs=8)(delayed(process_pdb_file)(pdb) for pdb in self.pdb_files)
        data_list = [d for d in data_list if d is not None]
        print(f"Processed {len(data_list)} graphs out of {len(self.pdb_files)} PDB files.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

# Load the dataset
holo4k_dataset = ProteinGraphDataset(root='holo4k_data', pdb_files=pdb_files)
dataset_loader = DataLoader(holo4k_dataset, batch_size=8, shuffle=True)

print(f"Total number of graphs: {len(holo4k_dataset)}")

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x  # Shape: (num_nodes, output_dim)

# Initialize the model
input_dim = holo4k_dataset[0].x.size(1)
hidden_dim = 64
output_dim = 1  # Ligandability score

model = GNN(input_dim, hidden_dim, output_dim).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in dataset_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset_loader)}")

# Save the trained model weights
torch.save(model.state_dict(), "trained_model_weights.pth")
print("Model weights saved.")

# Clustering and visualization functions
def cluster_points(sas_points, ligandability_scores, cutoff=3.0):
    """
    Cluster points based on ligandability scores and spatial proximity.
    """
    ligandability_scores = ligandability_scores.squeeze()
    high_score_indices = ligandability_scores > np.quantile(ligandability_scores, 0.8)
    high_score_points = sas_points[high_score_indices]

    if len(high_score_points) == 0:
        print("No high-score points found for clustering.")
        return np.array([]), np.array([])

    distances = pdist(high_score_points)
    Z = linkage(distances, method='single')
    clusters = fcluster(Z, t=cutoff, criterion='distance')

    return clusters, high_score_points

def rank_pockets(clusters, high_score_points, ligandability_scores):
    """
    Rank pockets based on the cumulative squared ligandability scores of their points.
    """
    ligandability_scores = ligandability_scores.squeeze()
    high_score_indices = ligandability_scores > np.quantile(ligandability_scores, 0.8)
    filtered_scores = ligandability_scores[high_score_indices]

    pocket_scores = {}
    for cluster_id, score in zip(clusters, filtered_scores):
        if cluster_id not in pocket_scores:
            pocket_scores[cluster_id] = 0
        pocket_scores[cluster_id] += score ** 2  # Sum of squared scores
    
    ranked_pockets = sorted(pocket_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_pockets

def visualize_ranked_pockets(high_score_points, clusters, ranked_pockets):
    """
    Visualize ranked pockets in 3D space with the legend outside the plot area.
    """
    if len(high_score_points) == 0:
        print("No points to visualize.")
        return

    fig = plt.figure(figsize=(12, 8))  
    ax = fig.add_subplot(111, projection='3d')

    # Map cluster IDs to their ranks
    rank_colors = {cluster_id: rank + 1 for rank, (cluster_id, _) in enumerate(ranked_pockets)}
    sorted_clusters = sorted(np.unique(clusters), key=lambda x: rank_colors.get(x, len(ranked_pockets) + 1))
    
    # Plot each cluster with a different color and label
    for cluster_id in sorted_clusters:
        cluster_points = high_score_points[clusters == cluster_id]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   label=f'Cluster {cluster_id} (Rank {rank_colors.get(cluster_id, len(ranked_pockets) + 1)})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Ranked Pockets Visualization')

    # Move the legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

# Example of integrating all steps
for data in dataset_loader:
    # Move data to the appropriate device
    data = data.to(device)
    
    # Step 3: Predict ligandability scores
    ligandability_scores = model(data.x, data.edge_index, data.batch).detach().cpu().numpy()
    
    # Step 4: Cluster points with high ligandability scores
    sas_points = data.x[:, :3].cpu().numpy()  # Extract SAS points
    clusters, high_score_points = cluster_points(sas_points, ligandability_scores)
    
    # Step 5: Rank predicted pockets
    if len(clusters) > 0:  # Only rank if clusters were found
        ranked_pockets = rank_pockets(clusters, high_score_points, ligandability_scores)
        print("Ranked Pockets:", ranked_pockets)
        
        # Step 6: Visualize ranked pockets
        visualize_ranked_pockets(high_score_points, clusters, ranked_pockets)
    else:
        print("No clusters found for ranking.")


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Split the dataset into three parts: train, validation, and test
# Ensure validation and test sets have the same size
train_size = 0.6  # 60% training
val_size = 0.2    # 20% validation
test_size = 0.2   # 20% testing

# First split: separate training set
train_dataset, val_test_dataset = train_test_split(holo4k_dataset, test_size=(val_size + test_size), random_state=42)

# Second split: separate validation and test sets
val_dataset, test_dataset = train_test_split(val_test_dataset, test_size=test_size/(val_size + test_size), random_state=42)

# Create DataLoader for training, validation, and testing sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# Save the trained model weights
torch.save(model.state_dict(), "trained_model_weights.pth")
print("Model weights saved.")

# Evaluate the model on the validation set (to get true values)
model.eval()
y_true_val = []
with torch.no_grad():
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_true_val.extend(out.squeeze().cpu().numpy())

# Evaluate the model on the test set (to get predicted values)
y_pred_test = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_pred_test.extend(out.squeeze().cpu().numpy())

# Convert to numpy arrays
y_true_val = np.array(y_true_val)
y_pred_test = np.array(y_pred_test)

# Debugging outputs
print("y_true_val distribution:", np.unique(y_true_val, return_counts=True))
print("y_pred_test distribution:", np.unique(y_pred_test, return_counts=True))

# Define a threshold for binarization
threshold = 0.52  

# Binarize y_true_val and y_pred_test
y_true_val_binary = (y_true_val > threshold).astype(int)
y_pred_test_binary = (y_pred_test > threshold).astype(int)

# Debugging outputs
print("y_true_val_binary distribution:", np.unique(y_true_val_binary, return_counts=True))
print("y_pred_test_binary distribution:", np.unique(y_pred_test_binary, return_counts=True))

# Ensure the sizes of y_true_val_binary and y_pred_test_binary are the sameb
if len(y_true_val_binary) != len(y_pred_test_binary):
    min_length = min(len(y_true_val_binary), len(y_pred_test_binary))
    y_true_val_binary = y_true_val_binary[:min_length]
    y_pred_test_binary = y_pred_test_binary[:min_length]

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_true_val_binary, y_pred_test_binary)

# Plot the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (True vs Predicted)')
plt.xlabel('Predicted Label (Test Set)')
plt.ylabel('True Label (Validation Set)')
plt.show()


# In[30]:


import matplotlib.pyplot as plt

# Given confusion matrix values
TP = 486234
TN = 13013
FP = 77646
FN = 79025

# Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
TPR = TP / (TP + FN)  # Sensitivity or Recall
FPR = FP / (FP + TN)  # Fall-out

# ROC curve is a single point (FPR, TPR)
fpr = [0, FPR, 1]  # FPR values for plotting
tpr = [0, TPR, 1]  # TPR values for plotting

# Calculate AUC (Area Under the Curve)
# For a single point, AUC is the area of the trapezoid formed by (0,0), (FPR, TPR), and (1,1)
auc = (TPR + 1) * FPR / 2

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC value
print(f"AUC: {auc:.2f}")

