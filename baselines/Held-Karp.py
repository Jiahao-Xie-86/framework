import os
import time
import numpy as np
import random
from itertools import combinations
from torch.utils.data import DataLoader, Subset
from dataset import HamiltonianGraphDataset, custom_collate
import torch

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Test Dataset
hamiltonian_test_dir = './hamiltonian_small_mat'
non_hamiltonian_test_dir = './non_hamiltonian_small_mat'
test_dataset = HamiltonianGraphDataset(hamiltonian_test_dir, non_hamiltonian_test_dir)
test_dataloader = DataLoader(test_dataset, batch_size=500, shuffle=False, collate_fn=lambda batch: custom_collate(batch, device))

def held_karp_detection(adj_matrix):
    """
    Detects whether a Hamiltonian cycle exists using a variation
    of the Held-Karp dynamic programming algorithm.
    
    Parameters:
      adj_matrix (numpy.ndarray): An n x n adjacency matrix representing
                                  the graph, where a positive value indicates
                                  the presence of an edge.
    
    Returns:
      bool: True if a Hamiltonian cycle exists, False otherwise.
    """
    n = adj_matrix.shape[0]
    dp = {}
    # Base step: Start at node 0 and record reachable nodes in one step.
    for k in range(1, n):
        if adj_matrix[0][k] > 0:
            dp[(1 << k, k)] = True

    # Build up solutions for subsets of increasing size (covering nodes 1 ... n-1)
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = sum(1 << k for k in subset)
            for k in subset:
                # Remove node k from the current subset to obtain the previous state.
                prev_bits = bits & ~(1 << k)
                # Check if there is any node m in the subset (other than k) that
                # can reach k and has a valid path from node 0.
                for m in subset:
                    if m == k:
                        continue
                    if dp.get((prev_bits, m), False) and adj_matrix[m][k] > 0:
                        dp[(bits, k)] = True
                        break  # Early exit if a valid path is found for (bits, k)

    # Finally, check if there is a way to return to the start node (0)
    # from any path covering all nodes.
    full_mask = (1 << n) - 2  # Bitmask covering all nodes 1 to n-1.
    for k in range(1, n):
        if dp.get((full_mask, k), False) and adj_matrix[k][0] > 0:
            return True  # Hamiltonian cycle detected
    return False  # No Hamiltonian cycle exists



def test_held_karp_on_dataset(dataloader, device):
    total_runtime = 0
    total_graphs = 0

    for i, (graph_layouts, adj_matrices, labels) in enumerate(dataloader):
        # Using just the first graph from the batch for demonstration.
        adj_matrix = adj_matrices[0].cpu().numpy()
        start_time = time.time()
        exists = held_karp_detection(adj_matrix)
        end_time = time.time()
        total_runtime += (end_time - start_time)
        total_graphs += 1
        # Optionally, print if a Hamiltonian cycle is found.
        print(f"Graph {i}: Hamiltonian cycle detected? {exists}")

    avg_runtime = total_runtime / total_graphs
    print(f"Average time per graph using Held-Karp for Hamiltonian Cycle Detection: {avg_runtime:.6f} seconds")
    return avg_runtime


# Set the desired sample sizes and seeds for testing
sample_sizes = [500]  # Different dataset sizes to test
seeds = [7, 17, 23, 42, 123]            # Random seeds to ensure varied sampling

def get_random_subset(dataset, sample_size, seed):
    """Randomly sample a subset of the dataset with a given size and seed."""
    random.seed(seed)
    indices = random.sample(range(len(dataset)), sample_size)
    return Subset(dataset, indices)

# Run the tests on each randomly sampled subset
for sample_size, seed in zip(sample_sizes, seeds):
    print(f"\nTesting with sample size {sample_size} and seed {seed}")
    
    # Create a random subset of the test dataset
    subset = get_random_subset(test_dataset, sample_size, seed)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=lambda batch: custom_collate(batch, device))

    
    print("Testing on subset with Held-Karp Algorithm:")
    avg_time_held_karp = test_held_karp_on_dataset(dataloader, device)
    
    # Summary of results
    print(f"\nSummary for sample size {sample_size} with seed {seed}:")
    print(f"Held-Karp Algorithm: {avg_time_held_karp:.6f} seconds")
