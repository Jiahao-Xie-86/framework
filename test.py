import torch
import time
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataset import HamiltonianGraphDataset, custom_collate
from model import ConditionalGraphGenerator, get_resnet50_classifier
from train import evaluate_model
from utils import remove_module_prefix
import torch.nn as nn

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- Load the New Test Dataset ----
new_hamiltonian_dir = './hamiltonian_medium_mat'
new_non_hamiltonian_dir = './non_hamiltonian_medium_mat'

# Load the full dataset for the new test data
full_new_test_dataset = HamiltonianGraphDataset(new_hamiltonian_dir, new_non_hamiltonian_dir)

# ---- Function to Sample a Random Subset of the Dataset ----
def sample_dataset(dataset, sample_size, seed=42):
    np.random.seed(seed)
    all_indices = list(range(len(dataset)))
    sampled_indices = np.random.choice(all_indices, size=sample_size, replace=False)
    return Subset(dataset, sampled_indices)

def adjust_state_dict_for_parallel(state_dict, use_dataparallel):
    """
    Adjusts the state_dict keys based on whether the model uses DataParallel.
    
    Args:
        state_dict (dict): Original state dictionary of the model.
        use_dataparallel (bool): True if DataParallel is used; False otherwise.
    
    Returns:
        dict: Adjusted state dictionary with or without 'module.' prefix.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if use_dataparallel:
            # Add 'module.' prefix if it's missing
            new_key = f"module.{key}" if not key.startswith("module.") else key
        else:
            # Remove 'module.' prefix if it's present
            new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

# ---- Load the Best Model ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = ConditionalGraphGenerator().to(device)
classifier = get_resnet50_classifier().to(device)

# Determine if DataParallel is required
use_dataparallel = torch.cuda.device_count() > 1

# Wrap models in DataParallel if needed
if use_dataparallel:
    generator = torch.nn.DataParallel(generator)
    classifier = torch.nn.DataParallel(classifier)

# Load the saved best model checkpoint
checkpoint_path = 'best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

# Adjust state_dict keys
generator_state_dict = adjust_state_dict_for_parallel(checkpoint['generator_state_dict'], use_dataparallel)
classifier_state_dict = adjust_state_dict_for_parallel(checkpoint['classifier_state_dict'], use_dataparallel)

# Load adjusted state dicts
generator.load_state_dict(generator_state_dict)
classifier.load_state_dict(classifier_state_dict)

# Now the models are loaded correctly with or without DataParallel
print("Model loaded successfully.")

# ---- Define Function for Running Model on Sampled Data and Measuring Time ----
def test_model_with_timing(generator, classifier, dataloader, device):
    generator.eval()
    classifier.eval()

    start_time = time.time()  # Start timing
    accuracy, f1_score = evaluate_model(generator, classifier, dataloader, device)
    end_time = time.time()  # End timing

    elapsed_time = end_time - start_time
    print(f"Sampled Test Dataset - Accuracy: {accuracy:.2f}%, F1 Score: {f1_score:.4f}")
    print(f"Elapsed Time for Evaluation: {elapsed_time:.4f} seconds")

# ---- Define Sample Size and Create DataLoader for the Sampled Dataset ----
sample_size = 200  # Set the desired sample size here
sampled_test_dataset = sample_dataset(full_new_test_dataset, sample_size)
sampled_test_dataloader = DataLoader(
    sampled_test_dataset, batch_size=36, shuffle=False, collate_fn=lambda batch: custom_collate(batch, device)
)

# ---- Run the Model on the Sampled Dataset and Output Running Time ----
test_model_with_timing(generator, classifier, sampled_test_dataloader, device)
