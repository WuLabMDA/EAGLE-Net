# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 01:27:10 2025

@author: mwaqas
"""

import argparse
import torch
from torch.utils.data import DataLoader
from Eagle_net_model import EagleNet  # Replace with your actual model file
from TCGA_loader import Subtype_CustomDataset # Replace with your actual dataset loader file
import numpy as np
from sklearn.metrics import balanced_accuracy_score 
import tqdm


def test_function(test_loader, model, device):
    """
    Function to test the AbDMIL_Base model on a test dataset.
    
    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model (nn.Module): Trained AbDMIL_Base model.
        device (str): Device for computation ('cuda' or 'cpu').
    
    Returns:
        logits_list (list): List of predicted logits for each sample.
        attention_scores_list (list): List of attention scores for each sample.
    """
    model.eval()
    logits_list = []
    attention_scores_list = []
    
    true_labels = []
    predicted_labels = []
    scores = []

    with torch.no_grad():
        for batch_idx, (data, bag_matrix, mask, bag_label, slide_id, subtype) in  tqdm.tqdm(enumerate(test_loader)):
            # Move data to device
            bag_matrix = bag_matrix.to(device)
            mask = mask.to(device)

            # Forward pass
            logits = model(bag_matrix, mask=mask, mode='Test')
            _, attention_scores = model.get_weights(bag_matrix)

            logits_list.append(logits.cpu())
            attention_scores_list.append(attention_scores.cpu())
            
            # Calculate probabilities from logits
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            true_labels.extend(bag_label.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            scores.extend(probs[:, 1].cpu().numpy())  # Assuming binary classification, get score for class 1

            print(f"Processed Batch {batch_idx + 1}/{len(test_loader)}")
        
        
    # compute balance accuracy
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    scores = np.array(scores)
    
    # Calculate metrics
    accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    return logits_list, attention_scores_list, accuracy

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the test dataset
    test_dataset = Subtype_CustomDataset(
        labels_paths=args.test_csv,
        data_paths=args.data_paths.split(','),
        packing_paths=args.packing_paths.split(','),
        mode='Test',
        device=device
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load the trained model
    params = {
       'patch_size': args.patch_size,
       'n_classes': args.n_classes,
       'embed_dim': args.embed_dim,
       'topk': args.topk,
       'device' : args.device
   }

   
   # Initialize and load model
    model = EagleNet(**params).to(device)#EAGLE_NET(**params).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully!")

    # Test the model
    print("Testing the model...")
    logits_list, attention_scores_list, accuracy = test_function(test_loader, model, device)

  
    
    # Save results
    print('Accuracy  = ', accuracy)
    print(f"Test results saved at {args.results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the AbDMIL_Base Model")

    # Data and paths
    parser.add_argument('--test_csv', type=str, default='./git_split/TCGA_lung_test_df.csv', help='Path to the test CSV file')
    parser.add_argument('--data_paths', type=str, default="./test/LUAD,./test/LUSC", help='Comma-separated paths to LUAD and LUSC data directories')
    parser.add_argument('--packing_paths', type=str, default="./packing/LUAD,./packing/LUSC", help='Comma-separated paths to LUAD and LUSC packing directories')
    parser.add_argument('--model_path', type=str, default='./best_tcga_model.pth', help='Path to the trained model')
    parser.add_argument('--results_path', type=str, default='./test_results.pth', help='Path to save the test results')

    # Model hyperparameters
    parser.add_argument('--patch_size', type=int, default=2048, help='Patch size for the model')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--topk', type=int, default=10, help='Top-k attention pooling')
    parser.add_argument('--device', type=str, default='cuda', help='Cuda GPU index')

    # Testing hyperparameters
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')

    args = parser.parse_args()
    main(args)
