

import argparse
import torch
from torch.utils.data import DataLoader
from Eagle_net_model import EagleNet
from weight_loss1 import CrossEntropyLoss as CE
from TCGA_loader import Subtype_CustomDataset
import tqdm
import torch.nn as nn
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim


# import torch.multiprocessing as mp
seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)

def initialize_weights(model):
    """
    Apply Xavier initialization to all nn.Linear layers in the model.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:  # Initialize bias to zero
                nn.init.zeros_(m.bias)


def train_function(epoch, train_loader, model, criterion, weight_criterion, optimizer, device,
                   lambda_value = 2,
                   beta_value = 1
                   ):
    """
    Training function for one epoch.
    Computes `loss1` bag loss +  'loss2' (L2 + CrossEntropy) and `loss3` (L3 weighted loss).
    """
    model.train()
    train_loss_sum = 0.0
    train_loss1=0.0
    train_loss2=0.0
    optimizer.zero_grad()
    for batch_idx, (data, bag_matrix, mask, bag_label, slide_id, subtype) in tqdm.tqdm(enumerate(train_loader)):
        # Move data to the device
        bag_label = bag_label.type(torch.LongTensor)
        bag_label = Variable(bag_label.to(device),requires_grad =False)
        
        bag_matrix = Variable(bag_matrix.to(device),requires_grad =False)
        # perform forward pass for first bag
        logits, background_alpha, neighbor_logits, neighbor_alpha = model(bag_matrix,mask)
        
        # Loss 1: CrossEntropyLoss + L2 (L3  background_alpha)
        #L1
        loss1  =  criterion(logits, bag_label)
        
        #L2
        instance_labels1 = bag_label*torch.squeeze(torch.ones(neighbor_alpha.shape[1],1)).type(torch.LongTensor).to(device)
        instance_labels1 = Variable(instance_labels1.to(device),requires_grad =False)
        loss2 = weight_criterion(neighbor_logits, instance_labels1, weights=neighbor_alpha)
      
        # total loss = L1 + L2 + L3
        loss1 = loss1  + (loss2 *lambda_value) +  (beta_value* background_alpha)
        train_loss_sum  = train_loss_sum + loss1
        
        
        # background pass with gradient accumulation 
        loss1.backward()
        
        if (batch_idx + 1) % 10 == 0: 
           optimizer.step()
           optimizer.zero_grad()
           torch.cuda.empty_cache() 
           
        if (batch_idx == len(train_loader)-1):
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache() 
            
    train_loss_sum /= len(train_loader)
    print('Epoch: {}, Accumulated Loss: {:.4f}'.format(epoch, train_loss_sum.data))
    return train_loss_sum

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Load training dataset and DataLoader
    train_dataset = Subtype_CustomDataset(
        labels_paths=args.train_csv,
        data_paths=args.data_paths.split(','),
        packing_paths=args.packing_paths.split(','),
        mode='Train',
        max_size=args.max_size,
        kernel_size=args.kernel_size,
        pool_stride=args.pool_stride,
        device=device,
       
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers)

    # Model parameters
    params = {
        'patch_size': args.patch_size,
        'n_classes': args.n_classes,
        'embed_dim': args.embed_dim,
        'topk': args.topk,
        'device' : args.device
    }

    
    # Initialize model, optimizer, and losses
    model = EagleNet(**params).to(device)#EAGLE_NET(**params).to(device)
    # Apply Xavier initialization
    initialize_weights(model)
    learning_rate = args.learning_rate   # 0.001
    decay =         args.weight_decay   # 0.0001
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    weight_criterion =   CE(aggregate='sum')
   
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # Loss 1: CrossEntropy
    weight_criterion = CE(aggregate='sum')  # Loss 2: Weighted BCE Loss

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_function(epoch, train_loader, model, criterion, weight_criterion, optimizer, device)

    # Save the trained model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved at {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCGA Subtyping Model with L2 and L3 Losses")

    # Data and paths
    parser.add_argument('--train_csv', type=str, default='./git_split/TCGA_lung_training_df.csv', help='Path to the training CSV file')
    parser.add_argument('--data_paths', type=str, default="./train/LUAD,./train/LUSC", help='Comma-separated paths to LUAD and LUSC data directories')
    parser.add_argument('--packing_paths', type=str, default="./packing/LUAD,./packing/LUSC", help='Comma-separated paths to LUAD and LUSC packing directories')

    # Model hyperparameters
    parser.add_argument('--patch_size', type=int, default=2048, help='Patch size for the model')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--topk', type=int, default=10, help='Top-k attention pooling')
    parser.add_argument('--device', type=str, default='cuda', help='Cuda GPU index')

    # Training hyperparameters
    parser.add_argument('--learning_rate', type=float, default=10e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=10e-2, help='Weight decay for the optimizer')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--max_size', type=int, default=50000, help='Maximum size for pooling')
    parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size for pooling')
    parser.add_argument('--pool_stride', type=int, default=2, help='Stride for pooling')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')

    # Model saving
    parser.add_argument('--save_path', type=str, default='./best_tcga_model.pth', help='Path to save the trained model')

    args = parser.parse_args()
    main(args)

# python train.py --train_csv ./split/TCGA_lung_final_training_df.csv --data_paths ./train/LUAD,./train/LUSC --packing_paths ./packing/LUAD,./packing/LUSC --epochs 10 --save_path ./best_tcga_model.pth
