
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import gc




seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2048, embed_dim=200):
        super(PatchEmbed, self).__init__()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(patch_size, embed_dim),
            # nn.ReLU(),
            # nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.feature_extractor_part1(x)

class MASE(nn.Module):
    """A 3D CNN architecture with multiple filter sizes."""
    def __init__(self):
        super(MASE, self).__init__()
        self.conv1x1 = nn.Conv3d(1, 8, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv3d(1, 8, kernel_size=5, stride=1, padding=2)
        self.conv7x7 = nn.Conv3d(1, 8, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU()
        self.conv_reduce = nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.relu(self.conv1x1(x))
        x3 = self.relu(self.conv3x3(x))
        x5 = self.relu(self.conv5x5(x))
        x7 = self.relu(self.conv7x7(x))
        # print(x1.shape,  x3.shape, x5.shape, x7.shape)
        cat_x = torch.cat([x1, x3, x5, x7], dim=1).contiguous()
      
        
        x =  self.relu(self.conv_reduce(cat_x)) + x
        
        del cat_x, x1, x3, x5, x7
        torch.cuda.empty_cache()
        gc.collect()
        
      
        
        
        return x
def get_neighbors(row_vector, col_vector, max_rows, max_cols, device = 'cuda'):
    """
    Get all neighbors (including diagonals) for each index in row_vector and col_vector.
    
    Args:
        row_vector (Tensor): Tensor containing row indices.
        col_vector (Tensor): Tensor containing column indices.
        max_rows (int): The maximum number of rows.
        max_cols (int): The maximum number of columns.
        
    Returns:
        Tuple of tensors: Row and column neighbors combined for all indices.
    """
    device = row_vector.device  # Get the device of the input tensors
    
    # Define all possible relative neighbor positions
    relative_positions = torch.tensor([
        (-1, -1), (-1, 0), (-1, 1),  # Top-left, Top, Top-right
        (0, -1),  (0, 0),  (0, 1),   # Left, Center, Right
        (1, -1),  (1, 0), (1, 1)      # Bottom-left, Bottom, Bottom-right
    ], dtype=torch.long, device=device)  # Move to the same device as row_vector and col_vector
    
    # Expand row_vector and col_vector dimensions
    row_vector = row_vector.unsqueeze(1)
    col_vector = col_vector.unsqueeze(1)
    
    # Calculate neighbor indices
    neighbors_rows = row_vector + relative_positions[:, 0]
    neighbors_cols = col_vector + relative_positions[:, 1]
    
    # Mask out-of-bound indices
    valid_rows = (neighbors_rows >= 0) & (neighbors_rows < max_rows)
    valid_cols = (neighbors_cols >= 0) & (neighbors_cols < max_cols)
    valid_neighbors = valid_rows & valid_cols
    
    # Apply the mask and filter valid neighbors
    neighbors_rows = neighbors_rows[valid_neighbors]
    neighbors_cols = neighbors_cols[valid_neighbors]
    
    return neighbors_rows.flatten(), neighbors_cols.flatten()

class EagleNet(nn.Module):
    def __init__(self, patch_size, n_classes, embed_dim, topk=20, device = 'cuda'):
        super(EagleNet, self).__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.ATTENTION_BRANCHES = n_classes
        self.norm_C = nn.LayerNorm(embed_dim, eps=1e-6)
        self.topk = topk

        self.classifier_att = nn.Sequential(
            nn.Linear(embed_dim, n_classes),
           
        )
        
        self.mase = MASE()
        self.device = device
    def forward(self, cube, mask=None, bag_label=None, test=False, option=None, mode = 'Train'):
     
        
        # patch embedding 
        cube = self.patch_embed(cube)
        
        # CNN Module to integrate the positional encoding with input
        x = self.mase(cube.unsqueeze(0))
        x = x.squeeze().view(-1, x.size(-1))       
        x = self.norm_C(x) 
        
      
        # Attention Weighting 
        out_c = self.classifier_att(x)
        out = out_c - out_c.max()
        out = out.exp()
        out = out.sum(1, keepdim=True)
        alpha = out / out.sum(0)
        alpha = alpha.t()
        v = torch.mm(alpha,x)
        logits = self.classifier_att(v)
        
        if mode=='Test':
            return logits
        # compute mask for background patches
        else:
            if mask is not None:
                mask_flat = mask.flatten()
                bool_index = mask_flat == 0
                A_masked = alpha[:,bool_index]
                A_masked = A_masked.sum()
                # x = x[bool_index]
                
                
            # select the neighborhood of the top-K 
        
            # Select top-K attention scores and their indices
            topk_scores, topk_indices = torch.topk(alpha, self.topk, dim=1, largest=True, sorted=False)
            topk_scores, topk_indices = topk_scores.to(self.device), topk_indices.to(self.device)
            
            # Calculate rows and columns in cube from the flat indices
            rows = topk_indices//cube.shape[2]
            cols = topk_indices % cube.shape[2]
            r,c = get_neighbors(rows.squeeze(), cols.squeeze(), cube.shape[1], cube.shape[2])
            
            # Convert neighbor rows and columns to linear indices
            neighbor_linear_indices = (r) * cube.shape[2] + c
            neighbor_linear_indices = neighbor_linear_indices
            
            out_c_nei = out_c[neighbor_linear_indices,:]
            alpha_nei = alpha[:, neighbor_linear_indices]
         
            del x, cube, mask_flat, alpha, bool_index, out
            torch.cuda.empty_cache() 
            gc.collect()
            
            return logits, A_masked, out_c_nei, alpha_nei

      

    def get_weights(self, cube, mask=None, bag_label=None, test=False, option=None):
        cube = self.patch_embed(cube)
        x = self.mase(cube.unsqueeze(0))
        x = x.squeeze()
       
        x = x.view(-1, x.size(-1))
        x = self.norm_C(x) 
        
       
        out_c = self.classifier_att(x)
        out = out_c - out_c.max()
        out = out.exp()
        out = out.sum(1, keepdim=True)
        alpha = out / out.sum(0)
        alpha = alpha.t()
        
        
        v = torch.mm(alpha,x)
        logits = self.classifier_att(v)
        
        return logits, alpha
