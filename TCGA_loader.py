# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:37:38 2024

@author: mwaqas
"""

import warnings

# Suppress all UserWarnings globally
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import random
import tqdm
import gc

# Set seeds for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


# Dataset class
class Subtype_CustomDataset(Dataset):
    def __init__(self, labels_paths, data_paths, packing_paths, mode='Test', max_size=50000, kernel_size=2, pool_stride=2,
                 device='cpu'):
        self.data_paths = data_paths
        self.packing_paths = packing_paths
        self.df = pd.read_csv(labels_paths)
        self.mode = mode
        self.device = device
        self.max_size = max_size
        self.kernel_size = kernel_size
        self.pool_stride = pool_stride

        self.slide_ids = self.df['slide_id'].tolist()
        self.labels = self.df['label'].tolist()
        self.datasets = self.df['dataset'].tolist()

        self.all_files = self.get_all_directories()
        self.df_pooling_details = pd.DataFrame(
            columns=["slide_id", "cube_shape", "pooled_cube_shape", "mask_shape", "pooled_mask_shape"]
        )

    def get_all_directories(self):
        """
        Retrieve all directories from the packing paths.
        """
        all_files = []
        for path in self.packing_paths:
            directories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            all_files.append(directories)
        return all_files

    def read_files(self, idxs, row):
        """
        Read .npy data and corresponding packing files for a given row.
        """
        data_type = row['dataset']
        slide_id = row['slide_id']
        data_path_index = 0 if data_type == "LUAD" else 1
        packing_path_index = 0 if data_type == "LUAD" else 1

        data_path_actual = self.data_paths[data_path_index]
        packing_path_base = self.packing_paths[packing_path_index]

        data_path = os.path.join(data_path_actual, f"{slide_id}.npy")
        if os.path.exists(data_path):
            data = np.load(data_path)
        else:
            raise FileNotFoundError(f"Data file not found for slide ID: {slide_id}")

        # Load packing data
        all_folders = self.all_files[packing_path_index]
        matching_folders = [f for f in all_folders if f.startswith(slide_id)]
        if len(matching_folders) == 0:
            raise FileNotFoundError(f"Packing folder not found for slide ID: {slide_id}")

        packing_path = os.path.join(packing_path_base, matching_folders[0], f"{slide_id}.csv")
        if not os.path.exists(packing_path):
            raise FileNotFoundError(f"Packing CSV not found for slide ID: {slide_id}")

        packing_df = pd.read_csv(packing_path)

        # Cube and mask construction
        df_selected = packing_df[['x_optimized', 'y_optimized', 'optimized_shape']]
        height, width = map(int, df_selected['optimized_shape'].iloc[0].strip('()').split(', '))
        cube_depth = data.shape[-1]

        cube_3d = np.zeros((height, width, cube_depth), dtype=np.float32)
        mask_2d = np.zeros((height, width), dtype=np.float32)

        x_coords = df_selected['x_optimized'].values
        y_coords = df_selected['y_optimized'].values

        cube_3d[y_coords, x_coords, :] = data
        mask_2d[y_coords, x_coords] = 1

        if self.mode in ['Test', 'Validate']:
            return data, cube_3d, mask_2d

        # Pool the cube if it exceeds max size
        if self.mode == 'Train':
            while (cube_3d.shape[0] * cube_3d.shape[1]) > self.max_size:
                cube_3d, mask_2d, row = self.pool_cube(cube_3d, cube_3d.shape[0], cube_3d.shape[1], cube_depth, mask_2d,
                                                       self.kernel_size, self.pool_stride, slide_id)
                self.df_pooling_details = pd.concat([self.df_pooling_details, pd.DataFrame([row])], ignore_index=True)

            return data, cube_3d, mask_2d

    def pool_cube(self, cube, height, width, depth, mask_2d, pool_kernel_size, pool_stride, slide_id):
        """
        Pool a cube and mask to reduce their sizes.
        """
        new_height = height // pool_stride
        new_width = width // pool_stride

        adjusted_cube_3d = torch.tensor(cube[:new_height * pool_stride, :new_width * pool_stride, :], dtype=torch.float)
        adjusted_mask_2d = torch.tensor(mask_2d[:new_height * pool_stride, :new_width * pool_stride], dtype=torch.float)

        cube_pooling_layer = torch.nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        pooling_layer = torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        pooled_cube = cube_pooling_layer(adjusted_cube_3d.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        pooled_mask = pooling_layer(adjusted_mask_2d.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        row = {
            "slide_id": slide_id,
            "cube_shape": str(cube.shape),
            "pooled_cube_shape": str(pooled_cube.shape),
            "mask_shape": str(mask_2d.shape),
            "pooled_mask_shape": str(pooled_mask.shape)
        }

        del cube, mask_2d, adjusted_cube_3d, adjusted_mask_2d
        gc.collect()

        return pooled_cube.numpy(), pooled_mask.numpy(), row

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data, cube_3d, mask_2d = self.read_files([idx], row)
        return data, cube_3d, mask_2d, row['label'], row['slide_id'], row['dataset']


# # Example usage
# if __name__ == "__main__":
#     data_paths = [
#         "./train/LUAD",
#         "./train/LUSC"
#     ]
#     packing_paths = [
#         "./packing/LUAD",
#         "./packing/LUSC"
#     ]

#     train_csv = './git_split/TCGA_lung_training_df.csv'
#     device = 'cuda'

#     dataset = Subtype_CustomDataset(train_csv, data_paths, packing_paths, mode='Train', max_size=50000)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     for batch_idx, (data, cube_3d, mask_2d, label, slide_id, subtype) in tqdm.tqdm(enumerate(dataloader)):
#         print('batch index is  = ', batch_idx)
#         print('Matrix shape is = ', cube_3d.shape)
#         print('Bag shape is  = ', data.shape)
#         print('Mask_2d shape is  = ', mask_2d.shape)
#         if cube_3d.shape[:2] != mask_2d.shape:
#             print(f"Shape mismatch at batch {batch_idx}")
#         continue

#     dataset.df_pooling_details.to_csv(f"{dataset.mode}_pool_details.csv", index=False)
