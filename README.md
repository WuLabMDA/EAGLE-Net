# EAGLE-Net: Preserving Global Tissue Structure and Attention-driven Local Awareness in Computational Pathology

**Abstract:** Recent methods for analyzing whole slide images (WSIs)leverage multiple instance learning (MIL) to handle the gigapixel scale by treating WSI as a bag of patches or instances. Despite the notable success, the existing MIL algorithms neglect the intricate global spatial structure of tissues and the local context of clinically relevant instances in the training process, which are of significant importance in understanding tumor micro-environment (TME). To ad-dress these limitations, we introduce EAGLE-Net, a novelframework that utilizes adaptive absolute positional encoding along with the neighborhood context of critical patchesin an end-to-end trainable manner. This approach extractsthe global spatial structure of tissues through a convolution-based multi-scale spatial encoding technique and highlightsimportant patches using an attention mechanism. This information is used to perform attention-guided region profiling. EAGLE-Net not only highlights essential patchesbut also captures their global spatial context and localneighborhood information, which enhances both model performance and interpretability. We extensively evaluatedEAGLE-Net on six prognostic and four diagnostic tasks using benchmark datasets. EAGLE-Net outperforms and/ormatches the performance of state-of-the-art supervised andunsupervised model

![EAGLE-Net Overview](./Assets/main_figure.png)

## Key Features

- **Global and Local Awareness**: Incorporates multi-scale absolute spatial encoding (MASE) to preserve global tissue structure and attention-driven region profiling for local awareness.
- **Attention Mechanism**: Highlights essential patches and their neighborhood to improve interpretability.
- **Efficient Patch Pooling**: Utilizes pooling layers to reduce computational complexity while preserving critical information.
- **Diverse Clinical Tasks**: Validated across six prognostic and four diagnostic tasks using benchmark datasets.
- **Customizable Training**: Adjustable hyperparameters and loss weights for domain-specific optimizations.

## Repository Structure

- **Packing Folder**: Contains positional information for patches.
- **Train/Test Folder**: Holds feature embeddings of different slides.
- **Core Python Files**:
  - `Eagle_net_model.py`: Defines the core EAGLE-Net architecture, including MASE and attention mechanisms.
  - `train_model.py`: Implements the training pipeline with loss functions and model optimization.
  - `test_model.py`: Tests the trained model on validation and test datasets.
  - `TCGA_loader.py`: Handles data loading and preprocessing, including patch extraction and pooling.
  - `weight_loss1.py`: Implements weighted cross-entropy loss for fine-grained control.
- **Requirements**: Lists all dependencies in `requirements.txt`.

---

## Installation

### Environment Setup

To use EAGLE-Net, follow these steps:

```bash
# Clone the repository
git clone https://github.com/WuLabMDA/EAGLE-Net.git
cd EAGLE-Net

# Create and activate the Conda environment
conda create -n eagle_net python=3.8 -y
conda activate eagle_net

# Install dependencies
pip install -r requirements.txt
```

---

## Data Preprocessing

Prepare your data using the following pipeline:

1. **Patch Extraction**: Use the `create_patches_fp.py` script to extract patches and their positional information.

2. **Feature Embedding**: Extract feature embeddings for patches using a pre-trained model.

---

## Training

To train the EAGLE-Net model, use:

```bash
python train_model.py \
    --train_csv ./data/train.csv \
    --data_paths ./train/LUAD,./train/LUSC \
    --packing_paths ./packing/LUAD,./packing/LUSC \
    --patch_size 2048 \
    --n_classes 2 \
    --embed_dim 512 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --save_path ./models/eagle_net.pth
```
```python
python train_model.py \
      --train_csv './git_split/TCGA_lung_training_df.csv' \
      --data_paths './train/LUAD,./train/LUSC' \
      --packing_paths './packing/LUAD,./packing/LUSC' \
      --patch_size 2048 --n_classes 2 --embed_dim 512 --topk 10 \
      --epochs 50 --batch_size 16 --learning_rate 0.001 \
      --save_path './models/eagle_net.pth'
      --data_paths './test/LUAD,./test/LUSC' \
```
---

## Testing

Run the trained model on test data:

```python
python test.py \
      --test_csv './git_split/TCGA_lung_test_df.csv' \
      --data_paths './test/LUAD,./test/LUSC' \
      --packing_paths './packing/LUAD,./packing/LUSC' \
      --batch_size 1 --num_workers 4  --device 'cuda' \
      --patch_size 2048 --n_classes 2
```

---

## Results
- **Treatment Recommendations**: Interpretability analysis of EAGLE-Net VS attention-based techniques. The first row of the figure illustrates a pathologist annotated slide alongside the attention heatmaps produced by Gated-Attention, CLAM, and the proposed EAGLE-Net. The second row shows zoomed-in regions in black square

![attention map](./Assets/attention_map.png)

- **Improved Outcomes**: Interpretablity and Visualization of ROIs in TCGA Subtyping using EAGLE-Net.
  
![attention tcga](./Assets/attention_TCGA.png)

---
## Citation

If you use EAGLE-Net in your research, please cite our work:

```bibtex
@article{eagle_net2025,
  title={Preserving Global Tissue Structure and Attention-driven Local Awareness in Computational Pathology},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  doi={}
}
```

For questions, issues, or contributions, please create a new issue in this repository.