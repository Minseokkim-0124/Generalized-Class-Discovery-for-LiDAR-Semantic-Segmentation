# GCDLSS: Generalized Category Discovery for LiDAR Semantic Segmentation

This is the official code repository for **Generalized Category Discovery for LiDAR Semantic Segmentation** presented at **WACV 2026**. This code is reconstructed based on [NOPS](https://github.com/LuigiRiz/NOPS).

The repository contains the implementation of a method for generalized category discovery in 3D semantic segmentation with limited labeled data. The approach leverages weak supervision and mean teacher frameworks to discover novel object categories in point cloud data.

## Overview

The method addresses the challenge of discovering novel object categories in 3D semantic segmentation tasks where only a limited number of labeled examples are available. It employs:

- **Mean Teacher Architecture**: Semi-supervised learning framework for consistency regularization
- **LaserMix Augmentation**: Domain-specific data augmentation for point clouds
- **Novel Candidate Classifier (NCC)**: Adaptive threshold learning for improved discovery
- **Kmeans Clustering**: For discovering novel category clusters

### Supported Datasets

- **SemanticKITTI**: Large-scale outdoor LiDAR semantic segmentation dataset
- **nuScenes**: Multi-modal autonomous driving dataset with 3D annotations

## Environment Setup

### Prerequisites (Conda)

- Python 3.8 CUDA 11.3
- spconv
- mmdet3d 1.2.0
- mmcv 2.0.0rc4
- mmdet 3.0.0
- mmengine 0.8.4
- pytorch-lightning 1.4.8 
- pytorch-scatter 1.2.0
- fast-pytorch-kmeans 0.2.0.1

## Project Structure

```
GCDLSS/
├── config/                    # Configuration files
│   ├── semkitti_minkunet.yaml # SemanticKITTI config
│   └── nuscenes_minkunet.yaml # nuScenes config
├── models/                    # Neural network architectures
│   ├── minkunet.py           # Base MinkUNet model 
│   ├── multiheadminkunet.py  # Multi-head variant
│   ├── encoder.py            # Encoder modules
│   ├── decoder.py            # Decoder modules
│   └── backbone.py           # Backbone networks
├── modules/                   # Training modules (Lightning)
│   ├── exp.py                # ExpPretrain: Stage-1 pretraining
│   └── exp_merge_mean_teacher.py  # ExpMergeDiscover_LaserMix_MeanTeacher_NCCAdaptive (Ours)
├── utils/                     # Utility functions
│   ├── dataset.py            # Dataset loading utilities
│   ├── dataset_remission_mean_teacher.py  # Extended dataset utilities
│   ├── clustering.py         # Clustering algorithms
│   ├── loss.py              # Loss functions
│   ├── eval.py              # Evaluation metrics
│   ├── callbacks.py         # PyTorch Lightning callbacks
│   ├── scheduler.py         # Learning rate schedulers
│   └── visualize.py         # Visualization utilities
├── split_npy/                # Split indices for train/val/test
│   ├── kitti/               # SemanticKITTI splits
│   └── nusc/                # nuScenes splits
├── main.py                   # Training script
└── requirements.txt          # Python dependencies
```


## Usage

### Stage 1: Pre-training

Pre-train the backbone network on all labeled data:

```bash
python main.py \
  -s 1 \
  --dataset SemanticKITTI \
  --dataset_config config/semkitti_minkunet.yaml \
  --module ExpPretrain \
  --experiment pretrain-split1 \
  --checkpoint_dir ./checkpoints/pretrain \
  --log_dir ./logs/pretrain \
  --epochs 50 \
  --batch_size 4 \
  --use_scheduler \
  --train_lr 0.01
```

For nuScenes:
```bash
python main.py \
  -s 1 \
  --dataset nuScenes \
  --dataset_config config/nuscenes_minkunet.yaml \
  --module ExpPretrain \
  --experiment pretrain-nusc-split1 \
  --checkpoint_dir ./checkpoints/pretrain \
  --log_dir ./logs/pretrain \
  --epochs 50 \
  --use_scheduler
```

### Stage 2: Novel Category Discovery

Run the main discovery training with the following command:

```bash
python main.py \
  -s 1 \
  --dataset SemanticKITTI \
  --dataset_config config/semkitti_minkunet.yaml \
  --use_scheduler \
  --module ExpMergeDiscover_LaserMix_MeanTeacher_NCCAdaptive \
  --experiment GCDLSS-split1 \
  --checkpoint_dir ./checkpoints/gcdlss \
  --log_dir ./logs/gcdlss \
  --pretrained ./checkpoints/pretrain/epoch=46-step=115713.ckpt \
  --epochs 50 \
  --batch_size 4 \
  --train_lr 0.01 \
  --finetune_lr 0.0001
```

For nuScenes:
```bash
python main_v2.py -s 0 --dataset_config config/nuscenes_minkunet.yaml --use_scheduler --module ExpPretrain --experiment nusc-split100-pretrain --checkpoint_dir ./AAAI_ckpt/nusc_ckpt/pretrain/split100 --log_dir ./AAAI_logdir/pretrain-nusc --dataset nuScenes
```


## Data Split Configuration

The method evaluates on three different unknown class splits:

### SemanticKITTI Splits
- **Split 1**: Unknown classes = [4, 6] (other-vehicle, bicyclist), labled data ratio = 50%
- **Split 2**: Unknown classes = [2, 4, 5] (motorcycle, other-vehicle, person), labled data ratio = 45%
- **Split 3**: Unknown classes = [3, 6, 9] (parking, bicyclist, truck), labled data ratio = 40%

### nuScenes Splits
- **Split 1**: Unknown classes = [1, 8] (bicycle, trailer), labled data ratio = 45%
- **Split 2**: Unknown classes = [1, 2, 5] (bicycle, bus, motorcycle), labled data ratio = 35%
- **Split 3**: Unknown classes = [0, 4, 7, 8] (barrier, motorcycle, trailer, bicycle), labled data ratio = 25%



## Citation

Please cite the following paper if you use our code:

```
TBD
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

For questions or issues, please open an issue on the repository.

