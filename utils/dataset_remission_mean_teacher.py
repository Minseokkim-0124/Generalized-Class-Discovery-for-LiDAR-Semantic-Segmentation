from copy import deepcopy
import os
import pickle 
import bisect
import re

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.utils
import yaml

from typing import Tuple
from utils.voxelizer import Voxelizer
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from utils.utils_polarmix import polarmix
# from mmengine.dataset.base_dataset import force_full_init

def get_dataset(name, type='disjoint'):
    if name == "SemanticKITTI":
        if type == 'labeled':
            return SemanticKITTIRestrictedDatasetLabel
        elif type == 'disjoint':
            return SemanticKITTIRestrictedDisjointDataset
        elif type == 'polarmix':
            return SemanticKITTI_PolarMix_RealAug_Dataset
        elif type == 'lasermix':
            return SemanticKITTI_LaserMix_RealAug_Dataset
        else:
            raise NameError(f'SemanticKITTI Dataset "{type}" is not yet implemented')
    elif name == "nuScenes":
        if type == 'labeled':
            return nuScenesRestrictedDataset
        elif type == 'disjoint':
            return nuScenesRestrictedDisjointDataset
        elif type =='finetuning': 
            return nuScenesRestrictedFineTuningDataset
        else:
            return nuScenes_LaserMix_RealAug_Dataset


class SemanticKITTIDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        augment=False,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            # selected_idx = np.random.choice(
            #     coordinates.shape[0], self.downsampling, replace=False
            # )
            selected_idx = np.arange(coordinates.shape[0])
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]

        discrete_coords, unique_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return discrete_coords, unique_feats, unique_labels, selected_idx, t

class SemanticKITTIRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        augment=False,
        label_mapping=None,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []

        self.voxel_size = voxel_size
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission

        features = np.ones((coordinates.shape[0], 1))

        # AUGMENTATION
        if self.augment:
            if self.downsampling == -1:
                selected_idx = np.arange(coordinates.shape[0])
            else:
                selected_idx = np.random.choice(
                    coordinates.shape[0], self.downsampling, replace=False
                )
                selected_idx = np.sort(selected_idx)

            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)

        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]

        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            t,
        )

class SemanticKITTIRestrictedDatasetLabel(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        percentage = 100,
        label_mapping=None,
        output_voxelize=True,
        nusc=None,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        if split != "test":
            self.files["label"] = []
        self.filenames = []
        self.downsampling = downsampling
        if split == 'train':
            if percentage == 100:
                self.ann_file ='semantickitti_infos_train.pkl'
            else:
                self.ann_file = 'semantickitti_infos_train.{}.pkl'.format(percentage)
        elif split == 'valid':
            self.ann_file = 'semantickitti_infos_val.pkl'
            self.downsampling = -1
        self.ann_root = os.path.join(self.config["dataset_path"], self.ann_file)

        self.voxel_size = voxel_size
        
        with open(self.ann_root, 'rb') as f:
            annotation = pickle.load(f)
            
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for data in annotation['data_list']:
            for idx, type in enumerate(self.files.keys()):
                if idx == 0:
                    files_path = os.path.join(self.config["dataset_path"],
                    data['lidar_points']['lidar_path'])
                    if not os.path.exists(files_path):
                        raise RuntimeError("Point cloud input directory missing: " + files_path)
                    self.files[type].append(files_path)
                    self.filenames.append(files_path)
                else:
                    files_path = os.path.join(self.config["dataset_path"],
                    data['pts_semantic_mask_path'])
                    if not os.path.exists(files_path):
                        raise RuntimeError("Point cloud label directory missing: " + files_path)
                    self.files[type].append(files_path)
        
        if len(self.filenames) == 0:
            raise RuntimeError("Missing data file")
        
        self.num_files = len(self.filenames)

        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        self.output_voxelize =output_voxelize

    def full_init(self):
        pass  # For compatibility with ConCatDataset

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        
        features = np.reshape(remissions, (remissions.shape[0],1))

        # AUGMENTATION
        if self.augment:
            if self.downsampling == -1:
                selected_idx = np.arange(coordinates.shape[0])
            else:
                selected_idx = np.random.choice(
                    coordinates.shape[0], self.downsampling, replace=False
                )
                selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        
        if self.output_voxelize:
            discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=coordinates,
                return_index=True,
                return_inverse=True,
                quantization_size=self.voxel_size,
            )

            unique_feats = features[unique_map]
            unique_labels = labels[unique_map]
            unique_mapped_labels = mapped_labels[unique_map]
            selected_idx = selected_idx[unique_map]

            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx,
                unique_mapped_labels,
                inverse_map,
                t,
            )
        else: 
            inverse_map = None 
            return(
                coordinates,
                features,
                labels,
                selected_idx,
                mapped_labels,
                inverse_map,
                t
            )
           
class SemanticKITTIRestrictedDatasetCleanSplit(SemanticKITTIRestrictedDataset):
    def __init__(
        self,
        clean_mask,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        augment=False,
        label_mapping=None,
    ):
        super().__init__(
            config_file, split, voxel_size, augment, label_mapping
        )
        self.filenames = np.array(self.filenames)[clean_mask]
        self.num_files = len(self.filenames)
        for key in self.files.keys():
            self.files[key] = np.array(self.files[key])[clean_mask]

class SemanticKITTIRestrictedDisjointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        data_idx_file='selected_idx.npy',
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        labeled=True,
        store_unlabeled_label = False,
        nusc=None,
        resize_aug = False,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.files = {"input": []}

        if labeled and split != "test":
            self.files["label"] = []
        elif store_unlabeled_label:
            self.files["label"] = []
                
        self.filenames = []
        self.labeled = labeled
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )
            self.fixed_selected_idx = None

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        if not os.path.isfile(data_idx_file):
            raise RuntimeError("Data index file does not exist: " + data_idx_file)
        
        selected_data_idx = np.load(data_idx_file)
        real_data_idx = np.arange(len(self.filenames))
        not_selected_data_idx = [x for x in real_data_idx if x not in selected_data_idx]

        if labeled:
            self.filenames = np.array(self.filenames.copy())[selected_data_idx].tolist()
            for key in self.files.keys():
                self.files[key] = np.array(self.files[key].copy())[selected_data_idx].tolist()
        else:
            self.filenames = np.array(self.filenames.copy())[not_selected_data_idx].tolist()
            for key in self.files.keys():
                self.files[key] = np.array(self.files[key].copy())[not_selected_data_idx].tolist()
  
        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
            
        self.num_files = len(self.files['input'])

    def __len__(self):
        return self.num_files

    def set_fixed_selected_idx(self, selected_idx):
        self.fixed_selected_idx = selected_idx 
        
    def __getitem__(self, t):
        
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        features = np.reshape(remissions, (remissions.shape[0],1))

        # AUGMENTATION
        if self.augment:
            if self.fixed_selected_idx is not None:
            # 고정된 인덱스를 사용하는 경우
                selected_idx = self.fixed_selected_idx
            else:
                # DOWNSAMPLING
                if self.downsampling == -1:
                    selected_idx = np.arange(coordinates.shape[0])
                else:
                    selected_idx = np.random.choice(
                        coordinates.shape[0], self.downsampling, replace=False
                    )
                    selected_idx = np.sort(selected_idx)
                    
                    selected_idx_ = np.random.choice(
                        coordinates.shape[0], self.downsampling, replace=False
                    )
                    selected_idx_ = np.sort(selected_idx_)

            coordinates = coordinates[selected_idx]
            features = features[selected_idx]
            
            coordinates_ = coordinates[selected_idx_]
            features_ = features[selected_idx_]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            if self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        else: 
            mapped_labels = np.copy(labels)

        # discrete_coords = coordinates 
        # unique_feats = features
        # unique_labels = labels
        # unique_mapped_labels = mapped_labels
        # inverse_map = None
        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )

        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
                
        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            t,
        )

class SemanticKITTI_LaserMix_RealAug_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        data_idx_file='selected_idx.npy',
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        labeled=True,
        store_unlabeled_label = False,
        resize_aug = True,
        nusc=None,
        output_both_voxel_point=False,
        resize_only_voxel=False,
        
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.files = {"input": []}

        if labeled and split != "test":
            self.files["label"] = []
        elif store_unlabeled_label:
            self.files["label"] = []
                
        self.filenames = []
        self.labeled = labeled
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.resize_aug = resize_aug
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        if not os.path.isfile(data_idx_file):
            raise RuntimeError("Data index file does not exist: " + data_idx_file)
        
        # For REAL Augmentation
        unknown_labels = re.findall(r'\d+', data_idx_file)
        self.unknown_labels = [int(num) for num in unknown_labels]
        self.unknown_labels = self.unknown_labels[:-1]
        
        selected_data_idx = np.load(data_idx_file)
        real_data_idx = np.arange(len(self.filenames))
        not_selected_data_idx = [x for x in real_data_idx if x not in selected_data_idx]
        if labeled:
            self.filenames = np.array(self.filenames.copy())[selected_data_idx].tolist()
            for key in self.files.keys():
                self.files[key] = np.array(self.files[key].copy())[selected_data_idx].tolist()
        else:
            self.filenames = np.array(self.filenames.copy())[not_selected_data_idx].tolist()
            for key in self.files.keys():
                self.files[key] = np.array(self.files[key].copy())[not_selected_data_idx].tolist()

        
        if label_mapping is not None:
            if self.resize_aug:
                label_mapping[100] = 19 - len(self.unknown_labels)
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
            
        self.num_files = len(self.files['input'])
        self.output_both_voxel_point = output_both_voxel_point
        self.resize_only_voxel = resize_only_voxel
       

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        features = np.reshape(remissions, (remissions.shape[0],1))

        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            if self.downsampling == -1:
                selected_idx = np.arange(coordinates.shape[0])
            else:
                selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
                )   
                selected_idx = np.sort(selected_idx)

            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            annotated_data = np.fromfile(label_filename, dtype=np.int32)
            # annotated_data_uint = np.fromfile(label_filename, dtype=np.uint32)
            annotated_data = annotated_data.reshape((-1))
            # annotated_data_uint = annotated_data_uint.reshape((-1))
            labels = annotated_data & 0xFFFF
            instance_data = annotated_data >> 16
            # instance_data_uint = annotated_data_uint >> 16

            # unique_instance_data_uint = np.unique(instance_data_uint)
            # unique_instance_data_uint_bin = [bin(val) for val in unique_instance_data_uint]
            # unique_instance_data = np.unique(instance_data)
            # unique_instance_data_bin = [bin(val) for val in unique_instance_data]
            if self.augment:
                labels = labels[selected_idx]
                instance_data = instance_data[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))
        

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            instance_data = instance_data[labelled_idx]
            if self.resize_aug:
                if self.resize_only_voxel:
                    resized_coordinates, resized_features, resized_labels, resized_instance_data = self.ds_sample(coordinates, features, labels, instance_data)
                else:
                    coordinates, features, labels, instance_data = self.ds_sample(coordinates, features, labels, instance_data)
            
            if self.label_mapping_function is not None:
                if self.resize_only_voxel:
                    resized_mapped_labels = self.label_mapping_function(resized_labels)
                    mapped_labels = self.label_mapping_function(labels)
                else:
                    mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        
        if self.augment:
            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations
            if self.resize_only_voxel:
                resized_homo_coords = np.hstack(
                    (
                        resized_coordinates,
                        np.ones((resized_coordinates.shape[0], 1), dtype=coordinates.dtype),
                    )
                )
                # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
                resized_coordinates = resized_homo_coords @ rigid_transformation.T[:, :3]
                
                homo_coords = np.hstack(
                    (
                        coordinates,
                        np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                    )
                )
                # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
                coordinates = homo_coords @ rigid_transformation.T[:, :3]
            
            else:
                homo_coords = np.hstack(
                    (
                        coordinates,
                        np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                    )
                )
                # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
                coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        

        # discrete_coords = coordinates 
        # unique_feats = features
        # unique_labels = labels
        # unique_mapped_labels = mapped_labels
        # inverse_map = None
        if self.output_both_voxel_point:
            if self.resize_only_voxel:
                discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
                    coordinates=resized_coordinates,
                    return_index=True,
                    return_inverse=True,
                    quantization_size=self.voxel_size,
                )
                unique_feats = resized_features[unique_map]
                unique_labels = resized_labels[unique_map]
                unique_mapped_labels = resized_mapped_labels[unique_map]
                selected_idx = selected_idx[unique_map]
                        
                return (
                    coordinates,
                    features,
                    labels,
                    selected_idx,
                    mapped_labels,
                    discrete_coords,
                    unique_feats,
                    unique_labels,
                    selected_idx,
                    unique_mapped_labels,
                    inverse_map,
                    t,
                )
            else:
                discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
                    coordinates=coordinates,
                    return_index=True,
                    return_inverse=True,
                    quantization_size=self.voxel_size,
                )
                unique_feats = features[unique_map]
                unique_labels = labels[unique_map]
                unique_mapped_labels = mapped_labels[unique_map]
                selected_idx = selected_idx[unique_map]
                        
                return (
                    coordinates,
                    features,
                    labels,
                    selected_idx,
                    mapped_labels,
                    discrete_coords,
                    unique_feats,
                    unique_labels,
                    selected_idx,
                    unique_mapped_labels,
                    inverse_map,
                    t,
                )
        else:
            return (
                    discrete_coords,
                    unique_feats,
                    unique_labels,
                    selected_idx,
                    unique_mapped_labels,
                    inverse_map,
                    t,
                )
    
    def ds_sample(self, xyz, ref, labels, instances):        
        minimum_pts_thre = 300
        instances = instances.squeeze()
        cls, cnt = np.unique(instances, return_counts=True)
        inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]
        unknown_clss = self.unknown_labels
        # if dataset_kitti:
        #     unknown_clss = [5]
        # if dataset_nusc:
        #     unknown_clss = [1, 5, 8, 9]
        for instance_idx in inst_basic_idx:
            rnd = np.random.rand()
            if rnd > 0.5 or labels[instances == instance_idx][0] in unknown_clss:
                continue

            obj_ins = xyz[instances==instance_idx]
            obj_ins_center = np.mean(obj_ins, axis=0)
            obj_ins = obj_ins - obj_ins_center
            scale_ds_large = np.random.rand()*1.5+1.5
            scale_ds_small = np.random.rand()*0.25+0.25
            rnd = np.random.rand()
            scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
            obj_ins = obj_ins * scale_ds + obj_ins_center
            xyz[instances == instance_idx] = obj_ins
            
            # For Only KITTI DATASET
            labels[instances == instance_idx] = 100
            # if dataset_nusc:
            #     labels[instances == instance_idx] = 17
                
        return xyz, ref, labels, instances
        
class SemanticKITTI_PolarMix_RealAug_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        data_idx_file='selected_idx.npy',
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        labeled=True,
        store_unlabeled_label = False,
        resize_aug = True,
        nusc=None,
        polarmix=True,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.files = {"input": []}

        if labeled and split != "test":
            self.files["label"] = []
        elif store_unlabeled_label:
            self.files["label"] = []
                
        self.filenames = []
        self.labeled = labeled
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.resize_aug = resize_aug
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )

        for sequence in self.config["split_sequence"][split]:
            for idx, type in enumerate(self.files.keys()):
                files_path = os.path.join(
                    self.config["dataset_path"],
                    "sequences",
                    sequence,
                    self.config["folder_name"][type],
                )
                if not os.path.exists(files_path):
                    raise RuntimeError("Point cloud directory missing: " + files_path)
                files = os.listdir(files_path)
                data = sorted([os.path.join(files_path, f) for f in files])
                if len(data) == 0:
                    raise RuntimeError("Missing data for " + type)
                self.files[type].extend(data)
                if idx == 0:
                    self.filenames.extend(data)

        if not os.path.isfile(data_idx_file):
            raise RuntimeError("Data index file does not exist: " + data_idx_file)
        
        # For REAL Augmentation
        unknown_labels = re.findall(r'\d+', data_idx_file)
        self.unknown_labels = [int(num) for num in unknown_labels]
        self.unknown_labels = self.unknown_labels[:-1]
        
        selected_data_idx = np.load(data_idx_file)
        real_data_idx = np.arange(len(self.filenames))
        not_selected_data_idx = [x for x in real_data_idx if x not in selected_data_idx]
        if labeled:
            self.filenames = np.array(self.filenames.copy())[selected_data_idx].tolist()
            for key in self.files.keys():
                self.files[key] = np.array(self.files[key].copy())[selected_data_idx].tolist()
        else:
            self.filenames = np.array(self.filenames.copy())[not_selected_data_idx].tolist()
            for key in self.files.keys():
                self.files[key] = np.array(self.files[key].copy())[not_selected_data_idx].tolist()

        
        if label_mapping is not None:
            if self.resize_aug:
                label_mapping[100] = 19 - len(self.unknown_labels)
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        self.num_files = len(self.files['input'])
        self.polarmix = polarmix

    def __len__(self):
        return self.num_files

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        features = scan[:, 3]  # get remission
        features = np.reshape(features, (features.shape[0],1))

        if self.polarmix:
            idx2 = np.random.randint(self.num_files)
            filename2 = self.files["input"][idx2]
            scan2 = np.fromfile(filename2, dtype=np.float32)
            scan2 = scan2.reshape((-1, 4))
            coordinates2 = scan2[:, 0:3]
            features2 = scan2[:, 3]
            features2 = np.reshape(features2, (features2.shape[0], 1))

        
        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            annotated_data = np.fromfile(label_filename, dtype=np.int32)
            # annotated_data_uint = np.fromfile(label_filename, dtype=np.uint32)
            annotated_data = annotated_data.reshape((-1))
            # annotated_data_uint = annotated_data_uint.reshape((-1))
            labels = annotated_data & 0xFFFF
            instance_data = annotated_data >> 16
            # instance_data_uint = annotated_data_uint >> 16
            # unique_instance_data_uint = np.unique(instance_data_uint)
            # unique_instance_data_uint_bin = [bin(val) for val in unique_instance_data_uint]
            # unique_instance_data = np.unique(instance_data)
            # unique_instance_data_bin = [bin(val) for val in unique_instance_data]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)

            if self.polarmix:
                label_filename2 = self.files["label"][idx2]
                annotated_data2 = np.fromfile(label_filename2, dtype=np.int32)
                annotated_data2 = annotated_data2.reshape((-1))
                labels2 = annotated_data2 & 0xFFFF
                instance_data2 = annotated_data2 >> 16
                for index, element in enumerate(labels2):
                    labels2[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            # selected_idx = selected_idx[labelled_idx]
            instance_data = instance_data[labelled_idx]
            
            if self.polarmix:
                labelled_idx = labels2 != -1
                coordinates2 = coordinates2[labelled_idx]
                features2 = features2[labelled_idx]
                labels2 = labels2[labelled_idx]
                # selected_idx = selected_idx[labelled_idx]
                instance_data2 = instance_data2[labelled_idx]

            if self.resize_aug:
                coordinates, features, labels, instance_data = self.ds_sample(coordinates, features, labels, instance_data)
                if self.polarmix:
                    coordinates2, features2, labels2, instance_data2 = self.ds_sample(coordinates2, features2, labels2, instance_data2)
            
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
                if self.polarmix:
                    mapped_labels2 = self.label_mapping_function(labels2)
            else:
                mapped_labels = np.copy(labels)

        # AUGMENTATION
        if self.augment:
            if self.polarmix: 
                # instance_classes = [10, 11, 15, 18, 20, 30, 31, 32]
                instance_classes = [0, 1, 2, 3, 4 ,5 ,6, 7]
                alpha = (np.random.random() -1) * np.pi
                beta = alpha + np.pi
                Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]
                
                scan = np.concatenate((coordinates, features), axis=1)
                scan2 = np.concatenate((coordinates2, features2), axis=1)

                polarmix_scan, polarmix_labels = polarmix(scan, labels, scan2, labels2,
                                        alpha=alpha, beta=beta,
                                        instance_classes=instance_classes,
                                        Omega=Omega)
                polarmix_coordinates = polarmix_scan[:, 0:3]  
                polarmix_features = polarmix_scan[:, 3]  
                polarmix_features = np.reshape(polarmix_features, (polarmix_features.shape[0],1))
               
                if self.label_mapping_function is not None:
                    polarmix_mapped_labels = self.label_mapping_function(polarmix_labels)

        if self.augment:
            # TRANSFORMATIONS (Rot Trans Scale)
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]

            if self.polarmix:
                homo_coords = np.hstack(
                (
                    polarmix_coordinates,
                    np.ones((polarmix_coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
                polarmix_coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])

        # DOWNSAMPLING
        if self.augment:
            if self.polarmix:
                if self.downsampling == -1:
                    selected_idx2 = np.arange(polarmix_coordinates.shape[0])
                else:
                    selected_idx2 = np.random.choice(
                    polarmix_coordinates.shape[0], self.downsampling, replace=False
                    )   
                    selected_idx2 = np.sort(selected_idx2)
                
                polarmix_coordinates = polarmix_coordinates[selected_idx2]
                polarmix_features = polarmix_features[selected_idx2]

                discrete_coords2, unique_map2, inverse_map2 = ME.utils.sparse_quantize(
                    coordinates=polarmix_coordinates,
                    return_index=True,
                    return_inverse=True,
                    quantization_size=self.voxel_size,
                )
                unique_feats2 = polarmix_features[unique_map2]
                unique_labels2 = polarmix_labels[unique_map2]
                unique_mapped_labels2 = polarmix_mapped_labels[unique_map2]
                selected_idx2 = selected_idx2[unique_map2]

            if self.downsampling == -1:
                selected_idx = np.arange(coordinates.shape[0])
            else:
                selected_idx = np.random.choice(
                coordinates.shape[0], self.downsampling, replace=False
                )   
                selected_idx = np.sort(selected_idx)
            
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]


        # discrete_coords = coordinates 
        # unique_feats = features
        # unique_labels = labels
        # unique_mapped_labels = mapped_labels
        # inverse_map = None
        


        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )
        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        
        if self.polarmix:
            return (
                discrete_coords2,
                unique_feats2,
                unique_labels2,
                selected_idx2,
                unique_mapped_labels2,
                inverse_map2,
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx,
                unique_mapped_labels,
                inverse_map,
                t,
            )
        else:
            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx,
                unique_mapped_labels,
                inverse_map,
                t,
            )

    
    def ds_sample(self, xyz, ref, labels, instances):        
        minimum_pts_thre = 300
        instances = instances.squeeze()
        cls, cnt = np.unique(instances, return_counts=True)
        inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]
        unknown_clss = self.unknown_labels
        # if dataset_kitti:
        #     unknown_clss = [5]
        # if dataset_nusc:
        #     unknown_clss = [1, 5, 8, 9]
        for instance_idx in inst_basic_idx:
            rnd = np.random.rand()
            if rnd > 0.5 or labels[instances == instance_idx][0] in unknown_clss:
                continue

            obj_ins = xyz[instances==instance_idx]
            obj_ins_center = np.mean(obj_ins, axis=0)
            obj_ins = obj_ins - obj_ins_center
            scale_ds_large = np.random.rand()*1.5+1.5
            scale_ds_small = np.random.rand()*0.25+0.25
            rnd = np.random.rand()
            scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
            obj_ins = obj_ins * scale_ds + obj_ins_center
            xyz[instances == instance_idx] = obj_ins
            
            # For Only KITTI DATASET
            labels[instances == instance_idx] = 100
            # if dataset_nusc:
            #     labels[instances == instance_idx] = 17
                
        return xyz, ref, labels, instances

### nuscenes dataset ###
class nuScenesRestrictedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        percentage=100,
        label_mapping=None,
        return_ref=False,
        nusc=None,
    ):
        self.return_ref = return_ref
        
        with open(config_file, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = nuscenesyaml['dataset_path']
        
        if split == 'train':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_train.pkl')
        elif split == 'valid':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_val.pkl')
        elif split == 'test':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_test.pkl')
        
        with open(self.imageset, 'rb') as f:
            data = pickle.load(f)

        # if split == 'train' or split == 'valid':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        # elif split == 'test':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-test', dataroot=self.data_path, verbose=True)

        self.nusc_infos = data['infos']
        self.nusc = nusc
        
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )
            
        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path']
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.imageset.find('test') != -1:
            raise NotImplementedError
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        coordinates = points[:, :3]
        # features = points[:, 3][:, None]/255 # 0~255
        features = points[:, 3][:, None] # 0~255
        
        # features = np.ones((coordinates.shape[0], 1))
        
        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])
        
        labels = points_label.squeeze(1)
        
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        
        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )
        
        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        
        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            index,
        )

class nuScenesRestrictedDisjointDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        config_file="config/dataset.yaml",
        data_idx_file='selected_idx.npy',
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        return_ref=False,
        nusc=None,
        labeled=True,
        store_unlabeled_label = True,
        resize_aug = False,
    ):
        self.return_ref = return_ref
        
        with open(config_file, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = nuscenesyaml['dataset_path']
        
        if split == 'train':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_train.pkl')
        elif split == 'valid':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_val.pkl')
        elif split == 'test':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_test.pkl')
        
        with open(self.imageset, 'rb') as f:
            data = pickle.load(f)

        # if split == 'train' or split == 'valid':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        # elif split == 'test':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-test', dataroot=self.data_path, verbose=True)

        nusc_infos = data['infos']
        self.nusc = nusc

        # for constructing disjoint dataset
        if not os.path.isfile(data_idx_file):
            raise RuntimeError("Data index file does not exist: " + data_idx_file)
        selected_data_idx = np.load(data_idx_file)
        real_data_idx = np.arange(len(nusc_infos))
        not_selected_data_idx = [x for x in real_data_idx if x not in selected_data_idx]
        
        self.nusc_infos = []
        if labeled:
            for i_idx in selected_data_idx.tolist():
                self.nusc_infos.append(nusc_infos[i_idx])
        else:
            for i_idx in not_selected_data_idx:
                self.nusc_infos.append(nusc_infos[i_idx])
            
        
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None

            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )
            
        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.imageset.find('test') != -1:
            raise NotImplementedError
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

        coordinates = points[:, :3]
        # features = points[:, 3][:,None]/255 # 0~255
        features = points[:, 3][:, None]

        # features = np.reshape(remissions, (remissions.shape[0],1))
        
        # AUGMENTATION
        if self.augment:
            # DOWNSAMPLING
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]

            # TRANSFORMATIONS
            
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])
        
        labels = points_label.squeeze(1)
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
        
        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )
        
        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        
        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            index,
        )    


class nuScenesRestrictedFineTuningDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        config_file="config/dataset.yaml",
        data_idx_file='selected_idx.npy',
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        return_ref=False,
        nusc=None,
        labeled=True,
        store_unlabeled_label = True,
        resize_aug = True,
        
    ):
        self.return_ref = return_ref
        
        with open(config_file, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = nuscenesyaml['dataset_path']
        
        unknown_labels = re.findall(r'\d+', data_idx_file)
        self.unknown_labels = [int(num) for num in unknown_labels]
        self.unknown_labels = self.unknown_labels[:-1]
        
        if split == 'train':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_train.pkl')
        elif split == 'valid':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_val.pkl')
        elif split == 'test':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_test.pkl')
        
        with open(self.imageset, 'rb') as f:
            data = pickle.load(f)

        # if split == 'train' or split == 'valid':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        # elif split == 'test':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-test', dataroot=self.data_path, verbose=True)

        nusc_infos = data['infos']
        self.nusc = nusc
        
        # for constructing disjoint dataset
        if not os.path.isfile(data_idx_file):
            raise RuntimeError("Data index file does not exist: " + data_idx_file)
        
        selected_data_idx = np.load(data_idx_file)
        real_data_idx = np.arange(len(nusc_infos))
        not_selected_data_idx = [x for x in real_data_idx if x not in selected_data_idx]
        
        self.nusc_infos = []
        if labeled:
            for i_idx in selected_data_idx.tolist():
                self.nusc_infos.append(nusc_infos[i_idx])
        else:
            for i_idx in not_selected_data_idx:
                self.nusc_infos.append(nusc_infos[i_idx])
            
        
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        
        
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None
            self.resize_aug = resize_aug
            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )
        
        
        if label_mapping is not None:
            if self.resize_aug:
                label_mapping[100] = (16 - len(self.unknown_labels))
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.imageset.find('test') != -1:
            raise NotImplementedError
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', lidar_sd_token)['filename'])
            # lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
            #                                         self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            
            points_label = np.load(lidarseg_labels_filename)['data'].reshape([-1, 1])
            sem_label = (points_label // 1000).astype(np.uint8)
            inst_label = (points_label % 1000).astype(np.uint8)
            sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        coordinates = points[:, :3]
        # features = points[:, 3][:,None]/255 # 0~255
        features = points[:, 3][:, None]
        
        # features = remissions.reshape(-1,1)
        
        # NOTE: AUGMENTATION (Only in Train Mode)
        if self.augment: 
            # DOWNSAMPLING
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]
            sem_label = sem_label[selected_idx]
            inst_label = inst_label[selected_idx]

        labels = sem_label.squeeze(1)
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            inst_label = inst_label[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            
            if self.resize_aug:
                coordinates, features, labels, inst_label = self.ds_sample(coordinates, features, labels, inst_label)
            
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
            

        
        if self.augment:
            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations
        
            homo_coords = np.hstack(
                (
                    coordinates,
                    np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                )
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])
        
        

        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coordinates,
            return_index=True,
            return_inverse=True,
            quantization_size=self.voxel_size,
        )
        
        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx = selected_idx[unique_map]
        
        return (
            discrete_coords,
            unique_feats,
            unique_labels,
            selected_idx,
            unique_mapped_labels,
            inverse_map,
            index,
        )      
        
    def ds_sample(self, xyz, ref, labels, instances):        
        minimum_pts_thre = 300
        instances = instances.squeeze()
        cls, cnt = np.unique(instances, return_counts=True)
        inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]
        unknown_clss = self.unknown_labels
        # if dataset_kitti:
        #     unknown_clss = [5]
        # if dataset_nusc:
        #     unknown_clss = [1, 5, 8, 9]
        for instance_idx in inst_basic_idx:
            rnd = np.random.rand()
            if rnd > 0.5 or labels[instances == instance_idx][0] in unknown_clss:
                continue

            obj_ins = xyz[instances==instance_idx]
            obj_ins_center = np.mean(obj_ins, axis=0)
            obj_ins = obj_ins - obj_ins_center
            scale_ds_large = np.random.rand()*1.5+1.5
            scale_ds_small = np.random.rand()*0.25+0.25
            rnd = np.random.rand()
            scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
            obj_ins = obj_ins * scale_ds + obj_ins_center
            xyz[instances == instance_idx] = obj_ins
            labels[instances == instance_idx] =  100
            # if dataset_nusc:
            # labels[instances == instance_idx] = 17
                
        return xyz, ref, labels, instances


class nuScenes_LaserMix_RealAug_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        config_file="config/dataset.yaml",
        data_idx_file='selected_idx.npy',
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        return_ref=False,
        nusc=None,
        labeled=True,
        store_unlabeled_label = True,
        resize_aug = True,
        output_both_voxel_point=False,
        resize_only_voxel=False,
    ):
        self.return_ref = return_ref
        
        with open(config_file, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.data_path = nuscenesyaml['dataset_path']
        
        unknown_labels = re.findall(r'\d+', data_idx_file)
        self.unknown_labels = [int(num) for num in unknown_labels]
        self.unknown_labels = self.unknown_labels[:-1]
        
        if split == 'train':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_train.pkl')
        elif split == 'valid':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_val.pkl')
        elif split == 'test':
            self.imageset = os.path.join(self.data_path, 'nuscenes_infos_test.pkl')
        
        with open(self.imageset, 'rb') as f:
            data = pickle.load(f)

        # if split == 'train' or split == 'valid':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        # elif split == 'test':
        #     from nuscenes import NuScenes
        #     nusc = NuScenes(version='v1.0-test', dataroot=self.data_path, verbose=True)

        nusc_infos = data['infos']
        self.nusc = nusc
        
        # for constructing disjoint dataset
        if not os.path.isfile(data_idx_file):
            raise RuntimeError("Data index file does not exist: " + data_idx_file)
        
        selected_data_idx = np.load(data_idx_file)
        real_data_idx = np.arange(len(nusc_infos))
        not_selected_data_idx = [x for x in real_data_idx if x not in selected_data_idx]
        
        self.nusc_infos = []
        if labeled:
            for i_idx in selected_data_idx.tolist():
                self.nusc_infos.append(nusc_infos[i_idx])
        else:
            for i_idx in not_selected_data_idx:
                self.nusc_infos.append(nusc_infos[i_idx])
            
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        
        
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = (0.95, 1.05)
            self.rotation_augmentation_bound = (
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
                (-np.pi / 20, np.pi / 20),
            )
            self.translation_augmentation_ratio_bound = None
            self.resize_aug = resize_aug
            self.voxelizer = Voxelizer(
                voxel_size=self.voxel_size,
                clip_bound=self.clip_bounds,
                use_augmentation=self.augment,
                scale_augmentation_bound=self.scale_augmentation_bound,
                rotation_augmentation_bound=self.rotation_augmentation_bound,
                translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
                ignore_label=-1,
            )
        
        
        if label_mapping is not None:
            if self.resize_aug:
                label_mapping[100] = (16 - len(self.unknown_labels))
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None
        
        self.output_both_voxel_point = output_both_voxel_point
        self.resize_only_voxel = resize_only_voxel

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path']
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.imageset.find('test') != -1:
            raise NotImplementedError
            points_label = np.expand_dims(np.zeros_like(points[:, 0], dtype=int), axis=1)
        else:
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('panoptic', lidar_sd_token)['filename'])
            # lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
            #                                         self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            
            points_label = np.load(lidarseg_labels_filename)['data'].reshape([-1, 1])
            sem_label = (points_label // 1000).astype(np.uint8)
            inst_label = (points_label % 1000).astype(np.uint8)
            sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        coordinates = points[:, :3]
        # features = points[:, 3][:,None]/255 # 0~255
        features = points[:, 3][:, None]
        
        # features = remissions.reshape(-1,1)
        
        # NOTE: AUGMENTATION (Only in Train Mode)
        if self.augment: 
            # DOWNSAMPLING
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]
            sem_label = sem_label[selected_idx]
            inst_label = inst_label[selected_idx]

        labels = sem_label.squeeze(1)
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            inst_label = inst_label[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            
            if self.resize_aug:
                if self.resize_only_voxel:
                    resized_coords, resized_feats, resized_labels, resized_inst_label = self.ds_sample(coordinates, features, labels, inst_label)
                else:
                    coordinates, features, labels, inst_label = self.ds_sample(coordinates, features, labels, inst_label)
            
            if self.label_mapping_function is not None:
                if self.resize_only_voxel:
                    resized_mapped_labels = self.label_mapping_function(resized_labels)
                    mapped_labels = self.label_mapping_function(labels)
                else:
                    mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
            

        
        if self.augment:
            # TRANSFORMATIONS
            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations
            if self.resize_only_voxel:
                resized_homo_coords = np.hstack((
                    resized_coords,
                    np.ones((resized_coords.shape[0], 1), dtype=coordinates.dtype),
                    )
                )
                resized_coords = resized_homo_coords @ rigid_transformation.T[:, :3]
                
                homo_coords = np.hstack(
                    (
                        coordinates,
                        np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                    )
                )
                # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
                coordinates = homo_coords @ rigid_transformation.T[:, :3]
            else:
                homo_coords = np.hstack(
                    (
                        coordinates,
                        np.ones((coordinates.shape[0], 1), dtype=coordinates.dtype),
                    )
                )
                coordinates = homo_coords @ rigid_transformation.T[:, :3]
        else:
            selected_idx = np.arange(coordinates.shape[0])
        
        
        if self.output_both_voxel_point:
            if self.resize_only_voxel:
                discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
                    coordinates=resized_coords,
                    return_index=True,
                    return_inverse=True,
                    quantization_size=self.voxel_size,
                )
                
                unique_feats = resized_feats[unique_map]
                unique_labels = resized_labels[unique_map]
                unique_mapped_labels = resized_mapped_labels[unique_map]
                selected_idx = selected_idx[unique_map]
                
                return (
                    coordinates,
                    features,
                    labels,
                    selected_idx,
                    mapped_labels,
                    discrete_coords,
                    unique_feats,
                    unique_labels,
                    selected_idx,
                    unique_mapped_labels,
                    inverse_map,
                    index,
                )   
            else:
                discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
                    coordinates=coordinates,   
                    return_index=True,
                    return_inverse=True,
                    quantization_size=self.voxel_size,
                )
                unique_feats = features[unique_map]
                unique_labels = labels[unique_map]
                unique_mapped_labels = mapped_labels[unique_map]
                selected_idx = selected_idx[unique_map]
                
                return (
                    coordinates,
                    features,
                    labels,
                    selected_idx,
                    mapped_labels,
                    discrete_coords,
                    unique_feats,
                    unique_labels,
                    selected_idx,
                    unique_mapped_labels,
                    inverse_map,
                    index,
                )
        
    def ds_sample(self, xyz, ref, labels, instances):        
        minimum_pts_thre = 300
        instances = instances.squeeze()
        cls, cnt = np.unique(instances, return_counts=True)
        inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]
        unknown_clss = self.unknown_labels
        # if dataset_kitti:
        #     unknown_clss = [5]
        # if dataset_nusc:
        #     unknown_clss = [1, 5, 8, 9]
        for instance_idx in inst_basic_idx:
            rnd = np.random.rand()
            if rnd > 0.5 or labels[instances == instance_idx][0] in unknown_clss:
                continue

            obj_ins = xyz[instances==instance_idx]
            obj_ins_center = np.mean(obj_ins, axis=0)
            obj_ins = obj_ins - obj_ins_center
            scale_ds_large = np.random.rand()*1.5+1.5
            scale_ds_small = np.random.rand()*0.25+0.25
            rnd = np.random.rand()
            scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
            obj_ins = obj_ins * scale_ds + obj_ins_center
            xyz[instances == instance_idx] = obj_ins
            labels[instances == instance_idx] =  100
            # if dataset_nusc:
            # labels[instances == instance_idx] = 17
                
        return xyz, ref, labels, instances
# NOPS SwaV 
class dataset_wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, t):
        data = self.dataset.__getitem__(t)
        if isinstance(data, tuple):
            to_ret = self.dataset.__getitem__(t)[:-1] + self.dataset.__getitem__(t)
            return to_ret
        
        elif isinstance(data, dict):
            to_ret = (self.dataset.__getitem__(t), self.dataset.__getitem__(t))
            return to_ret

class swav_wrapper_ssl(torch.utils.data.Dataset):
    def __init__(self, lab_dataset, unlab_dataset):
        super().__init__()
        self.lab_dataset = lab_dataset
        self.unlab_dataset = unlab_dataset
        self.len_lab = len(self.lab_dataset)
        self.len_unlab = len(self.unlab_dataset)

    def __len__(self):
        return max(self.len_lab, self.len_unlab)
    
    def __getitem__(self,t):
        if t < self.len_lab:
            lab_data = self.lab_dataset.__getitem__(t)[:-1] + self.lab_dataset.__getitem__(t)
        else: 
            idx = int(np.random.choice(self.len_lab, 1, replace=False))
            lab_data = self.lab_dataset.__getitem__(idx)[:-1] + self.lab_dataset.__getitem__(idx)
        
        if t < self.len_unlab:
            unlab_data = self.unlab_dataset.__getitem__(t)[:-1] + self.unlab_dataset.__getitem__(t)
        else:
            idx = int(np.random.choice(self.len_unlab, 1, replace =False))
            unlab_data = self.unlab_dataset.__getitem__(idx)[:-1] + self.unlab_dataset.__getitem__(idx)
            
            
        to_ret = lab_data + unlab_data
            
        # try:
        #     to_ret = self.lab_dataset.__getitem__(t)[:-1] + self.lab_dataset.__getitem__(t) +  \
        #         self.unlab_dataset.__getitem__(t)[:-1] + self.unlab_dataset.__getitem__(t)
        # except:
        #     breakpoint()
        
                
        return to_ret
    

