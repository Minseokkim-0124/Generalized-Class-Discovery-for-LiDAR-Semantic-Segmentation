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
# from mmengine.dataset.base_dataset import force_full_init

def get_dataset(name, type='disjoint'):
    if name == "SemanticKITTI":
        if type == 'labeled':
            return SemanticKITTIRestrictedDatasetLabel
        elif type == 'unlabeled':
            return SemanticKITTIRestrictedDatasetUnlabel
        elif type == 'disjoint':
            return SemanticKITTIRestrictedDisjointDataset
        elif type == 'finetuning':
            return SemanticKITTIRestrictedFineTuningDataset
        elif type == 'gcd':
            return SemanticKITTIGCDDataset
        else:
            raise NameError(f'Nuscenes Dataset "{type}" is not yet implemented')
    elif name == "nuScenes":
        if type == 'labeled':
            return nuScenesRestrictedDataset
        elif type == 'disjoint':
            return nuScenesRestrictedDisjointDataset
        elif type =='finetuning': 
            return nuScenesRestrictedFineTuningDataset
        elif type == 'gcd':
            return nuScenesGCDDataset
    else:
        raise NameError(f'Dataset "{name}" is not yet implemented')


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
        use_all_feats=False,
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
        self.use_all_feats = use_all_feats

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
        
        if self.use_all_feats:
            features = deepcopy(scan)
        else:
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
    
class SemanticKITTIRestrictedDatasetUnlabel(torch.utils.data.Dataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        split="train",
        voxel_size=0.05,
        augment=False,
        percentage = 1,
        label_mapping=None,
        output_voxelize = True,
    ):
        """Load data from given dataset directory."""

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.files = {"input": []}
        self.filenames = []
        
        if split == 'train':
            self.ann_file = 'semantickitti_infos_train.{}-unlabeled.pkl'.format(percentage)
        elif split == 'valid':
            self.ann_file = 'semantickitti_infos_val.pkl'
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

        self.output_voxelize = output_voxelize

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
        
        if self.output_voxelize:
            discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
                coordinates=coordinates,
                return_index=True,
                return_inverse=True,
                quantization_size=self.voxel_size,
            )

            unique_feats = features[unique_map]
            selected_idx = selected_idx[unique_map]

            return (
                discrete_coords,
                unique_feats,
                selected_idx,
                inverse_map,
                t,
            )
        else:
            inverse_map = None 
            return (
                coordinates,
                features,
                selected_idx,
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

class SemanticKITTIRestrictedFineTuningDataset(torch.utils.data.Dataset):
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
        self.resize_aug = resize_aug
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
        else:
            selected_idx = np.arange(coordinates.shape[0])
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
                coordinates, features, labels, instance_data = self.ds_sample(coordinates, features, labels, instance_data)
            
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
        
class SemanticKITTIGCDDataset(SemanticKITTIRestrictedDisjointDataset):
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
        con_aug=False,
        no_aug_downsample = False,
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
            self.scale_augmentation_bound = None
            # self.scale_augmentation_bound = (0.95, 1.05)
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
        self.con_aug = con_aug
        self.no_aug_ds = no_aug_downsample

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        # remissions = scan[:, 3]  # get remission
        # feats = deepcopy(scan)
        # pos_features = feats[:, :3]
        intensity = scan[:, 3]
        intensity = intensity.reshape(-1, 1)

        if self.no_aug_ds:
            selected_idx = np.random.choice(
                    coordinates.shape[0], self.downsampling, replace=False
                )
            selected_idx = np.sort(selected_idx)
            
            coordinates = coordinates[selected_idx]
            intensity = intensity[selected_idx]
        
        else:            
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
                # pos_features = pos_features[selected_idx]
                intensity = intensity[selected_idx]

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
                # homo_feats = np.hstack(
                #     (
                #         pos_features,
                #         np.ones((pos_features.shape[0], 1), dtype=coordinates.dtype),
                #     )
                # )
                coordinates = homo_coords @ rigid_transformation.T[:, :3]

                if torch.randint(0, 2, size=(1,)).item():
                    coordinates[:, 0] = -coordinates[:, 0]
                # pos_features = homo_feats @ rigid_transformation.T[:, :3]
                # feats = np.hstack(
                #     (
                #         pos_features,
                #         intensity,
                #     )
                # )
                if self.con_aug:
                    voxel_mtx1, affine_mtx1 = self.voxelizer.get_transformation_matrix()

                    rigid_transformation1 = affine_mtx1 @ voxel_mtx1
                    
                    coordinates1 = homo_coords @ rigid_transformation1.T[:, :3]
                    # pos_features1 = homo_feats @ rigid_transformation1.T[:, :3]

                #     feats1 = np.hstack(
                #     (
                #         pos_features1,
                #         intensity,
                #     )
                # ) 
                    if torch.randint(0, 2, size=(1,)).item():
                        coordinates1[:, 0] = -coordinates1[:, 0]
            else:
                selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            
            if self.no_aug_ds or self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            intensity = intensity[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.con_aug:
                coordinates1 = coordinates1[labelled_idx]
                # intensity = intensity[labelled_idx]
            
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
        pc_ = np.round(coordinates / self.voxel_size).astype(np.int32)
        # pos = (torch.tensor(coordinates)/self.voxel_size).long()
        pc_ -= pc_.min(0, keepdims=1)
        
        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=pc_,
            return_index=True,
            return_inverse=True,
        )
        coordinates = coordinates[unique_map]
        unique_intensity = intensity[unique_map]
        unique_feats = np.hstack(
                    (
                        coordinates,
                        unique_intensity,
                    )
                )
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx_ = selected_idx[unique_map]

        if self.con_aug:
            pc_1 = np.round(coordinates1 / self.voxel_size).astype(np.int32)
            # pos_1 = (torch.tensor(coordinates1)/self.voxel_size).long()
            pc_1 -= pc_1.min(0, keepdims=1)
            
            discrete_coords1, unique_map1, inverse_map1 = ME.utils.sparse_quantize(
                coordinates=pc_1,
                return_index=True,
                return_inverse=True,
            )
            coordinates1 = coordinates1[unique_map1]
            unique_intensity1 = intensity[unique_map1]
            unique_feats1 = np.hstack(
                    (
                        coordinates1,
                        unique_intensity1,
                    )
                )
            unique_labels1 = labels[unique_map1]
            unique_mapped_labels1 = mapped_labels[unique_map1]
            selected_idx1 = selected_idx[unique_map1]
            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx_,
                unique_mapped_labels,
                inverse_map,
                discrete_coords1,
                unique_feats1,
                unique_labels1,
                selected_idx1,
                unique_mapped_labels1,
                inverse_map1,
                t,
            )
        else:
            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx_,
                unique_mapped_labels,
                inverse_map,
                t,
            )
class SemanticKITTIGCDToyDataset(SemanticKITTIRestrictedDisjointDataset):
    def __init__(
        self,
        config_file="config/dataset.yaml",
        percentage = 100,
        split="train",
        voxel_size=0.05,
        downsampling=80000,
        augment=False,
        label_mapping=None,
        labeled=True,
        store_unlabeled_label = False,
        nusc=None,
        con_aug=False,
        no_aug_downsample = False,
    ):
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
            # self.scale_augmentation_bound = (0.95, 1.05)
            self.scale_augmentation_bound = None
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
            
        self.num_files = len(self.files['input'])
        self.con_aug = con_aug
        self.no_aug_ds = no_aug_downsample

    def __getitem__(self, t):
        pc_filename = self.files["input"][t]
        scan = np.fromfile(pc_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        coordinates = scan[:, 0:3]  # get xyz
        # remissions = scan[:, 3]  # get remission
        # feats = deepcopy(scan)
        # pos_features = feats[:, :3]
        intensity = scan[:, 3]
        intensity = intensity.reshape(-1, 1)

        if self.no_aug_ds:
            selected_idx = np.random.choice(
                    coordinates.shape[0], self.downsampling, replace=False
                )
            selected_idx = np.sort(selected_idx)
            
            coordinates = coordinates[selected_idx]
            intensity = intensity[selected_idx]
        
        else:            
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
                # pos_features = pos_features[selected_idx]
                intensity = intensity[selected_idx]

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
                # homo_feats = np.hstack(
                #     (
                #         pos_features,
                #         np.ones((pos_features.shape[0], 1), dtype=coordinates.dtype),
                #     )
                # )
                coordinates = homo_coords @ rigid_transformation.T[:, :3]

                if torch.randint(0, 2, size=(1,)).item():
                    coordinates[:, 0] = -coordinates[:, 0]
                # pos_features = homo_feats @ rigid_transformation.T[:, :3]
                # feats = np.hstack(
                #     (
                #         pos_features,
                #         intensity,
                #     )
                # )
                if self.con_aug:
                    voxel_mtx1, affine_mtx1 = self.voxelizer.get_transformation_matrix()

                    rigid_transformation1 = affine_mtx1 @ voxel_mtx1
                    
                    coordinates1 = homo_coords @ rigid_transformation1.T[:, :3]
                    # pos_features1 = homo_feats @ rigid_transformation1.T[:, :3]

                #     feats1 = np.hstack(
                #     (
                #         pos_features1,
                #         intensity,
                #     )
                # ) 
                    if torch.randint(0, 2, size=(1,)).item():
                        coordinates1[:, 0] = -coordinates1[:, 0]
            else:
                selected_idx = np.arange(coordinates.shape[0])

        if "label" in self.files.keys():
            label_filename = self.files["label"][t]
            labels = np.fromfile(label_filename, dtype=np.int32)
            labels = labels.reshape((-1))
            labels = labels & 0xFFFF
            
            if self.no_aug_ds or self.augment:
                labels = labels[selected_idx]
            for index, element in enumerate(labels):
                labels[index] = self.config["learning_map"].get(element, -1)
        else:
            labels = np.negative(np.ones(coordinates.shape[0]))

        # REMOVE UNLABELED POINTS IF NOT IN TESTING
        if "label" in self.files.keys():
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            intensity = intensity[labelled_idx]
            labels = labels[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            if self.con_aug:
                coordinates1 = coordinates1[labelled_idx]
                # intensity = intensity[labelled_idx]
            
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
        pc_ = np.round(coordinates / self.voxel_size).astype(np.int32)
        # pos = (torch.tensor(coordinates)/self.voxel_size).long()
        pc_ -= pc_.min(0, keepdims=1)
        
        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=pc_,
            return_index=True,
            return_inverse=True,
        )
        coordinates = coordinates[unique_map]
        unique_intensity = intensity[unique_map]
        unique_feats = np.hstack(
                    (
                        coordinates,
                        unique_intensity,
                    )
                )
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx_ = selected_idx[unique_map]

        if self.con_aug:
            pc_1 = np.round(coordinates1 / self.voxel_size).astype(np.int32)
            # pos_1 = (torch.tensor(coordinates1)/self.voxel_size).long()
            pc_1 -= pc_1.min(0, keepdims=1)
            
            discrete_coords1, unique_map1, inverse_map1 = ME.utils.sparse_quantize(
                coordinates=pc_1,
                return_index=True,
                return_inverse=True,
            )
            coordinates1 = coordinates1[unique_map1]
            unique_intensity1 = intensity[unique_map1]
            unique_feats1 = np.hstack(
                    (
                        coordinates1,
                        unique_intensity1,
                    )
                )
            unique_labels1 = labels[unique_map1]
            unique_mapped_labels1 = mapped_labels[unique_map1]
            selected_idx1 = selected_idx[unique_map1]
            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx_,
                unique_mapped_labels,
                inverse_map,
                discrete_coords1,
                unique_feats1,
                unique_labels1,
                selected_idx1,
                unique_mapped_labels1,
                inverse_map1,
                t,
            )
        else:
            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx_,
                unique_mapped_labels,
                inverse_map,
                t,
            )
        
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
        use_all_feats=False,
        remission_not_divide=True,
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
        self.remission_not_divide = remission_not_divide
        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

        self.use_all_feats = use_all_feats
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
        if self.use_all_feats:
            features = deepcopy(points[:,:4])
        else: 
            if self.remission_not_divide:
                features = points[:, 3][:, None] # 0~255
            else:
                features = points[:, 3][:, None]/255
        
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
            
        if label_mapping is not None:
            self.label_mapping_function = np.vectorize(lambda x: label_mapping[x])
        else:
            self.label_mapping_function = None

    def __len__(self):
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path']
        try:
            points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        except:
            breakpoint()
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
        remission_not_divide=True,
        
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
        self.remission_not_divide = remission_not_divide
        self.resize_aug = resize_aug
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
        if self.remission_not_divide:
            features = points[:, 3][:, None]
        else: 
            features = points[:, 3][:, None]/255
        
        # features = remissions.reshape(-1,1)
        
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
            sem_label = sem_label[selected_idx]
            inst_label = inst_label[selected_idx]
        else:
            selected_idx = np.arange(coordinates.shape[0])
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

class nuScenesGCDDataset(nuScenesRestrictedDisjointDataset):
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
        con_aug=False,
        no_aug_downsample = False,
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
            
        self.labeled = labeled
        self.voxel_size = voxel_size
        self.downsampling = downsampling
        self.augment = False
        if split == "train" and augment:
            self.augment = True
            self.clip_bounds = None
            self.scale_augmentation_bound = None
            # self.scale_augmentation_bound = (0.95, 1.05)
            
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
        self.con_aug = con_aug
        self.no_aug_ds = no_aug_downsample
    
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
        # NOTE: 여기서 부터 수정해야할듯?
        features = points[:, 3][:, None]
        
        if self.no_aug_ds:
            downsampling = min(self.downsampling, coordinates.shape[0])
            selected_idx = np.random.choice(
                coordinates.shape[0], downsampling, replace=False
            )
            selected_idx = np.sort(selected_idx)
            coordinates = coordinates[selected_idx]
            features = features[selected_idx]
            sem_label = sem_label[selected_idx]
            inst_label = inst_label[selected_idx]
        else:
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
                sem_label = sem_label[selected_idx]
                inst_label = inst_label[selected_idx]
            else:
                selected_idx = np.arange(coordinates.shape[0])
        labels = sem_label.squeeze(1)
        if self.imageset.find('test') == -1:
            labelled_idx = labels != -1
            coordinates = coordinates[labelled_idx]
            features = features[labelled_idx]
            labels = labels[labelled_idx]
            inst_label = inst_label[labelled_idx]
            selected_idx = selected_idx[labelled_idx]
            
            
            if self.label_mapping_function is not None:
                mapped_labels = self.label_mapping_function(labels)
            else:
                mapped_labels = np.copy(labels)
            

        if self.no_aug_ds:
            pass
    
        else:
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
                # homo_feats = np.hstack(
                #     (
                #         pos_features,
                #         np.ones((pos_features.shape[0], 1), dtype=coordinates.dtype),
                #     )
                # )
                # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
                coordinates = homo_coords @ rigid_transformation.T[:, :3]
                # pos_features = homo_feats @ rigid_transformation.T[:,:3]
                # aug_feats = np.hstack(
                #     (
                #         pos_features,
                #         intensity,
                #     )
                # )
                if torch.randint(0, 2, size=(1,)).item():
                    coordinates[:, 0] = -coordinates[:, 0]
                if self.con_aug:
                    voxel_mtx1, affine_mtx1 = self.voxelizer.get_transformation_matrix()
                    rigid_transformation1 = affine_mtx1 @ voxel_mtx1
                    coordinates1 = homo_coords @ rigid_transformation1.T[:, :3]
                    # pos_features1 = homo_feats @ rigid_transformation1.T[:, :3]
                    # aug_feats1 = np.hstack(
                    # (
                    #     pos_features1,
                    #     intensity,
                    # )
                    # )
                    if torch.randint(0, 2, size=(1,)).item():
                        coordinates1[:, 0] = -coordinates1[:, 0]
            else:
                selected_idx = np.arange(coordinates.shape[0])
        
        pc_ = np.round(coordinates / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        discrete_coords, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=pc_,
            return_index=True,
            return_inverse=True,
        )
        
        unique_feats = features[unique_map]
        unique_labels = labels[unique_map]
        unique_mapped_labels = mapped_labels[unique_map]
        selected_idx_ = selected_idx[unique_map]
        
        if self.con_aug:
            pc_1 = np.round(coordinates1 / self.voxel_size).astype(np.int32)
            pc_1 -= pc_1.min(0, keepdims=1)

            discrete_coords1, unique_map1, inverse_map1 = ME.utils.sparse_quantize(
            coordinates=pc_1,
            return_index=True,
            return_inverse=True,
            )
        
            unique_feats1 = features[unique_map1]
            unique_labels1 = labels[unique_map1]
            unique_mapped_labels1 = mapped_labels[unique_map1]
            selected_idx1 = selected_idx[unique_map1]

            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx_,
                unique_mapped_labels,
                inverse_map,
                discrete_coords1,
                unique_feats1,
                unique_labels1,
                selected_idx1,
                unique_mapped_labels1,
                inverse_map1,
                index,
            )
        else:
            return (
                discrete_coords,
                unique_feats,
                unique_labels,
                selected_idx_,
                unique_mapped_labels,
                inverse_map,
                index,
            )      


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
    

