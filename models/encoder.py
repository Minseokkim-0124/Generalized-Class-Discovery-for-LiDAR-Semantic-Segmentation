import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch import Tensor

import numpy as np

from numbers import Number
from mmdet3d.utils import OptConfigType
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet.models import DetDataPreprocessor
import spconv.pytorch as spconv
from typing import Any, List, Optional, Tuple, Union, Sequence, Dict
from mmengine.utils import is_seq_of
from mmcv.utils import ext_loader
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter



class SegVFE(nn.Module):
    """Voxel feature encoder used in segmentation task.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 6.
        feat_channels (list(int)): Channels of features in VFE.
        with_voxel_center (bool): Whether to use the distance
            to center of voxel for each points inside a voxel.
            Defaults to False.
        voxel_size (tuple[float]): Size of a single voxel (rho, phi, z).
            Defaults to None.
        grid_shape (tuple[float]): The grid shape of voxelization.
            Defaults to (480, 360, 32).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Defaults to (0, -3.14159265359, -4, 50, 3.14159265359, 2).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points
            inside a voxel. Available options include 'max' and 'avg'.
            Defaults to 'max'.
        with_pre_norm (bool): Whether to use the norm layer before
            input vfe layer.
        feat_compression (int, optional): The voxel feature compression
            channels, Defaults to None
        return_point_feats (bool): Whether to return the features
            of each points. Defaults to False.
    """

    def __init__(self,
                 in_channels: int = 6,
                 feat_channels: Sequence[int] = [],
                 with_voxel_center: bool = False,
                 voxel_size: Optional[Sequence[float]] = None,
                 grid_shape: Sequence[float] = (480, 360, 32),
                 point_cloud_range: Sequence[float] = (0, -3.14159265359, -4,
                                                       50, 3.14159265359, 2),
                 norm_cfg: dict = dict(type='BN1d', eps=1e-5, momentum=0.1),
                 mode: bool = 'max',
                 with_pre_norm: bool = True,
                 feat_compression: Optional[int] = None,
                 return_point_feats: bool = False) -> None:
        super(SegVFE, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        assert not (voxel_size and grid_shape), \
            'voxel_size and grid_shape cannot be setting at the same time'
        if with_voxel_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        self.point_cloud_range = point_cloud_range
        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] -
                          point_cloud_range[:3] ) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (
                grid_shape - 1)
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = self.voxel_size[0]
        self.vy = self.voxel_size[1]
        self.vz = self.voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                vfe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                vfe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters), norm_layer,
                        nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.vfe_scatter = DynamicScatter(self.voxel_size,
                                          self.point_cloud_range,
                                          (mode != 'max'))
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression), nn.ReLU())

    def forward(self, features: Tensor, coors: Tensor, *args,
                **kwargs) -> Tuple[Tensor]:
        """Forward functions.

        Args:
            features (Tensor): Features of voxels, shape is NxC.
            coors (Tensor): Coordinates of voxels, shape is  Nx(1+NDim).

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels additionally.
        """
        features_ls = [features]

        # Find distance of x, y, and z from voxel center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 1].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 3].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        # Combine together feature decorations
        features = torch.cat(features_ls[::-1], dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = []
        for vfe in self.vfe_layers:
            features = vfe(features)
            point_feats.append(features)
        voxel_feats, voxel_coors = self.vfe_scatter(features, coors)

        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        if self.return_point_feats:
            return voxel_feats, voxel_coors, point_feats
        return voxel_feats, voxel_coors