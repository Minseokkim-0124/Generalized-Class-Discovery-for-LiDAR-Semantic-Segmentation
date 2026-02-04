# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmcv.ops import SparseConvTensor, SparseModule, SubMConv3d
from abc import ABCMeta, abstractmethod
from typing import Dict, List

from mmengine.model import BaseModule, normal_init
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig\

class Base3DDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.5.
        conv_cfg (dict or :obj:`ConfigDict`): Config of conv layers.
            Defaults to dict(type='Conv1d').
        norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers.
            Defaults to dict(type='BN1d').
        act_cfg (dict or :obj:`ConfigDict`): Config of activation layers.
            Defaults to dict(type='ReLU').
        loss_decode (dict or :obj:`ConfigDict`): Config of decode loss.
            Defaults to dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            class_weight=None, loss_weight=1.0).
        conv_seg_kernel_size (int): The kernel size used in conv_seg.
            Defaults to 1.
        ignore_index (int): The label index to be ignored. When using masked
            BCE loss, ignore_index should be set to None. Defaults to 255.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0.5,
                 conv_cfg: dict = dict(type='Conv1d'),
                 norm_cfg: dict = dict(type='BN1d'),
                 act_cfg: dict = dict(type='ReLU'),
                 loss_decode: dict = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 conv_seg_kernel_size: int = 1,
                 ignore_index: int = 255,
                 init_cfg=None):
        super(Base3DDecodeHead, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_decode = MODELS.build(loss_decode)
        self.ignore_index = ignore_index

        self.conv_seg = self.build_conv_seg(
            channels=channels,
            num_classes=num_classes,
            kernel_size=conv_seg_kernel_size)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

    def init_weights(self) -> None:
        """Initialize weights of classification layer."""
        super().init_weights()
        normal_init(self.conv_seg, mean=0, std=0.01)

    @abstractmethod
    def forward(self, feats_dict: dict) -> Tensor:
        """Placeholder of forward function."""
        pass

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Conv1d(channels, num_classes, kernel_size=kernel_size)

    def cls_seg(self, feat: Tensor) -> Tensor:
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def loss(self, inputs: dict, batch_data_samples,
             train_cfg) -> Dict[str, Tensor]:
        """Forward function for training.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            train_cfg (dict or :obj:`ConfigDict`): The training config.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: dict, batch_input_metas: List[dict],
                test_cfg) -> Tensor:
        """Forward function for testing.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            test_cfg (dict or :obj:`ConfigDict`): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs)

        return seg_logits

    def _stack_batch_gt(self, batch_data_samples) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logit: Tensor,
                     batch_data_samples) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (Tensor): Predicted per-point segmentation logits of
                shape [B, num_classes, N].
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss


class Cylinder3DHead(Base3DDecodeHead):
    """Cylinder3D decoder head.

    Decoder head used in `Cylinder3D <https://arxiv.org/abs/2011.10033>`_.
    Refer to the
    `official code <https://https://github.com/xinge008/Cylinder3D>`_.

    Args:
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.
        conv_cfg (dict or :obj:`ConfigDict`): Config of conv layers.
            Defaults to dict(type='Conv1d').
        norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers.
            Defaults to dict(type='BN1d').
        act_cfg (dict or :obj:`ConfigDict`): Config of activation layers.
            Defaults to dict(type='ReLU').
        loss_ce (dict or :obj:`ConfigDict`): Config of CrossEntropy loss.
            Defaults to dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0).
        loss_lovasz (dict or :obj:`ConfigDict`): Config of Lovasz loss.
            Defaults to dict(type='LovaszLoss', loss_weight=1.0).
        conv_seg_kernel_size (int): The kernel size used in conv_seg.
            Defaults to 3.
        ignore_index (int): The label index to be ignored. When using masked
            BCE loss, ignore_index should be set to None. Defaults to 19.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0,
                 conv_cfg: dict = dict(type='Conv1d'),
                 norm_cfg: dict = dict(type='BN1d'),
                 act_cfg: dict = dict(type='ReLU'),
                 loss_ce: dict = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_lovasz: dict = dict(
                     type='LovaszLoss', loss_weight=1.0),
                 conv_seg_kernel_size: int = 3,
                 ignore_index: int = 19,
                 init_cfg=None) -> None:
        super(Cylinder3DHead, self).__init__(
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            conv_seg_kernel_size=conv_seg_kernel_size,
            init_cfg=init_cfg)

        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.loss_ce = MODELS.build(loss_ce)
        self.ignore_index = ignore_index

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> SparseModule:
        return SubMConv3d(
            channels,
            num_classes,
            indice_key='logit',
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=True)

    def forward(self, sparse_voxels: SparseConvTensor) -> SparseConvTensor:
        """Forward function."""
        sparse_logits = self.cls_seg(sparse_voxels)
        return sparse_logits

    def loss_by_feat(self, seg_logit: SparseConvTensor,
                     batch_data_samples) -> dict:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (SparseConvTensor): Predicted per-voxel
                segmentation logits of shape [num_voxels, num_classes]
                stored in SparseConvTensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = seg_logit.features
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        loss['loss_lovasz'] = self.loss_lovasz(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)

        return loss

    def predict(
        self,
        inputs: SparseConvTensor,
        batch_inputs_dict: dict,
        batch_data_samples,
    ) -> torch.Tensor:
        """Forward function for testing.

        Args:
            inputs (SparseConvTensor): Feature from backbone.
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Dict of voxelized voxels and the corresponding
                coordinates.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. We use `point2voxel_map` in this function.

        Returns:
            List[torch.Tensor]: List of point-wise segmentation logits.
        """
        seg_logits = self.forward(inputs).features

        seg_pred_list = []
        coors = batch_inputs_dict['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            seg_logits_sample = seg_logits[coors[:, 0] == batch_idx]
            point2voxel_map = batch_data_samples[
                batch_idx].point2voxel_map.long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list

class MinkUNetHead(Base3DDecodeHead):
    r"""MinkUNet decoder head with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
    """

    def __init__(self, batch_first: bool = True, **kwargs) -> None:
        super(MinkUNetHead, self).__init__(**kwargs)
        self.batch_first = batch_first

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Linear(channels, num_classes)

    def forward(self, voxel_dict: dict) -> dict:
        """Forward function."""
        logits = self.cls_seg(voxel_dict['voxel_feats'])
        voxel_dict['logits'] = logits
        return voxel_dict

    def loss_by_feat(self, voxel_dict: dict,
                     batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            voxel_dict (dict): The dict may contain `logits`,
                `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        voxel_semantic_segs = []
        voxel_inds = voxel_dict['voxel_inds']
        for batch_idx, data_sample in enumerate(batch_data_samples):
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            voxel_semantic_mask = pts_semantic_mask[voxel_inds[batch_idx]]
            voxel_semantic_segs.append(voxel_semantic_mask)
        seg_label = torch.cat(voxel_semantic_segs)
        seg_logit_feat = voxel_dict['logits']
        loss = dict()
        loss['loss_ce'] = self.loss_decode(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, voxel_dict: dict,
                batch_data_samples: SampleList) -> List[Tensor]:
        """Forward function for testing.

        Args:
            voxel_dict (dict): Features from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. We use `point2voxel_map` in this function.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """
        voxel_dict = self.forward(voxel_dict)
        seg_pred_list = self.predict_by_feat(voxel_dict, batch_data_samples)
        return seg_pred_list

    def predict_by_feat(self, voxel_dict: dict,
                        batch_data_samples: SampleList) -> List[Tensor]:
        """Predict function.

        Args:
            voxel_dict (dict): The dict may contain `logits`,
                `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[Tensor]: List of point-wise segmentation logits.
        """
        seg_logits = voxel_dict['logits']

        seg_pred_list = []
        coors = voxel_dict['coors']
        for batch_idx in range(len(batch_data_samples)):
            if self.batch_first:
                batch_mask = coors[:, 0] == batch_idx
            else:
                batch_mask = coors[:, -1] == batch_idx
            seg_logits_sample = seg_logits[batch_mask]
            point2voxel_map = voxel_dict['point2voxel_maps'][batch_idx].long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list
