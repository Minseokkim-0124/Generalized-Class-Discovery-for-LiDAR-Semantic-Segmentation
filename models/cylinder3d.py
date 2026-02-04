import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch import Tensor

import spconv.pytorch as spconv

from typing import Any, List, Optional, Tuple, Union, Sequence
from mmcv.utils import ext_loader
from mmcv.cnn import build_norm_layer
from mmcv.ops import DynamicScatter
from .backbone import Asymm3DSpconv 
# from .decoder import
from .voxelizer import Voxelizer
from .encoder import SegVFE
from .decoder import Cylinder3DHead

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = spconv.SparseConv3d(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False)
        
    def forward(self, x):
        return self.prototypes(x).F 

#얘로 unlabel을 feature를 뽑는듯?
class MultiHead(nn.Module):
    def __init__(
        self, input_dim, num_prototypes, num_heads
    ):
        super().__init__()
        self.num_heads = num_heads
    
        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
        )

    def forward_head(self, head_idx, feats):
        return self.prototypes[head_idx](feats), feats.F

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]

class MultiHeadCylinder3D(nn.Module):
    """
    Lasermix model porting for Pytorch lightninig training for baseline.
     
    모델 구성 순서: 
    voxelize - voxel_encoder - backbone - decode_head 
    
    """
    def __init__(
      self,
      num_labeled,
      num_unlabeled,  
      overcluster_factor=None,
      num_heads=1,
      config_file=None,
      ):
        super().__init__()
        num_classes = num_labeled + num_unlabeled
        # self.voxelizer = Voxelizer(voxel= True,
        #     voxel_type='cylindrical',
        #     point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
        #     grid_shape=[240, 180, 20],
        #     max_num_points=-1,
        #     max_voxels=-1,) 
        self.encoder = SegVFE(feat_channels=[64, 128, 256, 256], in_channels=6, with_voxel_center=True,
                              feat_compression=16, grid_shape=[240, 180, 20])
        self.backbone = Asymm3DSpconv(grid_size=[240, 180, 20], input_channels=16, base_channels=32,
                                      norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1))
        self.decoder_head = Cylinder3DHead(channels=128, num_classes=num_classes,
                                           loss_ce=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, class_weight=None, loss_weight=1.0),
                                           loss_lovasz=dict(type='LovaszLoss', loss_weight=3.0, reduction='none'))
        breakpoint()
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()
        self.head_lab = Prototypes(output_dim=self.feat_dim,
                                    num_prototypes=num_labeled)
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads
            )

        if overcluster_factor is not None:
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads
            )

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(feats)}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                }
            )
        if hasattr(self, "head_unlab_over"):
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats.F #비어있지 않은 텐서 데이터를 나타냄
            return out
    