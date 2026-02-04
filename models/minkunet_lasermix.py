import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch import Tensor

from .backbone import MinkUNetBackbone 
from .minkunet import MinkUNet18A

import torchsparse.nn as spnn
from torchsparse import SparseTensor

from .decoder import MinkUNetHead

def to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_device(i, device) for i in obj]
        else:
            return obj
        
class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()
        self.prototypes = spnn.Conv3d(output_dim, num_prototypes, kernel_size=1, stride=1, bias=False, dilation=1)
        
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
        
class MultiHeadMinkUnet18(nn.Module):
    """
    Lasermix model porting for Pytorch lightninig training for baseline.
     
    모델 구성 순서: 
    backbone - decode_head 
    
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
        # self.backbone = MinkUNet18A(4, num_unlabeled) #ME based
        self.backbone = MinkUNetBackbone(in_channels=4, num_stages=4, base_channels=32, #mmdet3d based
            encoder_channels=[32, 64, 128, 256], encoder_blocks=[2, 2, 2, 2],
            decoder_channels=[256, 128, 96, 96], decoder_blocks=[2, 2, 2, 2],
            block_type='basic', sparseconv_backend='torchsparse')
        self.decoder_head = MinkUNetHead(channels=96, num_classes=num_classes, batch_first=True, dropout_ratio=0,
                                         loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True), ignore_index=-1)
        self.feat_dim = self.decoder_head.channels
        # backbone -> pretrained model + identity as final
        self.decoder_head.conv_seg = nn.Identity()
        
       
        # self.decoder_head. = nn.Identity()
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
        if isinstance(feats, dict):
            feats = feats['voxel_feats']
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
    
    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """Extract features from voxels.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'voxels' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - voxels (dict): Voxel feature and coords after voxelization.

        Returns:
            dict: The dict containing features.
        """
        voxel_dict = batch_inputs_dict['voxels'].copy()
        x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
        if self.with_neck:
            x = self.neck(x)
        voxel_dict['voxel_feats'] = x
        return voxel_dict
    
    def extract_feat(self, batch_inputs_dict: dict) -> dict:
            """Extract features from voxels.

            Args:
                batch_inputs_dict (dict): Input sample dict which
                    includes 'points' and 'voxels' keys.

                    - points (List[Tensor]): Point cloud of each sample.
                    - voxels (dict): Voxel feature and coords after voxelization.

            Returns:
                dict: The dict containing features.
            """
            batch_inputs_dict = batch_inputs_dict['inputs']
            voxel_dict = batch_inputs_dict['voxels'].copy()
            x = self.backbone(voxel_dict['voxels'], voxel_dict['coors'])
            voxel_dict['voxel_feats'] = x
            return voxel_dict
    
    def forward(self, views):
        if isinstance(views, list):
            feats = [self.backbone(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        
        elif isinstance(views, dict):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            views = to_device(views, device)
            # NOPS는 backbone에 sparsetensor가 들어감 lasermix는 그냥 일반 voxel에 들어감 
            feats = self.extract_feat(views)  # backbone 여기서 sparsetensor의 feature 추출
            feats = self.decoder_head(feats)  # backbone
            feats = SparseTensor(feats['voxel_feats'].float(), feats['coors'].int())
            out = self.forward_heads(feats)
            out["feats"] = feats.F
            return out
    
    