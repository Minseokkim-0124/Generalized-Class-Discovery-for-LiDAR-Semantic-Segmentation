import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.minkunet import MinkUNet34C, MinkUNet18A

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)
        # 여기를 orca에선 cosine similiarity로 바꿈

    def forward(self, x):
        return self.prototypes(x).F #feature 반환

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()
        
        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)
        # 여기를 orca에선 cosine similiarity로 바꿈

    def forward(self, x):
        return self.prototypes(x).F #feature 반환

# Not Multihead
class MinkUnetToy18(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels = 1,
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        # MinkUnet18
        self.encoder = MinkUNet18A(in_channels, num_classes)
        # MinkUnet34
        # self.encoder = MinkUNet34C(1, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.head_lab = Prototypes(output_dim=self.feat_dim,
                                    num_prototypes=num_classes,)

    def forward_heads(self, feats):
        out = {"logits": self.head_lab(feats)}
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
            out["feats"] = feats.F 
            return out

# Not Multihead
class MinkUnet34(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels = 1,
    ):
        super().__init__()
        # backbone -> pretrained model + identity as final
        # MinkUnet18
        # self.encoder = MinkUNet18A(in_channels, num_classes)
        # MinkUnet34
        self.encoder = MinkUNet34C(in_channels, num_classes)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.linear = NormedLinear(in_features=self.feat_dim,
                                       out_features=num_classes,)

    def forward_heads(self, feats):
        out = {"logits": self.linear(feats)}
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
            out = self.forward_heads(feats.features)
            out["feats"] = feats.F 
            return out
        
    # np.unique(pseudo_label.cpu())        
    # np.unique(sup_mapped_label.cpu())

    # class MultiHead(nn.Module):
    #     def __init__(
    #         self, input_dim, num_prototypes, num_heads
    #     ):
    #         super().__init__()
    #         self.num_heads = num_heads
        
    #         # prototypes
    #         self.prototypes = torch.nn.ModuleList(
    #             [Prototypes(input_dim, num_prototypes) for _ in range(num_heads)]
    #         )

    #     def forward_head(self, head_idx, feats):
    #         return self.prototypes[head_idx](feats), feats.F

    #     def forward(self, feats):
    #         out = [self.forward_head(h, feats) for h in range(self.num_heads)]
    #         return [torch.stack(o) for o in map(list, zip(*out))]        
        
# class MultiHeadMinkUnet(nn.Module):
#     def __init__(
#         self,
#         num_labeled,
#         num_unlabeled,
#         overcluster_factor=None,
#         num_heads=1,
#         in_channels = 1,
#     ):
#         super().__init__()

#         # backbone -> pretrained model + identity as final
#         # MinkUnet18
#         self.encoder = MinkUNet18A(in_channels, num_labeled)
#         # MinkUnet34
#         # self.encoder = MinkUNet34C(1, num_labeled)
#         self.feat_dim = self.encoder.final.in_channels
#         self.encoder.final = nn.Identity()

#         self.head_lab = Prototypes(output_dim=self.feat_dim,
#                                     num_prototypes=num_labeled)
#         if num_heads is not None:
#             self.head_unlab = MultiHead(
#                 input_dim=self.feat_dim,
#                 num_prototypes=num_unlabeled,
#                 num_heads=num_heads
#             )

#         if overcluster_factor is not None:
#             self.head_unlab_over = MultiHead(
#                 input_dim=self.feat_dim,
#                 num_prototypes=num_unlabeled * overcluster_factor,
#                 num_heads=num_heads
#             )

#     def forward_heads(self, feats):
#         out = {"logits_lab": self.head_lab(feats)}
#         if hasattr(self, "head_unlab"):
#             logits_unlab, proj_feats_unlab = self.head_unlab(feats)
#             out.update(
#                 {
#                     "logits_unlab": logits_unlab,
#                     "proj_feats_unlab": proj_feats_unlab,
#                 }
#             )
#         if hasattr(self, "head_unlab_over"):
#             logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
#             out.update(
#                 {
#                     "logits_unlab_over": logits_unlab_over,
#                     "proj_feats_unlab_over": proj_feats_unlab_over,
#                 }
#             )
#         return out

#     def forward(self, views):
#         if isinstance(views, list):
#             feats = [self.encoder(view) for view in views]
#             out = [self.forward_heads(f) for f in feats]
#             out_dict = {"feats": torch.stack(feats)}
#             for key in out[0].keys():
#                 out_dict[key] = torch.stack([o[key] for o in out])
#             return out_dict
#         else:
#             feats = self.encoder(views)
#             out = self.forward_heads(feats)
#             out["feats"] = feats.F 
#             return out