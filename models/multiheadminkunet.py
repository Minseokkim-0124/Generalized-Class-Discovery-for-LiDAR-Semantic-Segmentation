import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import torch.nn.functional

from models.minkunet import MinkUNet34C, MinkUNet34RC

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = ME.MinkowskiConvolution(
            output_dim,
            num_prototypes,
            kernel_size=1,
            bias=False,
            dimension=D)

    def forward(self, x):
        return self.prototypes(x).F #feature 반환


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(96, 128),  
            nn.ReLU(),
            nn.Linear(128, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.mlp(x)


# class CosinePrototypes(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(CosinePrototypes, self).__init__()
#         # self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
#         # self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#         # self.prototypes = nn.Linear(
#         #     output_dim,
#         #     num_prototypes,
#         #     bias=False)
#         self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
#         self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

#     def forward(self, x):
#         # breakpoint()
#         out = F.normalize(x.features, dim=1).mm(F.normalize(self.weight, dim=0))
#         return 10 * out
    
class CosinePrototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes, D=3):
        super().__init__()

        self.prototypes = nn.Linear(
            output_dim,
            num_prototypes,
            bias=False)
        self.prototypes.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        x = F.normalize(x.F, dim=-1)
        weight = F.normalize(self.prototypes.weight, dim=-1)
        return 10 * torch.mm(x, weight.T)

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
    
class MultiHeadCosine(nn.Module):
    def __init__(
        self, input_dim, num_prototypes, num_heads
    ):
        super().__init__()
        self.num_heads = num_heads
    
        # prototypes
        # self.prototypes = torch.nn.ModuleList(
        #     [NormedLinear(input_dim, num_prototypes) for _ in range(num_heads)]
        # )
        self.prototypes = torch.nn.ModuleList(
            [CosinePrototypes(input_dim, num_prototypes) for _ in range(num_heads)]
        )

    def forward_head(self, head_idx, feats):
        return self.prototypes[head_idx](feats), feats.F

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]

class MultiHeadMinkUnetFineTune(nn.Module):
    def __init__(
        self,
        num_labeled,
        num_classes,
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        # MinkUnet18
        # self.encoder = MinkUNet18A(in_channels, num_labeled)
        # MinkUnet34
        self.encoder = MinkUNet34C(1, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        self.head_lab = Prototypes(output_dim=self.feat_dim,
                                    num_prototypes=num_labeled)
        # self.head_lab2 = nn.Linear(in_features=num_labeled,
                                    # out_features=19)
        self.head_lab2 = nn.Linear(in_features=self.feat_dim,
                                    out_features=num_classes)
    

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab2(feats.F)}
        # out = {"logits_lab": self.head_lab(feats)}
        # logits_ft = self.head_lab2(out['logits_lab'])
        # out.update({
        #     "logits_ft": logits_ft,
        #     }
        # )
        
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

# base head, novel head
class MultiHeadMinkUnet(nn.Module):
    def __init__(
        self,
        num_labeled,
        num_unlabeled,
        overcluster_factor=None,
        num_heads=1,
        in_channels = 1,
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        # MinkUnet18
        # self.encoder = MinkUNet18A(in_channels, num_labeled)
        # MinkUnet34
        self.encoder = MinkUNet34C(1, num_labeled)
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
            out["feats"] = feats.F 
            return out
        
class MultiHeadMinkUnetCosine(nn.Module):
    def __init__(
        self,
        num_labeled,
        num_unlabeled,
        overcluster_factor=None,
        num_heads=1,
        in_channels = 1,
    ):
        super().__init__()

        # backbone -> pretrained model + identity as final
        # MinkUnet18
        # self.encoder = MinkUNet18A(in_channels, num_labeled)
        # MinkUnet34
        self.encoder = MinkUNet34C(in_channels, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()

        # self.head_lab = CosinePrototypes(in_features=self.feat_dim, out_features=num_labeled,)
        self.head_lab = CosinePrototypes(output_dim=self.feat_dim, num_prototypes=num_labeled,)
        
        if num_heads is not None:
            self.head_unlab = MultiHeadCosine(
                input_dim=self.feat_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads
            )
            if overcluster_factor is not None:
                self.head_unlab_over = MultiHeadCosine(
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
            out["feats"] = feats.F 
            return out

class MinkUNetBase(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels = 1,
    ):
        super().__init__()
        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34RC(in_channels, num_classes)

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder.forward_no_logits(view) for view in views]
            out = [self.encoder.forward(view) for view in views]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            if hasattr(self.encoder, 'final3'):
                feats = self.encoder.forward_no_logits(views)
                out = dict()
                out['logits'] = self.encoder.forward_novel(views)
                out["feats"] = feats.F 
                return out
            else:
                # feats = self.encoder.forward_no_logits(views)
                out = dict()
                logits, feats = self.encoder.forward(views, use_last=True)
                out['logits'] = logits.F
                out["feats"] = feats.F 
                return out
                
class MinkUNetRC(nn.Module):
    def __init__(
        self,
        num_labeled,
        in_channels = 1,    
    ):
        super().__init__()
        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34RC(in_channels, num_labeled)
        # self.encoder.final2 = ME.MinkowskiConvolution(
        #                     self.encoder.PLANES[7] * self.encoder.BLOCK.expansion,
        #                     out_channels=3,
        #                     kernel_size=1,
        #                     bias=True,
        #                     dimension=3)
            

    def forward(self, views):
        if isinstance(views, list):
            raise NotImplementedError
            feats = [self.encoder.forward_no_logits(view) for view in views]
            out = [self.encoder.forward_dummy(view) for view in views]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder.forward_no_logits(views)
            out = dict()
            # out['logits'] = self.encoder.forward_dummy(views)
            out['logits'] = self.encoder.forward_dummy(feats)
            out["feats"] = feats.F 
            return out
    
    def forward_discover(self, views):
        if isinstance(views, list):
            raise NotImplementedError
            feats = [self.encoder.forward_no_logits(view) for view in views]
            out = [self.encoder.forward_discover(view) for view in views]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        
        else:
            feats = self.encoder.forward_no_logits(views)
            out = dict()
            # out['logits'] = self.encoder.forward_novel(views)
            out['logits'] = self.encoder.forward_novel(feats)
            # out["feats"] = feats.F 
            return out


class MinkUNetRCAblation(nn.Module):
    def __init__(
        self,
        num_labeled,
        in_channels = 1,    
        ncc_head_mean = False,
        ncc_head_sum = False,
    ):
        super().__init__()
        self.ncc_head_mean = ncc_head_mean
        self.ncc_head_sum = ncc_head_sum
        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34RC(in_channels, num_labeled)
        # self.encoder.final2 = ME.MinkowskiConvolution(
        #                     self.encoder.PLANES[7] * self.encoder.BLOCK.expansion,
        #                     out_channels=3,
        #                     kernel_size=1,
        #                     bias=True,
        #                     dimension=3)
            

    def forward(self, views):
        if isinstance(views, list):
            raise NotImplementedError
            feats = [self.encoder.forward_no_logits(view) for view in views]
            out = [self.encoder.forward_dummy(view) for view in views]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder.forward_no_logits(views)
            out = dict()
            # out['logits'] = self.encoder.forward_dummy(views)
            if self.ncc_head_mean:
                out['logits'] = self.encoder.forward_dummy_mean(feats)
                out["feats"] = feats.F 
                return out
            elif self.ncc_head_sum:
                out['logits'] = self.encoder.forward_dummy_sum(feats)
                out["feats"] = feats.F 
                return out
    
    def forward_discover(self, views):
        if isinstance(views, list):
            raise NotImplementedError
            feats = [self.encoder.forward_no_logits(view) for view in views]
            out = [self.encoder.forward_discover(view) for view in views]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        
        else:
            feats = self.encoder.forward_no_logits(views)
            out = dict()
            # out['logits'] = self.encoder.forward_novel(views)
            out['logits'] = self.encoder.forward_novel(feats)
            # out["feats"] = feats.F 
            return out



class MinkUNetBaseCosine(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels = 1,
    ):
        super().__init__()
        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34RC(in_channels, num_classes)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()
        # self.head_lab = CosinePrototypes(in_features=self.feat_dim, out_features=num_classes,)
        self.head_lab = CosinePrototypes(output_dim=self.feat_dim, num_prototypes=num_classes,)

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder.forward_no_logits(views)
            out = self.forward_heads(feats)
            out["feats"] = feats.F 
            return out
            
    def forward_heads(self, feats):
        out = {"logits": self.head_lab(feats)}
        return out

class MinkUNetRCCosine(nn.Module):
    def __init__(
        self,
        num_labeled,
        in_channels = 1,
    ):
        super().__init__()
        # backbone -> pretrained model + identity as final
        self.encoder = MinkUNet34RC(in_channels, num_labeled)
        self.feat_dim = self.encoder.final.in_channels
        self.encoder.final = nn.Identity()
        self.head_lab = CosinePrototypes(output_dim=self.feat_dim, num_prototypes=num_labeled,)
        # self.head_lab = CosinePrototypes(in_features=self.feat_dim, out_features=num_labeled,)
            

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder.forward_no_logits(view) for view in views]
            out = [self.encoder.forward_dummy(view) for view in views]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder.forward_no_logits(views)
            out = dict()
            y_in_normal = self.head_lab(feats)
            y_out_normal = self.head_ncc(feats)
            y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
            y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
            out['logits_ncc'] = y_normal_dummy
            out["feats"] = feats.F 
            return out

class MultiHeadSelfSupMinkUnet(nn.Module):
    def __init__(
        self,
        dataset = 'SemanticKITTI',
    ):
        super().__init__()
        if dataset == 'nuScenes':
            in_channels = 1
        elif dataset == 'SemanticKITTI':
            in_channels = 4
        # MinkUnet34
        self.backbone = MinkUNet34RC(in_channels, 128, D=3)
        self.metric_learner = ProjectionHead()

    def forward(self, views):
        out= dict()
        feats = self.backbone.forward_no_logits(views)
        breakpoint()
        out["feats"] = feats.F 
        logits = self.backbone.final(feats)
        out['logits'] = logits.F
        return out


class MultiHeadSelfSupMinkUnet2(nn.Module):
    def __init__(
        self,
        dataset = 'SemanticKITTI',
        SimGCD = False,
    ):
        super().__init__()
        if dataset == 'nuScenes':
            in_channels = 1
        elif dataset == 'SemanticKITTI':
            in_channels = 4
        else:
            raise NotImplementedError
        # MinkUnet34
        self.backbone = MinkUNet34RC(in_channels, 128, D=3)
        # if load_final:
        #     pass
        # else:
        #     self.backbone.final = nn.Identity()
        # self.feat_dim = self.backbone.final.in_channels
        # self.head_lab = Prototypes(output_dim=self.feat_dim, num_prototypes=num_labeled)
        self.metric_learner = ProjectionHead()
        self.SimGCD = SimGCD

    # def forward_heads(self, feats):
    #     out = {"logits": self.head_lab(feats)}
    #     return out

    def forward(self, views):
        out= dict()
        feats = self.backbone.forward_no_logits(views)
        # out = self.forward_heads(feats)
        proj_feats = self.metric_learner(feats.F)    
        # out["feats"] = feats.F 
        out['proj_feats'] = proj_feats
        
        # feats = torch.nn.functional.normalize(feats, dim=-1, p=2)
        if self.SimGCD:
            normed_sp_tensor = ME.SparseTensor(features=F.normalize(feats.F, dim=1), coordinates=feats.C)
            logits = self.backbone.final(normed_sp_tensor)
            out['logits'] = logits.F

        else:
            logits = self.backbone.final(feats)
            out['logits'] = logits
        return out

class MultiHeadSelfSupMinkUnetTest(nn.Module):
    def __init__(
        self,
        dataset = 'SemanticKITTI',
    ):
        super().__init__()
        if dataset == 'nuScenes':
            in_channels = 1
        elif dataset == 'SemanticKITTI':
            in_channels = 4
        else:
            raise NotImplementedError
        # MinkUnet34
        self.backbone = MinkUNet34RC(in_channels, 128, D=3)
        # if load_final:
        #     pass
        # else:
        #     self.backbone.final = nn.Identity()
        # self.feat_dim = self.backbone.final.in_channels
        # self.head_lab = Prototypes(output_dim=self.feat_dim, num_prototypes=num_labeled)
        self.metric_learner = ProjectionHead()

    # def forward_heads(self, feats):
    #     out = {"logits": self.head_lab(feats)}
    #     return out

    def forward(self, views):
        out= dict()
        feats = self.backbone.forward_no_logits(views)
        # out = self.forward_heads(feats)
        proj_feats = feats.F
        # out["feats"] = feats.F 
        out['feats'] = proj_feats
        
        return out