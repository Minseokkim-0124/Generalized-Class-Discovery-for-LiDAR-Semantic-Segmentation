import os
import sys
from itertools import chain as chain_iterators
from itertools import zip_longest, cycle
import pickle

import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Loss Function
from utils.loss_LiON import crude_dynamic_energy_loss, Gambler

# Optimizer
from torch import optim
from utils.scheduler import LinearWarmupCosineAnnealingLR

# Clustering
from fast_pytorch_kmeans import KMeans
from sklearn.cluster import DBSCAN


from models.multiheadminkunet import MinkUNetBase, MinkUNetRC, MinkUNetBaseCosine, MinkUNetRCCosine
from utils.collation import (
    collation_fn_restricted_dataset,
    collation_fn_polarmix_dataset,
    collation_fn_lasermix_dataset,
)
from mmdet.datasets.samplers import MultiSourceSampler
from torch.utils.data import ConcatDataset, DataLoader, Subset, SequentialSampler
from utils.dataset_remission_mean_teacher import get_dataset as get_dataset_remission 
from utils.sinkhorn_knopp import SinkhornKnopp

# Visualization
from utils.visualize_ply import *
from utils.visualize import get_color, get_color_cluster

from .exp import ExpPretrain

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

class ExpMergeDiscover_PolarMix_MeanTeacher(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )
        self.model_t = MinkUNetRC(
            num_labeled=self.hparams.num_labeled_classes,
        )
        self.model_s = MinkUNetRC(
            num_labeled=self.hparams.num_labeled_classes,
        )

        self.label_mapping = label_mapping
        self.label_mapping_inv = label_mapping_inv
        self.unknown_label = unknown_label
        self.unknown_real_labels = []
        self.known_real_labels = []
        for key_, val_ in self.label_mapping.items():
            if val_ == self.unknown_label:
                self.unknown_real_labels.append(key_)
            else:
                self.known_real_labels.append(key_)

        breakpoint()
        if self.hparams.resume_checkpoint is not None:
            print("##############################Resume Training##############################")
            # NCC
            self.model_t.encoder.final2 = ME.MinkowskiConvolution(
                            self.model_t.encoder.PLANES[7] * self.model_t.encoder.BLOCK.expansion,
                            out_channels=3,
                            kernel_size=1,
                            bias=True,
                            dimension=3)
            # Novel Classifier 
            self.model_t.encoder.final3 = ME.MinkowskiConvolution(
                        self.model_t.encoder.PLANES[7] * self.model_t.encoder.BLOCK.expansion,
                        out_channels=self.hparams.num_unlabeled_classes,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
            
            self.model_s.encoder.final2 = ME.MinkowskiConvolution(
                            self.model_s.encoder.PLANES[7] * self.model_s.encoder.BLOCK.expansion,
                            out_channels=3,
                            kernel_size=1,
                            bias=True,
                            dimension=3)
            # Novel Classifier 
            self.model_s.encoder.final3 = ME.MinkowskiConvolution(
                        self.model_s.encoder.PLANES[7] * self.model_s.encoder.BLOCK.expansion,
                        out_channels=self.hparams.num_unlabeled_classes,
                        kernel_size=1,
                        bias=True,
                        dimension=3)

        elif self.hparams.pretrained is not None:
            self.load_pretrained(self.model_t, pretrained=self.hparams.pretrained)
            # NCC 
            
            self.load_pretrained(self.model_s, pretrained=self.hparams.pretrained)
            # NCC 
            
        
        # else:
        #     raise FileNotFoundError("Checkpoint is needed for Merge Discover Stage")
        
        self.model_t.encoder.final2 = ME.MinkowskiConvolution(
                            self.model_t.encoder.PLANES[7] * self.model_t.encoder.BLOCK.expansion,
                            out_channels=3,
                            kernel_size=1,
                            bias=True,
                            dimension=3)
            # Novel Classifier
        self.model_t.encoder.final3 = ME.MinkowskiConvolution(
                    self.model_t.encoder.PLANES[7] * self.model_t.encoder.BLOCK.expansion,
                    out_channels=self.hparams.num_unlabeled_classes,
                    kernel_size=1,
                    bias=True,
                    dimension=3)
        self.model_s.encoder.final2 = ME.MinkowskiConvolution(
                            self.model_s.encoder.PLANES[7] * self.model_s.encoder.BLOCK.expansion,
                            out_channels=3,
                            kernel_size=1,
                            bias=True,
                            dimension=3)
        # Novel Classifier
        self.model_s.encoder.final3 = ME.MinkowskiConvolution(
                    self.model_s.encoder.PLANES[7] * self.model_s.encoder.BLOCK.expansion,
                    out_channels=self.hparams.num_unlabeled_classes,
                    kernel_size=1,
                    bias=True,
                    dimension=3)
        # NOTE: Freeze teacher model
        self.freeze(self.model_t)

        dataset_config_file = self.hparams.dataset_config
        with open(dataset_config_file, "r") as f:
            dataset_config = yaml.safe_load(f)
        map_inv = dataset_config["learning_map_inv"]
        lab_dict = dataset_config["labels"]
        label_dict = {}
        label_dict_inv = {}
        for new_label, old_label in map_inv.items():
            label_dict[new_label] = lab_dict[old_label]
            label_dict_inv[lab_dict[old_label]] = new_label
        self.label_dict = label_dict
        self.label_dict_inv = label_dict_inv

        self.debug = self.hparams.debug

        if self.hparams.visualize:
            self.save_dir = os.path.join(self.hparams.log_dir,'visualize')
            os.makedirs(self.save_dir, exist_ok=True)   

        self.kmeans = KMeans(n_clusters=self.hparams.num_unlabeled_classes + 1, mode='euclidean', verbose=0)

        self.novel_queue = None
        self.novel_indices= []
        self.novel_queue_batches = 20
        self.use_first_dataloader = False 
        self.use_uncertainty_queue = True
        self.queue_percentage = 0.05
        
        self.novel_queue_thld = 0.2 #0.01
        self.novel_candidate_thld = 0.2 # 0.03
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.valid_criterion = torch.nn.CrossEntropyLoss()
        self.novel_criterion = torch.nn.CrossEntropyLoss()

        self.warmup_epoch = 0 # 1
        self.mixing_ratio_feat = 0.1
        self.unlabel_mixing = True
        self.unlabel_calib = True
        self.use_sup_logit = True
        self.use_unsup_ncc_loss = True
        
        if self.hparams.dataset == 'nuScenes':
            from nuscenes import NuScenes
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_config['dataset_path'], verbose=True)

            self.calib_coeff = 0.1
            self.unsup_calib_coeff = 0.1
            self.mix_unsup_seg_coeff = 1
            self.novel_coeff = 0.1 # 0~1
            self.sup_novel_coeff = 1
            self.ncc_coeff = 0.1

            self.centroid_sampling = False
            self.beta_sampling = True
            self.beta_coeff = 0.5
        
        # NOTE: KITTI
        else:
            self.nusc = None
            self.calib_coeff = 0.05 
            self.unsup_calib_coeff = 0.05
            self.mix_unsup_seg_coeff = 0.1
            self.novel_coeff = 0.1
            self.ncc_coeff = 0.1

            self.centroid_sampling = True
            self.beta_sampling = True
            self.beta_coeff = 0.5

        return
    
    def load_pretrained(self, model, pretrained):
        state_dict = torch.load(pretrained)
        state_dict = state_dict['state_dict']
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if 'model.' in key:
                new_state_dict[key.replace('model.', '')] = state_dict[key]
                # new_state_dict[key.replace('model', '{}').format(name)] = state_dict[key]
                # state_dict[key.replace(model_name, '')] = state_dict[key]
                del state_dict[key]
                
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        print(f'Missing: {missing_keys}', f'Unexpected: {unexpected_keys}')
        
        return

    def _update_ema_variables(self, momentum):
        for param_t, param_s in zip(self.model_t.parameters(), self.model_s.parameters()):
            param_t.data.mul_(1 - momentum).add_(param_s.data, alpha=momentum)

    @staticmethod
    def freeze(model: nn.Module) -> None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    def configure_optimizers(self):
        optimizer = optim.SGD(
            params=self.model_s.parameters(),
            lr=self.hparams.train_lr,
            momentum=self.hparams.momentum_for_optim,
            weight_decay=self.hparams.weight_decay_for_optim,
        )

        if self.hparams.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.epochs,
                warmup_start_lr=self.hparams.min_lr,
                eta_min=self.hparams.min_lr,
            )

            return [optimizer], [scheduler]

        return optimizer

    def train_dataloader(self):
        lab_dataset = get_dataset_remission(self.hparams.dataset, "polarmix")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
            resize_aug = True,
            polarmix = True,
        )

        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'polarmix')(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
                resize_aug = False,
                polarmix = False,
            )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                lab_dataset.nusc_infos = lab_dataset.nusc_infos[::100]
                unlab_dataset.nusc_infos = unlab_dataset.nusc_infos[::100][:len(lab_dataset)]
            else:
                lab_dataset.num_files = 50
                unlab_dataset.num_files = 50

        lab_dataloader = DataLoader(
            dataset=lab_dataset,
            batch_size=self.hparams.batch_size // 2,  
            collate_fn=collation_fn_polarmix_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,  
        )

        unlab_dataloader = DataLoader(
            dataset=unlab_dataset,
            batch_size=self.hparams.batch_size // 2,  
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,  
        )

        self.lab_dataloader_iter = cycle(lab_dataloader)
        self.unlab_dataloader_iter = cycle(unlab_dataloader)
        
        return [lab_dataloader, unlab_dataloader] 
    
    def val_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, 'labeled')(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                dataset.nusc_infos = dataset.nusc_infos[::100]
            else:
                dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return dataloader
        
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.use_first_dataloader = False
            
        if self.current_epoch < self.warmup_epoch:
            print("################## Warmup stage starts ##################")
        
        else:
            print("################## Main stage starts ##################")

    def training_step(self, data, _, dataloader_idx=0):
        
        sup_data = data[0]
        pm_coords = sup_data['polarmix']['coords']
        pm_feats = sup_data['polarmix']['feats']
        pm_labels = sup_data['polarmix']['labels']
        pm_mapped_labels = sup_data['polarmix']['mapped_labels']
        pm_inverse_maps = sup_data['polarmix']['inverse_maps']
        pm_selected_indexes = sup_data['polarmix']['selected_idx']

        unsup_coords, unsup_feats, _, _, unsup_mapped_labels, unsup_inverse_maps, unsup_pcd_indexes = data[1]
        unsup_pcd_masks = []
        # for i in range(len(pcd_indexes)):
        for i in range(len(unsup_pcd_indexes)):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)

        # unsup_coords[:,0] += len(pcd_indexes)
        unsup_coords[:,0] += len(unsup_pcd_indexes)
        
        coords_cat = torch.cat((pm_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((pm_feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())

        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']

        logits_s = out_s['logits']
        feats_s = out_s['feats']

        sup_data_num = len(pm_selected_indexes)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]

        # logits_sup_t = logits_t[:sup_data_num, :]
        # feats_sup_t = feats_t[:sup_data_num, :]
        
        """
        TODO
        1. labeled 로 mixing 하고 soft logits으로 cross entropy 주기
        2. calibration할때 soft logits들은 다 빼기
        3. unlabel에 대해서 어떻게 할지 고민하기
        - ignore
        - calib only
        - calib + confident prob based mixing 
        4. novel queue 쌓기  : queue 어떻게 쌓는게 의미가 있을까..
        """
        
        # NOTE: labeled dataset
        loss = 0
        # probs_sup = F.softmax(logits_sup, dim=1)
        # max_probs, target_labels = torch.max(probs_sup, dim=1)
        
        if self.centroid_sampling:
            mix_feats_s, mix_targets_s = self.mix_centroid_features(feats=feats_sup_s, labels=pm_mapped_labels)
        else:
            mix_feats_s, mix_targets_s = self.mix_features(feats=feats_sup_s, labels=pm_mapped_labels, mixing_ratio=self.mixing_ratio_feat)
        
        final_layer_kernel = self.model_s.encoder.final.kernel
        final_layer_bias = self.model_s.encoder.final.bias
        final_layer2_kernel = self.model_s.encoder.final2.kernel
        final_layer2_bias = self.model_s.encoder.final2.bias
        
        mix_in_normal = torch.matmul(mix_feats_s.detach(), final_layer_kernel) + final_layer_bias
        mix_out_normal = torch.matmul(mix_feats_s.detach(), final_layer2_kernel) + final_layer2_bias
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits_s = torch.cat([mix_in_normal, mix_out_normal_2], dim =1)

        logits_mix_sup_s = torch.cat([logits_sup_s, mix_logits_s], dim=0)
                    
        targets_sup_s = (
        F.one_hot(
            pm_mapped_labels.long(),
            num_classes=self.hparams.num_labeled_classes + 1,
        )
        .float()
        .to(self.device)
        )
        target_mix_sup_s = torch.cat([targets_sup_s, mix_targets_s])

        mix_seg_loss_s = self.criterion(logits_mix_sup_s, target_mix_sup_s)
        loss += mix_seg_loss_s
        self.log("train/student_sup_mix_seg_loss", mix_seg_loss_s.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("ss_mix", mix_seg_loss_s.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
    
        target_tmp = target_mix_sup_s.clone()

        # Mixing soft labels에서 각 클래스 확률이 존재하는 위치 추출
        mixing_class_indices = torch.nonzero(target_tmp > 0, as_tuple=True)  # (샘플 인덱스, 클래스 인덱스)

        # Breakpoint 이후 수정된 로직
        # index_tmp = torch.arange(0, logits_mix_sup.shape[0]).unsqueeze(0).to(logits_sup.device)

        # preds_dummy 초기화
        preds_dummy = logits_mix_sup_s.clone()

        # Mixing된 soft labels에서 확률 무효화
        preds_dummy[mixing_class_indices[0], mixing_class_indices[1]] = -1e9
        # for idx in torch.unique(mixing_class_indices[0]):  
        #     sample_indices = mixing_class_indices[0] == idx  
        #     class_indices = mixing_class_indices[1][sample_indices]  
        #     preds_dummy[idx, class_indices] = -1e9  

        # labels_dummy 생성
        labels_dummy = torch.ones(logits_mix_sup_s.shape[0]).long().to(logits_mix_sup_s.device) * self.unknown_label
        # labels_dummy[target_tmp.squeeze() == -1] = -1

        # Calibration loss 계산
        calib_loss_s = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss_s
        self.log("train/sup_calib_loss", calib_loss_s.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("s_cal", calib_loss_s.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # NOTE: ema update 해줘야 함 teacher model에 
        # target_tmp = target_mix_sup.clone()
        # breakpoint()
        # index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        # index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

        # preds_dummy = logits_sup.clone()
        # preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        # labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        # labels_dummy[labels_tmp.squeeze() == -1] = -1

        # calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        # loss += calib_loss
    
        # NOTE: unlabeled dataset
        logits_unsup_s = logits_s[sup_data_num:, :]
        feats_unsup_s = feats_s[sup_data_num:, :]

        logits_unsup_t = logits_t[sup_data_num:, :]
        feats_unsup_t = feats_t[sup_data_num:, :]
        

        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)

        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        max_probs_unsup_s, target_labels_unsup_s = torch.max(prob_unsup_s, dim=1)
        
        confident_mask = max_probs_unsup_t > 0.9
        confident_target_label_t = target_labels_unsup_t[confident_mask]
        confident_feats_t = feats_unsup_t[confident_mask]
        confident_feats_s = feats_unsup_s[confident_mask]
        
        if self.unlabel_mixing:
            if self.centroid_sampling:
                mix_unsup_feats, mix_unsup_targets = self.mix_centroid_features(feats=confident_feats_t, labels=confident_target_label_t)
            else:
                mix_unsup_feats, mix_unsup_targets = self.mix_features(feats=confident_feats_t, labels=confident_target_label_t, mixing_ratio=self.mixing_ratio_feat)
            
            # pseudo_gt_unsup = torch.argmax(prob_unsup[confident_mask], dim=1)
            pseudo_gt_unsup = confident_target_label_t
            pseudo_target_unsup = (
            F.one_hot(
                pseudo_gt_unsup.to(torch.long),
                num_classes=self.hparams.num_labeled_classes + 1,
            )
            .float()
            .to(self.device)
            )
            
            mix_in_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer_kernel) + final_layer_bias
            mix_out_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer2_kernel) + final_layer2_bias
            mix_out_normal_2_unsup, _ = torch.max(mix_out_normal_unsup, dim=1, keepdim=True)
            mix_unsup_logits = torch.cat([mix_in_normal_unsup, mix_out_normal_2_unsup], dim =1)
            
            logits_mix_unsup_s = torch.cat([logits_unsup_s[confident_mask], mix_unsup_logits], dim=0)
            target_mix_unsup_t = torch.cat([pseudo_target_unsup, mix_unsup_targets], dim=0)
            
            mix_unsup_seg_loss = self.mix_unsup_seg_coeff * self.criterion(logits_mix_unsup_s, target_mix_unsup_t)
            loss += mix_unsup_seg_loss
            self.log("train/unsup_mix_seg_loss", mix_unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("u_mix", mix_unsup_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        if self.unlabel_calib:
            target_tmp = target_mix_unsup_t.clone()

            # Mixing soft labels에서 각 클래스 확률이 존재하는 위치 추출
            mixing_class_indices = torch.nonzero(target_tmp > 0, as_tuple=True)  # (샘플 인덱스, 클래스 인덱스)

            # Breakpoint 이후 수정된 로직
            # index_tmp = torch.arange(0, logits_mix_unsup.shape[0]).unsqueeze(0).to(logits_mix_unsup.device)

            # preds_dummy 초기화
            preds_dummy = logits_mix_unsup_s.clone()

            # Mixing된 soft labels에서 확률 무효화
            preds_dummy[mixing_class_indices[0], mixing_class_indices[1]] = -1e9
            # for idx in torch.unique(mixing_class_indices[0]):  
            #     sample_indices = mixing_class_indices[0] == idx  
            #     class_indices = mixing_class_indices[1][sample_indices]  
            #     preds_dummy[idx, class_indices] = -1e9  

            # labels_dummy 생성
            labels_dummy = torch.ones(logits_mix_unsup_s.shape[0]).long().to(logits_mix_unsup_s.device) * self.unknown_label
            # labels_dummy[target_tmp.squeeze() == -1] = -1

            # Calibration loss 계산
            unsup_calib_loss = self.unsup_calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += unsup_calib_loss
            self.log("train/unsup_calib_loss", unsup_calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("u_cal", unsup_calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # if self.current_epoch < self.warmup_epoch:
        #     # update novel queue
        #     ncc_unsup = prob_unsup[:, -1]

        #     # threshold = 0.2 # NCC threshold
        #     # novel_candidate_mask = ncc_unsup > threshold
        #     novel_candidate_mask = ncc_unsup > self.novel_queue_thld

        #     if novel_candidate_mask.sum() > 0:
        #         novel_feats = feats_unsup[novel_candidate_mask]
        #         # mean_novel_feats = torch.mean(novel_feats, dim=0)
        #         self.update_queue(novel_feats.detach())
        #         self.log("utils/novel_feats", len(novel_feats))
            
        #     if self.novel_queue is not None:
        #         self.log("utils/novel_queue", len(self.novel_queue))
            
        #     return loss
            
        # print("Main Stage starts")
        """
        TODO
        1. clustering
        - over clustering 해서 metric에 따라 novel classifier 학습
        """
        
        # NOTE: unlabeled dataset
        # threshold = 0.2
        ncc_unsup = prob_unsup[:,-1]
        # novel_candidate_mask = ncc_unsup > threshold
        novel_candidate_mask = ncc_unsup > self.novel_candidate_thld
        num_novel_candidate = novel_candidate_mask.sum()

        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup[novel_candidate_mask]
            
            if self.novel_queue is not None:
                novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
            
            # if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            if len(novel_cand_feats) > int(self.hparams.num_unlabeled_classes + 1):
                # Run Kmeans clustering to the novel candidate
                # kmeans = self.kmeans.fit(novel_cand_feats.detach().cpu())
                # cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
                # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats.detach())
                cluster_centers = self.kmeans.centroids
                
                # NOTE: base classifier / ncc?, metric: euclidean / cosine?
                cluster_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                # cluster_logits.max(dim=1)[0]
                
                unrealiable_labels = torch.argmax(cluster_logits.max(dim=1)[0]).item()
                
                # discard queue features
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = cluster_curr_labels == unrealiable_labels
                
                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]
                
                # final_layer_kernel = base prototypes
                # base_prototypes = final_layer_kernel.double().detach()
                # distances = torch.tensor([
                #     torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
                #     for center in cluster_centers
                # ], device=self.device)
                
                # unrealiable_labels = torch.argmin(distances).item()
                # unrealiable_mask = cluster_labels == unrealiable_labels
                # unrealiable_feats = novel_feats[unrealiable_mask]
                # reliable_feats = novel_feats[~unrealiable_mask]
                # reliable_cluster_labels = cluster_labels[~unrealiable_mask]

                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels
                # label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

                # label_mapping을 사용해 reliable_cluster_labels 재정렬
                # reliable_cluster_labels = torch.tensor(
                #     [label_mapping[label.item()] for label in reliable_cluster_labels],
                #     device=reliable_cluster_labels.device
                # )
                
                novel_kernel = self.model.encoder.final3.kernel
                novel_bias = self.model.encoder.final3.bias
                
                # realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_kernel) + novel_bias
                realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

                unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                np.add.at(unsup_cost_matrix, tuple(idxs), 1)

                row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                for row, col in zip(row_ind, col_ind):
                    mapping_label[col] = row
                mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                
                # mapping = {row: col for row, col in zip(row_ind, col_ind)} # pred 2 label
                # mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

                # for pseudo_label in range(self.hparams.num_unlabeled_classes):
                #     mask_pseudo = realiable_novel_preds == pseudo_label
                #     for j, real_label in enumerate(self.unknown_real_labels):
                #         mask_real = real_labels[mask_unknown] == real_label
                #         cost_matrix[pseudo_label, j] += torch.logical_and(
                #             mask_pseudo, mask_real
                #         ).sum()
                
                # novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
                
                ### (optional)
                if self.use_sup_logit:
                    realiable_novel_logits = torch.cat((logits_unsup[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                    mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                
                novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                loss += novel_unsup_loss
                self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_sup_logit:
                    novel_logits_sup = torch.matmul(feats_sup, novel_kernel) + novel_bias
                    novel_logits_sup = torch.cat((logits_sup[:,:-1], novel_logits_sup), dim=1)
                    novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, mapped_labels.long())
                    loss += novel_sup_loss
                    self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_unsup_ncc_loss:
                    ncc_kernel = self.model.encoder.final2.kernel
                    ncc_bias = self.model.encoder.final2.bias
                    
                    # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                    realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                    realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                    realiable_ncc_logits = torch.cat((logits_unsup[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                    ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                    
                    ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                    loss += ncc_unsup_loss
                    self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ## update novel queue
                # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                # self.update_queue(mean_reliable_feats.detach())
                # 
                self.update_queue(reliable_feats.detach())
                self.log("utils/novel_feats", len(reliable_feats))
                self.log("utils/novel_queue", len(self.novel_queue))
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, _):
        self._update_ema_variables(momentum=0.01)

    def update_queue(self, feats: torch.Tensor):
        """
        Updates self.queue with the features of the novel points in the current batch

        :param feats: the features for the novel points in the current batch
        """
        feats = feats.detach()
        if not self.use_uncertainty_queue:
            n_feats_to_retain = int(feats.shape[0] * self.queue_percentage)
            mask = torch.randperm(feats.shape[0])[:n_feats_to_retain]
        else:
            n_feats_to_retain = feats.shape[0]
            mask = torch.ones(n_feats_to_retain, device=feats.device, dtype=torch.bool)
        if self.novel_queue is None:
            self.novel_queue = feats[mask]
            self.novel_indices.append(n_feats_to_retain)
            return

        if len(self.novel_indices) < self.novel_queue_batches:
            self.novel_queue = torch.vstack((feats[mask], self.novel_queue))
            self.novel_indices.insert(0, n_feats_to_retain)
        else:
            self.novel_queue = torch.vstack(
                (feats[mask], self.novel_queue[: -self.novel_indices[-1]])
            )
            self.novel_indices.insert(0, n_feats_to_retain)
            del self.novel_indices[-1]

    def on_validation_epoch_start(self):
        D = int(self.hparams.num_classes)
        self.cost_matrix = np.zeros((D, D), dtype=np.int64)
        self.cost_matrix_sep = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)
        for i in range(len(self.unknown_real_labels)):
            self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
            
    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())

        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # out = self.model(sp_tensor)
        # out_novel = self.model.forward_discover(sp_tensor)
        out = self.model.forward_discover(sp_tensor)
        
        for pcd_idx in range(pcd_indexes.shape[0]):
            logits = out['logits'][coords[:,0] == pcd_idx]
            # novel_logits = out_novel['logits'][coords[:,0] == pcd_idx][:,-self.hparams.num_unlabeled_classes:]
            probs_all = F.softmax(logits, dim=1)
            probs = probs_all[:,:self.hparams.num_labeled_classes]
            probs_novel = probs_all[:, self.hparams.num_labeled_classes:-1]

            # for seperate evaluation
            rc_probs = probs_all[:,-1]
            novel_cand_mask = rc_probs > self.novel_candidate_thld
            
            preds = torch.argmax(probs[~novel_cand_mask], dim=1)
            novel_preds = torch.argmax(probs_novel[novel_cand_mask], dim=1)
            novel_preds += self.hparams.num_labeled_classes

            pcd_labels = real_labels[coords[:,0] == pcd_idx]
            total_preds = torch.cat((preds, novel_preds), dim=-1)
            total_labels = torch.cat((pcd_labels[~novel_cand_mask], pcd_labels[novel_cand_mask]), dim=-1)
            # logits_novel = out_novel['logits_novel']
            # total_logits = torch.cat((logits, logits_novel), dim=1)
            # preds = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)

            for key, value in self.label_mapping_inv.items():
                total_preds[total_preds == key] = -value
            total_preds = -total_preds

            pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
            pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix_sep, tuple(idxs), 1)
            
            # for classifier only
            total_preds = torch.argmax(probs_all[:,:-1], dim=1)
            total_labels = real_labels[coords[:,0] == pcd_idx]
            
            for key, value in self.label_mapping_inv.items():
                total_preds[total_preds == key] = -value
            total_preds = -total_preds
            
            pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
            pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            
            
            # # For Greedy-Hungarian
            # pcd_mask_lab = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]].cpu().numpy()
            # idxs_old = np.stack([pcd_preds[pcd_mask_lab], pcd_labels[pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_old, tuple(idxs_old), 1)
            
            # idxs_new = np.stack([pcd_preds[~pcd_mask_lab], pcd_labels[~pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_new, tuple(idxs_new), 1)

        return 
    
    def validation_epoch_end(self, valid_step_outputs):
        unknown_real_labels = np.array(self.unknown_real_labels)
        
        conf_matrix = self.cost_matrix[self.unknown_real_labels, :][:, self.unknown_real_labels]
        row_ind, col_ind = linear_sum_assignment(conf_matrix.max() - conf_matrix)
        self.cost_matrix[:,self.unknown_real_labels] = self.cost_matrix[:,np.array(self.unknown_real_labels)[col_ind]]
        include = np.arange(self.hparams.num_classes)
        include[self.unknown_real_labels] = np.array(self.unknown_real_labels)[np.argsort(col_ind)]
        
        
        
        conf_matrix_sep = self.cost_matrix_sep[self.unknown_real_labels, :][:, self.unknown_real_labels]
        row_ind_sep, col_ind_sep = linear_sum_assignment(conf_matrix_sep.max() - conf_matrix_sep)
        self.cost_matrix_sep[:,self.unknown_real_labels] = self.cost_matrix_sep[:,np.array(self.unknown_real_labels)[col_ind_sep]]
        
        include_sep = np.arange(self.hparams.num_classes)
        include_sep[self.unknown_real_labels] = np.array(self.unknown_real_labels)[np.argsort(col_ind_sep)]
        
        
        # for i, j in zip(unknown_real_labels[row_ind], unknown_real_labels[col_ind]):
        #     if i != j:
        #         self.cost_matrix[:, [i, j]] = self.cost_matrix[:, [j, i]]
        #         original_index = np.where(include == i)[0][0]
        #         matched_index = np.where(include == j)[0][0]
        
        #         include[original_index], include[matched_index] = include[matched_index], include[original_index]
        
        # for i, j in zip(unknown_real_labels[row_ind_sep], unknown_real_labels[col_ind_sep]):
        #     if i != j:
        #         self.cost_matrix_sep[:, [i, j]] = self.cost_matrix_sep[:, [j, i]]
        #         original_index = np.where(include == i)[0][0]
        #         matched_index = np.where(include == j)[0][0]
        
        #         include[original_index], include[matched_index] = include[matched_index], include[original_index]
        
        # for original, matched in zip(self.unknown_real_labels, matched_indices):
        #     # original과 matched가 다를 때만 교환
        #     if original != matched:
        #         # original과 matched의 위치를 찾아서 swap
        #         original_index = np.where(include == original)[0][0]
        #         matched_index = np.where(include == matched)[0][0]

        # ind = linear_sum_assignment(self.cost_matrix.max() - self.cost_matrix)
        # ind = np.vstack(ind).T
        # conf_matrix = self.cost_matrix[:, ind[:,1]]
        # include_ = np.argsort(ind[:,1])[:self.hparams.num_classes]

        iou = self.get_iou(self.cost_matrix, include)
        
        miou = iou.mean()
        miou_old = iou[self.known_real_labels].mean()
        miou_new = iou[self.unknown_real_labels].mean()

        _, output_dict = self.log_metric(miou, iou)

        self.log(f'valid/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'valid/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
            
        iou_sep = self.get_iou(self.cost_matrix_sep, include_sep)
        
        miou_sep = iou_sep.mean()
        miou_old_sep = iou_sep[self.known_real_labels].mean()
        miou_new_sep = iou_sep[self.unknown_real_labels].mean()

        _, output_dict_sep = self.log_metric(miou_sep, iou_sep)

        self.log(f'valid_sep/mIoU', miou_sep, on_step=False, rank_zero_only=True)
        self.log(f'valid_sep/mIoU/old', miou_old_sep, on_step=False, rank_zero_only=True)
        self.log(f'valid_sep/mIoU/new', miou_new_sep, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict_sep.items():
            if class_ == 'all': continue
            self.log(f'valid_sep/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        self.visualization = visualization
        D = int(self.hparams.num_classes)
        self.cost_matrix = np.zeros((D, D), dtype=np.int64)
        self.novel_thresholds = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26]
        # self.novel_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        # self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)
        if self.visualization:
            self.save_dir = os.path.join(self.hparams.log_dir,'visualize')
            os.makedirs(self.save_dir, exist_ok=True)
            
        
        for i in range(len(self.unknown_real_labels)):
            self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
        
        return
    
    def test_step(self, data, _):
        # coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # # Forward
        # sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        # if self.global_step % self.hparams.clear_cache_int == 0:
        #     torch.cuda.empty_cache()

        # out = self.model(sp_tensor)
        # mask_lab = mapped_labels != self.unknown_label
        # out_novel = self.model.forward_discover(sp_tensor)
                    
        # # loss = self.valid_criterion(out['logits'], mapped_labels.long())
        # # Change mIoU metric
        #         # Change mIoU metric
        # # preds = F.softmax(out['logits'], dim=1)
        # # rc_prob = preds[:,-1]
        # # pred_labels = torch.argmax(preds, dim=1)
        
        # # pred_novel_masks = [(rc_prob > thresh) for thresh in self.novel_thresholds]
        # # pred_labels_list = [pred_labels.clone() for _ in self.novel_thresholds]
        
        # # for i, pred_labels in enumerate(pred_labels_list):
        # #     pred_labels[pred_novel_masks[i]] = self.unknown_label
        
        # # # Transform predictions
        # # gt_labels = real_labels
        # # for key, value in self.label_mapping_inv.items():
        # #     pred_labels[pred_labels == key] = -value
        # # pred_labels = -pred_labels
        
        # # Add point-wise predicitons to cost matrix
        # for pcd_idx in range(pcd_indexes.shape[0]):
        #     for i, threshold in enumerate(self.novel_thresholds):
        #         logits = out['logits'][coords[:,0] == pcd_idx]
        #         novel_logits = out_novel['logits'][coords[:,0] == pcd_idx][:,-self.hparams.num_unlabeled_classes:]
        #         probs = F.softmax(logits, dim=1)
        #         rc_probs = probs[:,-1]
        #         novel_cand_mask = rc_probs > threshold
        #         preds = torch.argmax(probs[~novel_cand_mask], dim=1)
        #         novel_preds = torch.argmax(torch.softmax(novel_logits[novel_cand_mask], dim=1), dim=1)
        #         novel_preds += self.hparams.num_labeled_classes

        #         pcd_labels = real_labels[coords[:,0] == pcd_idx]
        #         total_preds = torch.cat((preds, novel_preds), dim=-1)
        #         total_labels = torch.cat((pcd_labels[~novel_cand_mask], pcd_labels[novel_cand_mask]),dim=-1)
        #         # logits_novel = out_novel['logits'][coords[:,0] == pcd_idx]
        #         # total_logits = torch.cat((logits, logits_novel), dim=1)
        #         # preds = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)

        #         for key, value in self.label_mapping_inv.items():
        #             total_preds[total_preds == key] = -value
        #         total_preds = -total_preds

        #         pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
        #         pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

        #         idxs = np.stack([pcd_preds, pcd_labels], axis=0)
        #         np.add.at(self.cost_matrices[i], tuple(idxs), 1)


        #     # pcd_labels = mapped_labels[coords[:,0] == pcd_idx]
        #     # pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
        #     # for i, pred_labels in enumerate(pred_labels_list):
        #     #     pcd_preds = pred_labels[coords[:,0] == pcd_idx]
        #     #     pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
                
        #     #     # For Strict-Hungarian
        #     #     idxs = np.stack([pcd_preds, pcd_labels], axis=0)
        #     #     np.add.at(self.cost_matrices[i], tuple(idxs), 1) 
        
        # NOTE: one threshold
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())

        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        out_novel = self.model.forward_discover(sp_tensor)
        
        for pcd_idx in range(pcd_indexes.shape[0]):
            logits = out['logits'][coords[:,0] == pcd_idx]
            novel_logits = out_novel['logits'][coords[:,0] == pcd_idx][:,-self.hparams.num_unlabeled_classes:]
            probs = F.softmax(logits, dim=1)
            rc_probs = probs[:,-1]
            novel_cand_mask = rc_probs > 0.2

            preds = torch.argmax(probs[~novel_cand_mask], dim=1)
            novel_preds = torch.argmax(torch.softmax(novel_logits[novel_cand_mask], dim=1), dim=1)
            novel_preds += self.hparams.num_labeled_classes

            pcd_labels = real_labels[coords[:,0] == pcd_idx]
            total_preds = torch.cat((preds, novel_preds), dim=-1)
            total_labels = torch.cat((pcd_labels[~novel_cand_mask], pcd_labels[novel_cand_mask]),dim=-1)
            # logits_novel = out_novel['logits_novel']
            # total_logits = torch.cat((logits, logits_novel), dim=1)
            # preds = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)

            

            pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
            pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            
  
            if self.visualization:
                save_filename = os.path.join(self.save_dir, 
                                        f"{pcd_indexes[pcd_idx]:06}")
                visual_coords = coords[:,1:][coords[:,0]==pcd_idx]
                visual_coords_total = torch.cat((visual_coords[~novel_cand_mask], visual_coords[novel_cand_mask]), dim=0)
                visual_pred = torch.cat((preds, novel_preds), dim=-1)
                
                for key, value in self.label_mapping_inv.items():
                    visual_pred[visual_pred == key] = -value
                visual_pred = -visual_pred

                visual_real_labels = real_labels[coords[:,0]==pcd_idx]
                visual_real_labels_total = torch.cat((visual_real_labels[~novel_cand_mask], visual_real_labels[novel_cand_mask]), dim=-1)
                self.visualize(save_filename, visual_coords_total.detach().cpu().numpy(), visual_pred.detach().cpu().numpy(), visual_real_labels_total.detach().cpu().numpy())

        # if self.visualize:
        #     for i in range(len(pcd_indexes)):
        #         save_filename = os.path.join(self.save_dir, 
        #                                         f"{pcd_indexes[i]:06}")
        #         self.visualize(save_filename, coords[:,1:][coords[:,0]==i].detach().cpu().numpy(), 
        #                             preds[coords[:,0]==i].detach().cpu().numpy(), 
        #                             real_labels[coords[:,0]==i].detach().cpu().numpy())  

        # for pcd_idx in range(pcd_indexes.shape[0]):
        #     pcd_labels = total_labels[coords[:,0] == pcd_idx]
        #     pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
        #     pcd_preds = preds[coords[:,0] == pcd_idx]
        #     pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
            
        #     idxs = np.stack([pcd_preds, pcd_labels], axis=0)
        #     np.add.at(self.cost_matrix, tuple(idxs), 1)

            
            
            #  For Greedy-Hungarian
            # pcd_mask_lab = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]].cpu().numpy()
            # idxs_old = np.stack([pcd_preds[pcd_mask_lab], pcd_labels[pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_old, tuple(idxs_old), 1)
            
            # idxs_new = np.stack([pcd_preds[~pcd_mask_lab], pcd_labels[~pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_new, tuple(idxs_new), 1)

        return
    
    def test_epoch_end(self, _):
        # for m, cost_matrix in enumerate(self.cost_matrices):
        #     # Strict-Hungarian assignment
        #     # conf_base_base = conf_matrix[:-1,:-1].sum()
        #     unknown_real_labels = np.array(self.unknown_real_labels)
        #     conf_matrix = cost_matrix[self.unknown_real_labels, :][:, self.unknown_real_labels]
        #     row_ind, col_ind = linear_sum_assignment(conf_matrix)
            #     include = np.arange(self.hparams.num_classes)
        
        #     for i, j in zip(unknown_real_labels[row_ind], unknown_real_labels[col_ind]):
        #         if i != j:
        #             cost_matrix[:, [i, j]] = cost_matrix[:, [j, i]]
        #             original_index = np.where(include == i)[0][0]
        #             matched_index = np.where(include == j)[0][0]
            
        #             include[original_index], include[matched_index] = include[matched_index], include[original_index]
                        
        #     # ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
        #     # ind = np.vstack(ind).T

        #     # 새로운 conf_matrix에 대해 IoU 계산
        #     # conf_matrix = cost_matrix[:, ind[:, 1]]
        #     # include = np.argsort(ind[:, 1])[:int(self.hparams.num_labeled_classes + 1)]
        #     include = np.arange(self.hparams.num_classes)

        #     iou = self.get_iou(cost_matrix, include)
        #     miou = iou.mean()

        #     # known classes와 unknown (novel) classes에 대한 mIoU

        #     miou_old = iou[self.known_real_labels].mean()
        #     miou_new = iou[self.unknown_real_labels].mean()

        #     _   , output_dict = self.log_metric(miou, iou)
        #     # 각각의 threshold에 따른 mIoU 기록
        #     # self.log(f'valid/miou_threshold_{self.novel_thresholds[i]}', miou, on_step=False, rank_zero_only=True)
        #     self.log(f'threshold_{self.novel_thresholds[m]}/miou/all', miou, on_step=False, rank_zero_only=True)
        #     self.log(f'threshold_{self.novel_thresholds[m]}/miou/known', miou_old, on_step=False, rank_zero_only=True)
        #     self.log(f'threshold_{self.novel_thresholds[m]}/miou/novel', miou_new, on_step=False, rank_zero_only=True)
            
        #     for class_, iou_ in output_dict.items():
                # if class_ == 'all': continue
                # self.log(f'threshold_{self.novel_thresholds[m]}/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
        
        
        #NOTE:  only one threshold
        unknown_real_labels = np.array(self.unknown_real_labels)
        conf_matrix = self.cost_matrix[self.unknown_real_labels, :][:, self.unknown_real_labels]
        row_ind, col_ind = linear_sum_assignment(conf_matrix.max()-conf_matrix)
        include = np.arange(self.hparams.num_classes)
        
        for i, j in zip(unknown_real_labels[row_ind], unknown_real_labels[col_ind]):
            if i != j:
                self.cost_matrix[:, [i, j]] = self.cost_matrix[:, [j, i]]
                original_index = np.where(include == i)[0][0]
                matched_index = np.where(include == j)[0][0]
        
                include[original_index], include[matched_index] = include[matched_index], include[original_index]
        
        # for original, matched in zip(self.unknown_real_labels, matched_indices):
        #     # original과 matched가 다를 때만 교환
        #     if original != matched:
        #         # original과 matched의 위치를 찾아서 swap
        #         original_index = np.where(include == original)[0][0]
        #         matched_index = np.where(include == matched)[0][0]

        # ind = linear_sum_assignment(self.cost_matrix.max() - self.cost_matrix)
        # ind = np.vstack(ind).T
        # conf_matrix = self.cost_matrix[:, ind[:,1]]
        # include_ = np.argsort(ind[:,1])[:self.hparams.num_classes]

        iou = self.get_iou(self.cost_matrix, include)
        
        miou = iou.mean()
        miou_old = iou[self.known_real_labels].mean()
        miou_new = iou[self.unknown_real_labels].mean()

        _, output_dict = self.log_metric(miou, iou)

        self.log(f'valid/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'valid/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
        
        # For Greedy-Hungarian
        # ind_old = linear_sum_assignment(self.cost_matrix_old.max() - self.cost_matrix_old)
        # ind_old = np.vstack(ind_old).T
        
        # conf_matrix_old = self.cost_matrix_old[:, ind_old[:,1]]
        # include_old = np.argsort(ind_old[:,1])[:self.hparams.num_classes]
        # iou_old = self.get_iou(conf_matrix_old, include_old)
        
        # ind_new = linear_sum_assignment(self.cost_matrix_new.max() - self.cost_matrix_new)
        # ind_new = np.vstack(ind_new).T
        
        # conf_matrix_new = self.cost_matrix_new[:, ind_new[:,1]]
        # include_new = np.argsort(ind_new[:,1])[:self.hparams.num_classes]
        # iou_new = self.get_iou(conf_matrix_new, include_new)
        
        # iou_greedy = iou_old.copy()
        # iou_greedy[self.unknown_real_labels] = iou_new[self.unknown_real_labels]
        
        # miou_greedy = iou_greedy.mean()
        # miou_old_greedy = iou_greedy[self.known_real_labels].mean()
        # miou_new_greedy = iou_greedy[self.unknown_real_labels].mean()
        
        # _, output_dict_greedy = self.log_metric(miou_greedy, iou_greedy)
        # self.log(f'valid_greedy/mIoU', miou_greedy, on_step=False, rank_zero_only=True)
        # self.log(f'valid_greedy/mIoU/old', miou_old_greedy, on_step=False, rank_zero_only=True)
        # self.log(f'valid_greedy/mIoU/new', miou_new_greedy, on_step=False, rank_zero_only=True)
        
        # for class_, iou_ in output_dict_greedy.items():
        #     if class_ == 'all': continue
        #     self.log(f'valid_greedy/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
    
    def visualize(self, filename, coords, 
                  preds, 
                  labels):
            color_labels = get_color(labels, self.hparams.dataset_config)
            color_preds = get_color(preds, self.hparams.dataset_config)
            
            write_ply(filename + '-gt.ply', [coords, color_labels], ['x','y','z','red','green','blue'])
            write_ply(filename + '-pd.ply', [coords, color_preds], ['x','y','z','red','green','blue'])
    
    def mix_centroid_features(self, feats, labels):
        with torch.no_grad():
            idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            idx3 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)

            diff_mask_1 = (labels[idx1] != labels[idx2])
            diff_mask_2 = (labels[idx2] != labels[idx3])
            diff_mask_3 = (labels[idx1] != labels[idx3])

            valid_mask = diff_mask_1 & diff_mask_2 & diff_mask_3

            feats1 = feats[idx1][valid_mask]
            feats2 = feats[idx2][valid_mask]
            feats3 = feats[idx3][valid_mask]

            centroid_feats = (feats1 + feats2 + feats3) / 3.0

            labels1, labels2, labels3 = labels[idx1][valid_mask], labels[idx2][valid_mask], labels[idx3][valid_mask]

            num_classes = self.num_classes  

            # one-hot vector 형태로 soft label 생성
            mix_probs = torch.zeros((centroid_feats.shape[0], num_classes), device=self.device)
            mix_probs.scatter_add_(1, labels1.unsqueeze(1), (1.0 / 3.0).unsqueeze(1))
            mix_probs.scatter_add_(1, labels2.unsqueeze(1), (1.0 / 3.0).unsqueeze(1))
            mix_probs.scatter_add_(1, labels3.unsqueeze(1), (1.0 / 3.0).unsqueeze(1))

            mix_probs = mix_probs / mix_probs.sum(dim=1, keepdim=True)

        return centroid_feats, mix_probs
    
    def mix_features(self, feats, labels, mixing_ratio):
        with torch.no_grad():            
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            diff_mask = (labels[permutation_idx1] != labels[permutation_idx2])
                        
            feats1 = feats[permutation_idx1][diff_mask]
            feats2 = feats[permutation_idx2][diff_mask]
            
            labels1, labels2 = labels[permutation_idx1][diff_mask], labels[permutation_idx2][diff_mask]
            
            ### mixing features ###
            if self.beta_sampling:
                mixing_ratio = np.random.beta(self.beta_coeff, self.beta_coeff)
                
            ## For random mixing ratio
            # mixing_ratio = ((self.mixing_ratio_max - self.mixing_ratio_min) * torch.rand(feats1.shape[0], 1) + self.mixing_ratio_min).to(self.device)
            # mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            ## For static mixing ratio
            # mix_coords = mixing_ratio * coords1 + (1- mixing_ratio) * coords2
            # mix_coords.to(torch.int32)
            mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2

            num_classes = self.hparams.num_labeled_classes + 1
            mix_probs = torch.zeros((mix_feats.shape[0], num_classes), device=self.device)
            
            # mix_probs.scatter_add_(1, labels1.unsqueeze(1), (mixing_ratio).unsqueeze(1))
            # mix_probs.scatter_add_(1, labels2.unsqueeze(1), ((1 - mixing_ratio)).unsqueeze(1))
            mix_probs.scatter_add_(1, labels1.unsqueeze(1).long(), torch.full(mix_probs.shape, mixing_ratio, device=self.device))
            mix_probs.scatter_add_(1, labels2.unsqueeze(1).long(), torch.full(mix_probs.shape, 1 - mixing_ratio, device=self.device))
            
            mix_probs = mix_probs / mix_probs.sum(dim=1, keepdim=True)
            
        return mix_feats, mix_probs
    
    def mix_unsup_centroid(self, feats):
        with torch.no_grad():
            # mask unknown labels
            feats = feats
                        
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx3 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
                        
            feats1 = feats[permutation_idx1]
            feats2 = feats[permutation_idx2]
            feats3 = feats[permutation_idx3]
            
            
            mix_feats = (feats1 + feats2 + feats3) / 3.0
            mix_labels = torch.ones(len(mix_feats), dtype= torch.int32, device=self.device)
            mix_labels *= self.unknown_label
            
            
        
        return mix_feats, mix_labels
    
    def mix_unsup_features(self, feats, mixing_ratio):
        with torch.no_grad():
            # mask unknown labels
            feats = feats
                        
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
                        
            feats1 = feats[permutation_idx1]
            feats2 = feats[permutation_idx2]
            
            ### mixing features ###
            if self.beta_sampling:
                mixing_ratio = np.random.beta(self.beta_coeff, self.beta_coeff)
                
            ## For random mixing ratio
            # mixing_ratio = ((self.mixing_ratio_max - self.mixing_ratio_min) * torch.rand(feats1.shape[0], 1) + self.mixing_ratio_min).to(self.device)
            # mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            ## For static mixing ratio
            # mix_coords = mixing_ratio * coords1 + (1- mixing_ratio) * coords2
            # mix_coords.to(torch.int32)
            mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            ## For mixing ratio scheduling
            # mixing_ratio = self.ramp_up_mixing_ratio()
            # mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            # mix_labels = torch.full((mix_feats.shape[0],), self.unknown_label, device=self.device)
            
            # Labels
            # labels1 = labels[permutation_idx1][diff_mask]
            # labels2 = labels[permutation_idx2][diff_mask]
            # mix_labels= labels1 * len(self.known_real_labels) + labels2
            mix_labels = torch.ones(len(mix_feats), dtype= torch.int32, device=self.device)
            mix_labels *= self.unknown_label
        
        return mix_feats, mix_labels


class ExpMergeDiscover_LaserMix_MeanTeacher(ExpMergeDiscover_PolarMix_MeanTeacher):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)        
        self.alpha = 3 # NOTE: number of overcluster
        self.init_method = 'random' # DEFAULT
        self.kmeans = KMeans(n_clusters=self.hparams.num_unlabeled_classes + self.alpha, init_method=self.init_method ,mode='cosine', verbose=0)
        
        # NOTE : multi head ablation studies
        self.head_num = 3 # default 3
        # if self.hparams.resume_checkpoint is not None:
        #     print("##############################Resume Training##############################")
        #     # NCC
        #     self.model_t.encoder.final2 = ME.MinkowskiConvolution(
        #                     self.model_t.encoder.PLANES[7] * self.model_t.encoder.BLOCK.expansion,
        #                     out_channels=self.head_num,
        #                     kernel_size=1,
        #                     bias=True,
        #                     dimension=3)
            
        #     self.model_s.encoder.final2 = ME.MinkowskiConvolution(
        #                     self.model_s.encoder.PLANES[7] * self.model_s.encoder.BLOCK.expansion,
        #                     out_channels=self.head_num,
        #                     kernel_size=1,
        #                     bias=True,
        #                     dimension=3)

        # elif self.hparams.pretrained is not None:
        #     self.load_pretrained(self.model_t, pretrained=self.hparams.pretrained)
        #     # NCC 
        #     self.model_t.encoder.final2 = ME.MinkowskiConvolution(
        #                     self.model_t.encoder.PLANES[7] * self.model_t.encoder.BLOCK.expansion,
        #                     out_channels=self.head_num,
        #                     kernel_size=1,
        #                     bias=True,
        #                     dimension=3)
            
        #     self.load_pretrained(self.model_s, pretrained=self.hparams.pretrained)
        #     # NCC 
        #     self.model_s.encoder.final2 = ME.MinkowskiConvolution(
        #                     self.model_s.encoder.PLANES[7] * self.model_s.encoder.BLOCK.expansion,
        #                     out_channels=self.head_num,
        #                     kernel_size=1,
        #                     bias=True,
        #                     dimension=3)
        
        # else:
        #     raise FileNotFoundError("Checkpoint is needed for Merge Discover Stage")

        self.novel_queue = None
        self.novel_indices= []
        self.novel_queue_batches = 20
        self.use_first_dataloader = False 
        self.use_uncertainty_queue = True
        self.queue_percentage = 0.05
        
        # Novel Candidates Classifier
        self.warmup_epoch = 0 # 1
        self.mixing_ratio_feat = 0.1
        self.label_mixing = False
        self.label_calib = True # Default True
        self.label_training = True # Sup Training
        self.unlabel_mixing = False
        self.unlabel_calib = False
        self.use_all_mse = False # sup data만 할지 unsup 만 할지, default : false
        
        # Novel Classifier
        self.use_sup_logit = True
        self.use_unsup_ncc_loss = True
        
        # For LaserMix
        self.semi_train_cfg = dict(
                freeze_teacher=True, pseudo_thr=0.9, ignore_label=19,
                pitch_angles=[-25, 3], num_areas=[3, 4, 5, 6],
                sup_weight=1, unsup_weight=1,
            )
        
        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 0.1
            self.unsup_calib_coeff = 0.1
            self.mix_unsup_seg_coeff = 1
            self.novel_coeff = 0.1 # 0~1
            self.ncc_coeff = 0.1
            self.mse_coeff = 200
            self.lasermix_coeff = 0.1
            self.sup_novel_coeff = 1

            self.centroid_sampling = False
            self.beta_sampling = True
            self.beta_coeff = 0.5
            
            self.novel_queue_thld = 0.2 # default: 0.2
            self.novel_candidate_thld = 0.2 # default: 0.2
        
        # NOTE: KITTI
        else:            
            self.nusc = None
            self.calib_coeff = 0.05
            self.unsup_calib_coeff = 0.05
            self.mix_unsup_seg_coeff = 0.1
            self.novel_coeff = 0.1
            self.ncc_coeff = 0.1
            self.mse_coeff = 200
            self.lasermix_coeff = 0.1
            self.sup_novel_coeff = 1

            self.centroid_sampling = False
            self.beta_sampling = True
            self.beta_coeff = 0.5
            
            self.novel_queue_thld = 0.2 # default: 0.2
            self.novel_candidate_thld = 0.2 # default: 0.2

        return
    
    def train_dataloader(self):
        lab_dataset = get_dataset_remission(self.hparams.dataset, "lasermix")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
            resize_aug = True,
            output_both_voxel_point=True,
            resize_only_voxel=False, # resize points and voxel 
            # resize_only_voxel=True, # resize only voxel not using lasermix / 
            
        )

        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'lasermix')(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
                resize_aug = False,
                output_both_voxel_point=True,
            )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                lab_dataset.nusc_infos = lab_dataset.nusc_infos[::100]
                unlab_dataset.nusc_infos = unlab_dataset.nusc_infos[::100][:len(lab_dataset)]
            else:
                lab_dataset.num_files = 200
                unlab_dataset.num_files = 200

        lab_dataloader = DataLoader(
            dataset=lab_dataset,
            batch_size=self.hparams.batch_size // 2,  
            collate_fn=collation_fn_lasermix_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,  
        )

        unlab_dataloader = DataLoader(
            dataset=unlab_dataset,
            batch_size=self.hparams.batch_size // 2,  
            collate_fn=collation_fn_lasermix_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,  
        )
        
        self.lab_dataloader_iter = cycle(lab_dataloader)
        self.unlab_dataloader_iter = cycle(unlab_dataloader)
        
        return [lab_dataloader, unlab_dataloader] 

    def test_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, 'labeled')(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                dataset.nusc_infos = dataset.nusc_infos[::100]
            else:
                dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return dataloader

    def mix_transform(self, sup_data, unsup_data, mix_unsup_pseudo_labels):
        # batch_size 4 only
        mix_batch_points = []
        mix_batch_labels = []
        mix_batch_features = []
        

        sup_frame_start_point = 0
        unsup_frame_start_point = 0
        
        sup_coords = sup_data['coords']
        unsup_coords = unsup_data['coords']
        
        if len(sup_data['coords'][:,0].unique()) != len(unsup_data['coords'][:,0].unique()):
            points_sup = sup_coords[sup_coords[:,0] == 0]
            points_unsup = unsup_coords[unsup_coords[:,0] == 0]
            
            features_sup = sup_data['feats'][sup_coords[:,0] == 0]  
            features_unsup = unsup_data['feats'][unsup_coords[:,0] == 0]  
            
            sup_frame_start_point += len(points_sup)
            unsup_frame_start_point += len(points_unsup)
            
            labels = sup_data['mapped_labels'][:sup_frame_start_point]
            pseudo_labels = mix_unsup_pseudo_labels[:unsup_frame_start_point]
            
            points_sup = points_sup[:,1:]
            points_unsup = points_unsup[:,1:]
            
            points_mix1, points_mix2, features_mix1, features_mix2, labels_mix1, labels_mix2 = \
                    self.laser_mix_transform(
                        points_sup=points_sup,
                        points_unsup=points_unsup,
                        features_sup=features_sup,
                        features_unsup=features_unsup,
                        labels=labels,
                        pseudo_labels=pseudo_labels)
            
            mix_batch_points.append(points_mix1)
            mix_batch_points.append(points_mix2)
            
            mix_batch_features.append(features_mix1)
            mix_batch_features.append(features_mix2)
            
            mix_batch_labels.append(labels_mix1)
            mix_batch_labels.append(labels_mix2)
            
            for i in range(len(mix_batch_points)):
                points = mix_batch_points[i].cpu().numpy()
                labels = mix_batch_labels[i].cpu().numpy()
                features = mix_batch_features[i].cpu().numpy()
                mix_batch_points[i] = points
                mix_batch_labels[i] = labels
                mix_batch_features[i] = features
                
            mix_bcoords= []
            for batch_idx, coords in enumerate(mix_batch_points):
                batch_indices = np.full((coords.shape[0],1), batch_idx)
                new_batch = np.hstack((batch_indices, coords))
                mix_bcoords.append(new_batch)
            mix_bcoords = np.concatenate(mix_bcoords, 0)
            mix_bcoords = torch.Tensor(mix_bcoords)
            # mix_batch_points = ME.utils.batched_coordinates(mix_batch_points)
            mix_batch_features = torch.from_numpy(np.concatenate(mix_batch_features,0)).float()
            mix_batch_labels = torch.from_numpy(np.concatenate(mix_batch_labels,0)).int()
            
            return mix_bcoords, mix_batch_features, mix_batch_labels
        else:
            points_sup = sup_coords[sup_coords[:,0] == 0]
            points_unsup = unsup_coords[unsup_coords[:,0] == 0]
            
            features_sup = sup_data['feats'][sup_coords[:,0] == 0]  
            features_unsup = unsup_data['feats'][unsup_coords[:,0] == 0]    
            
            sup_frame_start_point += len(points_sup)
            unsup_frame_start_point += len(points_unsup)

            labels = sup_data['mapped_labels'][:sup_frame_start_point]
            labels1 = sup_data['mapped_labels'][sup_frame_start_point:]

            pseudo_labels = mix_unsup_pseudo_labels[:unsup_frame_start_point]
            pseudo_labels1 = mix_unsup_pseudo_labels[unsup_frame_start_point:]

            points_sup1 = sup_coords[sup_coords[:, 0] != 0]
            points_unsup1 = unsup_coords[unsup_coords[:, 0] != 0]

            features_sup1 = sup_data['feats'][sup_coords[:, 0] != 0]
            features_unsup1 = unsup_data['feats'][unsup_coords[:, 0] != 0]
            
            # Remove batched coordinates
            points_sup = points_sup[:,1:]
            points_unsup = points_unsup[:,1:]
            points_sup1 = points_sup1[:,1:]
            points_unsup1 = points_unsup1[:,1:]

            points_mix1, points_mix2, features_mix1, features_mix2, labels_mix1, labels_mix2 = \
                        self.laser_mix_transform(
                            points_sup=points_sup,
                            points_unsup=points_unsup,
                            features_sup=features_sup,
                            features_unsup=features_unsup,
                            labels=labels,
                            pseudo_labels=pseudo_labels)

            points_mix1_, points_mix2_, features_mix1_, features_mix2_, labels_mix1_, labels_mix2_ = \
                        self.laser_mix_transform(
                            points_sup=points_sup1,
                            points_unsup=points_unsup1,
                            features_sup=features_sup1,
                            features_unsup=features_unsup1,
                            labels=labels1,
                            pseudo_labels=pseudo_labels1)
                    
            mix_batch_points.append(points_mix1)
            mix_batch_points.append(points_mix2)
            
            mix_batch_features.append(features_mix1)
            mix_batch_features.append(features_mix2)
            
            mix_batch_labels.append(labels_mix1)
            mix_batch_labels.append(labels_mix2)
            
            mix_batch_points.append(points_mix1_)
            mix_batch_points.append(points_mix2_)
            
            mix_batch_features.append(features_mix1_)
            mix_batch_features.append(features_mix2_)
            
            mix_batch_labels.append(labels_mix1_)
            mix_batch_labels.append(labels_mix2_)
            
            for i in range(len(mix_batch_points)):
                points = mix_batch_points[i].cpu().numpy()
                labels = mix_batch_labels[i].cpu().numpy()
                features = mix_batch_features[i].cpu().numpy()
                mix_batch_points[i] = points
                mix_batch_labels[i] = labels
                mix_batch_features[i] = features
            
            # breakpoint()
            mix_bcoords= []
            for batch_idx, coords in enumerate(mix_batch_points):
                batch_indices = np.full((coords.shape[0],1), batch_idx)
                new_batch = np.hstack((batch_indices, coords))
                mix_bcoords.append(new_batch)
                
            mix_bcoords = np.concatenate(mix_bcoords, 0)
            mix_bcoords = torch.Tensor(mix_bcoords)
            # mix_batch_points = ME.utils.batched_coordinates(mix_batch_points)
            mix_batch_features = torch.from_numpy(np.concatenate(mix_batch_features,0)).float()
            mix_batch_labels = torch.from_numpy(np.concatenate(mix_batch_labels,0)).int()
            
            return mix_bcoords, mix_batch_features, mix_batch_labels

    def laser_mix_transform(self, points_sup, points_unsup, features_sup, features_unsup, labels, pseudo_labels):
        pitch_angle_down = self.semi_train_cfg['pitch_angles'][0] / 180 * np.pi
        pitch_angle_up = self.semi_train_cfg['pitch_angles'][1] / 180 * np.pi

        rho_sup = torch.sqrt(points_sup[:, 0]**2 + points_sup[:, 1]**2)
        pitch_sup = torch.atan2(points_sup[:, 2], rho_sup)
        pitch_sup = torch.clamp(pitch_sup, pitch_angle_down + 1e-5,
                                pitch_angle_up - 1e-5)

        rho_unsup = torch.sqrt(points_unsup[:, 0]**2 + points_unsup[:, 1]**2)
        pitch_unsup = torch.atan2(points_unsup[:, 2], rho_unsup)
        pitch_unsup = torch.clamp(pitch_unsup, pitch_angle_down + 1e-5,
                                  pitch_angle_up - 1e-5)

        num_areas = np.random.choice(self.semi_train_cfg['num_areas'], size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down,
                                 num_areas + 1)
        
        points_mix1 = []
        points_mix2 = []

        labels_mix1 = []
        labels_mix2 = []
        
        features_mix1 = []
        features_mix2 = []

        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            idx_sup = (pitch_sup > start_angle) & (pitch_sup <= end_angle)
            idx_unsup = (pitch_unsup > start_angle) & (
                pitch_unsup <= end_angle)
            if i % 2 == 0:  # pick from original point cloud
                points_mix1.append(points_sup[idx_sup])
                labels_mix1.append(labels[idx_sup])
                features_mix1.append(features_sup[idx_sup])
                points_mix2.append(points_unsup[idx_unsup])
                labels_mix2.append(pseudo_labels[idx_unsup])
                features_mix2.append(features_unsup[idx_unsup])
            else:  # pickle from mixed point cloud
                points_mix1.append(points_unsup[idx_unsup])
                labels_mix1.append(pseudo_labels[idx_unsup])
                features_mix1.append(features_unsup[idx_unsup])
                points_mix2.append(points_sup[idx_sup])
                labels_mix2.append(labels[idx_sup])
                features_mix2.append(features_sup[idx_sup])

        points_mix1 = torch.cat(points_mix1)
        points_mix2 = torch.cat(points_mix2)
        labels_mix1 = torch.cat(labels_mix1)
        labels_mix2 = torch.cat(labels_mix2)
        features_mix1 = torch.cat(features_mix1)
        features_mix2 = torch.cat(features_mix2)

        return points_mix1, points_mix2, features_mix1, features_mix2, labels_mix1, labels_mix2

    def training_step(self, data, batch_idx, dataloader_idx=0):
        """
        여기서 voxelize랑 lasermix를 동시에 수행해줘야 함 
        그외에 파이프 라인은 기존 과 동일하게 유지하는 것이 중요
        """    
        sup_data = data[0]
        unsup_data = data[1]

        sup_coords = sup_data['voxel']['coords']
        sup_feats = sup_data['voxel']['feats']
        sup_selected_idx = sup_data['voxel']['selected_idx']
        sup_mapped_labels = sup_data['voxel']['mapped_labels']

        sup_data_points = sup_data['points']

        unsup_coords = unsup_data['voxel']['coords']
        unsup_feats = unsup_data['voxel']['feats']
        unsup_inverse_map = unsup_data['voxel']['inverse_maps']
        
        if len(unsup_inverse_map) == 2:
            unsup_inverse_map = torch.cat((unsup_inverse_map[0], unsup_inverse_map[1])) # For batch size 4 only
        else:
            unsup_inverse_map = unsup_inverse_map[0]
            
        unsup_data_points = unsup_data['points']
        unsup_pcd_indexes = unsup_data['voxel']['pcd_indexes']
        
        unsup_pcd_masks = []
        # for i in range(len(pcd_indexes)):
        for i in range(len(unsup_data['voxel']['pcd_indexes'])):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)

        # unsup_coords[:,0] += len(pcd_indexes)
        unsup_coords[:,0] += len(sup_data['voxel']['pcd_indexes'])
        
        coords_cat = torch.cat((sup_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((sup_feats, unsup_feats), axis=0)

        # sup and unsup sparsetensor 
        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        
        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']

        logits_s = out_s['logits']
        feats_s = out_s['feats']
        sup_data_num = len(sup_selected_idx)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]

        # logits_sup_t = logits_t[:sup_data_num, :]
        feats_unsup_s = feats_s[sup_data_num:, :]   
        feats_unsup_t = feats_t[sup_data_num:, :]
        
        # NOTE: labeled dataset
        loss = 0
        
        if self.label_mixing:
            mix_feats_s, mix_targets_s = self.mix_features(feats=feats_sup_s, labels=sup_mapped_labels, mixing_ratio=self.mixing_ratio_feat)
            
            final_layer_kernel = self.model_s.encoder.final.kernel
            final_layer_bias = self.model_s.encoder.final.bias
            final_layer2_kernel = self.model_s.encoder.final2.kernel
            final_layer2_bias = self.model_s.encoder.final2.bias
            
    
            mix_in_normal = torch.matmul(mix_feats_s.detach(), final_layer_kernel) + final_layer_bias
            mix_out_normal = torch.matmul(mix_feats_s.detach(), final_layer2_kernel) + final_layer2_bias
            mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
            mix_logits_s = torch.cat([mix_in_normal, mix_out_normal_2], dim =1)

            logits_mix_sup_s = torch.cat([logits_sup_s, mix_logits_s], dim=0)
            
            targets_sup_s = (
            F.one_hot(
                sup_mapped_labels.long(),
                num_classes=self.hparams.num_labeled_classes + 1,
            )
            .float()
            .to(self.device)
            )
            target_mix_sup_s = torch.cat([targets_sup_s, mix_targets_s])
            mix_seg_loss = self.criterion(logits_mix_sup_s, target_mix_sup_s.long())
            loss += mix_seg_loss
            self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        else:
            if self.label_training:
                final_layer_kernel = self.model_s.encoder.final.kernel
                final_layer_bias = self.model_s.encoder.final.bias
                final_layer2_kernel = self.model_s.encoder.final2.kernel
                final_layer2_bias = self.model_s.encoder.final2.bias
                
                mix_seg_loss = self.criterion(logits_sup_s, sup_mapped_labels.long())
                loss += mix_seg_loss
                self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
            
            else:
                final_layer_kernel = self.model_s.encoder.final.kernel
                final_layer_bias = self.model_s.encoder.final.bias
                final_layer2_kernel = self.model_s.encoder.final2.kernel
                final_layer2_bias = self.model_s.encoder.final2.bias
        
        # NOTE: ema update 해줘야 함 teacher model에 
        # target_tmp = target_mix_sup.clone()
        # breakpoint()
        # index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        # index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

        # preds_dummy = logits_sup.clone()
        # preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        # labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        # labels_dummy[labels_tmp.squeeze() == -1] = -1

        # calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        # loss += calib_loss
    
        # NOTE: unlabeled dataset
        
        logits_unsup_s = logits_s[sup_data_num:, :]
        # feats_unsup_s = feats_s[sup_data_num:, :]
        
        logits_unsup_t = logits_t[sup_data_num:, :]
        # feats_unsup_t = feats_t[sup_data_num:, :]

        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1) 
        # LaserMix MSE Loss
        if self.use_all_mse:
            prob_s = F.softmax(logits_s, dim=1)
            prob_t = F.softmax(logits_t, dim=1)
            mse_loss = F.mse_loss(prob_s, prob_t.detach()) * self.mse_coeff 
            loss += mse_loss
            self.log("train/mse_loss", mse_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("mse", mse_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        else:
            mse_loss = F.mse_loss(prob_unsup_s, prob_unsup_t.detach()) * self.mse_coeff 
            loss += mse_loss
            self.log("train/mse_loss", mse_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("mse", mse_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        
        # Voxel to Points 
        points_max_probs_unsup_t = max_probs_unsup_t[unsup_inverse_map]
        points_target_labels_unsup_t = target_labels_unsup_t[unsup_inverse_map] 

        # # Filter only confident points based on softmax probability
        unconfident_mask = points_max_probs_unsup_t < 0.9
        points_target_labels_unsup_t[unconfident_mask] = -1
        
        # NOTE: Lasermix Augmentation for teacher model
        
        lasermix_points, lasermix_features, lasermix_labels = self.mix_transform(sup_data_points, unsup_data_points, points_target_labels_unsup_t)
        
        if lasermix_points is not None:
            # Make LaserMix points to voxel 
            lasermix_coords, lasermix_unique_map, lasermix_inverse_map = ME.utils.sparse_quantize(
                coordinates=lasermix_points,
                return_index=True,
                return_inverse=True,
                quantization_size=self.hparams.voxel_size,
            )
            
            lasermix_coords = lasermix_coords.to(self.device)
            lasermix_features = lasermix_features[lasermix_unique_map].to(self.device)
            lasermix_labels = lasermix_labels[lasermix_unique_map].to(self.device)
            mix_sp_tensor = ME.SparseTensor(features=lasermix_features.float(), coordinates=lasermix_coords.int())        
                    
            # out_lasermix_t = self.model_t(mix_sp_tensor)
            out_lasermix_s = self.model_s(mix_sp_tensor)
            # lasermix_logits_s = torch.argmax(torch.softmax(out_lasermix_s['logits'], dim=1),dim=1)
            lasermix_loss = self.criterion(out_lasermix_s['logits'], lasermix_labels.long()) * self.lasermix_coeff
            loss += lasermix_loss
            self.log("train/lasermix_loss", lasermix_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("lm", lasermix_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)

        
        if self.unlabel_mixing:
            mix_unsup_feats, mix_unsup_targets = self.mix_features(feats=confident_feats_t, labels=confident_target_label_t, mixing_ratio=self.mixing_ratio_feat)
            
            # pseudo_gt_unsup = torch.argmax(prob_unsup[confident_mask], dim=1)
            pseudo_gt_unsup = confident_target_label_t
            pseudo_target_unsup = (
            F.one_hot(
                pseudo_gt_unsup.to(torch.long),
                num_classes=self.hparams.num_labeled_classes + 1,
            )
            .float()
            .to(self.device)
            )
            
            mix_in_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer_kernel) + final_layer_bias
            mix_out_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer2_kernel) + final_layer2_bias
            mix_out_normal_2_unsup, _ = torch.max(mix_out_normal_unsup, dim=1, keepdim=True)
            mix_unsup_logits = torch.cat([mix_in_normal_unsup, mix_out_normal_2_unsup], dim =1)
            
            logits_mix_unsup_s = torch.cat([logits_unsup_s[confident_mask], mix_unsup_logits], dim=0)
            target_mix_unsup_t = torch.cat([pseudo_target_unsup, mix_unsup_targets], dim=0)
            
            mix_unsup_seg_loss = self.mix_unsup_seg_coeff * self.criterion(logits_mix_unsup_s, target_mix_unsup_t)
            loss += mix_unsup_seg_loss
            self.log("train/unsup_mix_seg_loss", mix_unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("u_mix", mix_unsup_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        if self.unlabel_calib:
            target_tmp = target_mix_unsup_t.clone()

            # Mixing soft labels에서 각 클래스 확률이 존재하는 위치 추출
            mixing_class_indices = torch.nonzero(target_tmp > 0, as_tuple=True)  # (샘플 인덱스, 클래스 인덱스)

            # Breakpoint 이후 수정된 로직
            # index_tmp = torch.arange(0, logits_mix_unsup.shape[0]).unsqueeze(0).to(logits_mix_unsup.device)

            # preds_dummy 초기화
            preds_dummy = logits_mix_unsup_s.clone()

            # Mixing된 soft labels에서 확률 무효화
            preds_dummy[mixing_class_indices[0], mixing_class_indices[1]] = -1e9
            # for idx in torch.unique(mixing_class_indices[0]):  
            #     sample_indices = mixing_class_indices[0] == idx  
            #     class_indices = mixing_class_indices[1][sample_indices]  
            #     preds_dummy[idx, class_indices] = -1e9  

            # labels_dummy 생성
            labels_dummy = torch.ones(logits_mix_unsup_s.shape[0]).long().to(logits_mix_unsup_s.device) * self.unknown_label
            # labels_dummy[target_tmp.squeeze() == -1] = -1

            # Calibration loss 계산
            unsup_calib_loss = self.unsup_calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += unsup_calib_loss
            self.log("train/unsup_calib_loss", unsup_calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("u_cal", unsup_calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        if self.label_mixing and self.label_calib:        
            target_tmp = target_mix_sup_s.clone()

            # Mixing soft labels에서 각 클래스 확률이 존재하는 위치 추출
            mixing_class_indices = torch.nonzero(target_tmp > 0, as_tuple=True)  # (샘플 인덱스, 클래스 인덱스)

            # Breakpoint 이후 수정된 로직
            # index_tmp = torch.arange(0, logits_mix_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
            # preds_dummy 초기화
            preds_dummy = logits_mix_sup_s.clone()

            # Mixing된 soft labels에서 확률 무효화
            preds_dummy[mixing_class_indices[0], mixing_class_indices[1]] = -1e9


            # labels_dummy 생성
            labels_dummy = torch.ones(logits_mix_sup_s.shape[0]).long().to(logits_mix_sup_s.device) * self.unknown_label
            # labels_dummy[target_tmp.squeeze() == -1] = -1

            # Calibration loss 계산
            calib_loss_s = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += calib_loss_s
            self.log("train/sup_calib_loss", calib_loss_s.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("s_cal", calib_loss_s.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        elif self.label_calib:
            target_tmp = sup_mapped_labels.clone()
            # target_tmp[target_tmp==self.unknown_label] = -1
            index_tmp = torch.arange(0, logits_sup_s.shape[0]).unsqueeze(0).to(self.device)
            index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

            # preds_dummy 초기화
            preds_dummy = logits_sup_s.clone()
            preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            
            # labels_dummy 생성
            labels_dummy = torch.ones(logits_sup_s.shape[0]).long().to(self.device) * self.unknown_label
            labels_dummy[target_tmp == self.unknown_label] = -1
            
            # Calibration loss 계산
            calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += calib_loss
            self.log("train/sup_calib_loss", calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("s_cal", calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        """
        TODO
        1. clustering
        - over clustering 해서 metric에 따라 novel classifier 학습
        """
        
        # NOTE: unlabeled dataset
        # threshold = 0.2
        ncc_unsup = prob_unsup_t[:,-1]
        # novel_candidate_mask = ncc_unsup > threshold
        novel_candidate_mask = ncc_unsup > self.novel_candidate_thld
        num_novel_candidate = novel_candidate_mask.sum()
        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup_t[novel_candidate_mask]
            
            if self.novel_queue is not None:
                novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
            
            # if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            if len(novel_cand_feats) > int(self.hparams.num_unlabeled_classes + self.alpha):
                # Run Kmeans clustering to the novel candidate
                # kmeans = self.kmeans.fit(novel_cand_feats.detach().cpu())
                # cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
                # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats.detach())
                cluster_centers = self.kmeans.centroids
                
                # NOTE: base classifier / ncc?, metric: euclidean / cosine?
                cluster_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                # cluster_logits.max(dim=1)[0]
                
                unrealiable_labels = torch.argsort(cluster_logits.max(dim=1)[0], descending=True)[:self.alpha]
                
                # discard queue features
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = torch.isin(cluster_curr_labels, unrealiable_labels)
                
                if (~unrealiable_mask).sum() == 0:
                    return loss
                
                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]
                
                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels
                
                # label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

                # label_mapping을 사용해 reliable_cluster_labels 재정렬
                # reliable_cluster_labels = torch.tensor(
                #     [label_mapping[label.item()] for label in reliable_cluster_labels],
                #     device=reliable_cluster_labels.device
                # )
                
                novel_kernel = self.model_s.encoder.final3.kernel
                novel_bias = self.model_s.encoder.final3.bias
                
                # realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_kernel) + novel_bias
                realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

                unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                try:
                    np.add.at(unsup_cost_matrix, tuple(idxs), 1)
                except:
                    breakpoint()

                row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                for row, col in zip(row_ind, col_ind):
                    mapping_label[col] = row
                mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                
                # mapping = {row: col for row, col in zip(row_ind, col_ind)} # pred 2 label
                # mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

                # for pseudo_label in range(self.hparams.num_unlabeled_classes):
                #     mask_pseudo = realiable_novel_preds == pseudo_label
                #     for j, real_label in enumerate(self.unknown_real_labels):
                #         mask_real = real_labels[mask_unknown] == real_label
                #         cost_matrix[pseudo_label, j] += torch.logical_and(
                #             mask_pseudo, mask_real
                #         ).sum()
                
                # novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
                
                ### (optional)
                if self.use_sup_logit:
                    realiable_novel_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                    mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                
                novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                loss += novel_unsup_loss
                self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_sup_logit:
                    novel_logits_sup = torch.matmul(feats_sup_s, novel_kernel) + novel_bias
                    novel_logits_sup = torch.cat((logits_sup_s[:,:-1], novel_logits_sup), dim=1)
                
                    novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, sup_mapped_labels.long())
                    loss += novel_sup_loss
                    self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_unsup_ncc_loss:
                    ncc_kernel = self.model_s.encoder.final2.kernel
                    ncc_bias = self.model_s.encoder.final2.bias
                    
                    # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                    realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                    realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                    realiable_ncc_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                    ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                    
                    ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                    loss += ncc_unsup_loss
                    self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ## update novel queue
                # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                # self.update_queue(mean_reliable_feats.detach())
                # 
                self.update_queue(reliable_feats.detach())
                self.log("utils/novel_feats", len(reliable_feats), batch_size=len(unsup_pcd_indexes))
                self.log("utils/novel_queue", len(self.novel_queue), batch_size=len(unsup_pcd_indexes))
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, _):
        self._update_ema_variables(momentum=0.01)

    def update_queue(self, feats: torch.Tensor):
        """
        Updates self.queue with the features of the novel points in the current batch

        :param feats: the features for the novel points in the current batch
        """
        feats = feats.detach()
        if not self.use_uncertainty_queue:
            n_feats_to_retain = int(feats.shape[0] * self.queue_percentage)
            mask = torch.randperm(feats.shape[0])[:n_feats_to_retain]
        else:
            n_feats_to_retain = feats.shape[0]
            mask = torch.ones(n_feats_to_retain, device=feats.device, dtype=torch.bool)
        if self.novel_queue is None:
            self.novel_queue = feats[mask]
            self.novel_indices.append(n_feats_to_retain)
            return

        if len(self.novel_indices) < self.novel_queue_batches:
            self.novel_queue = torch.vstack((feats[mask], self.novel_queue))
            self.novel_indices.insert(0, n_feats_to_retain)
        else:
            self.novel_queue = torch.vstack(
                (feats[mask], self.novel_queue[: -self.novel_indices[-1]])
            )
            self.novel_indices.insert(0, n_feats_to_retain)
            del self.novel_indices[-1]

    def on_validation_epoch_start(self):
        D = int(self.hparams.num_classes)
        self.cost_matrix = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_sep = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)
        for i in range(len(self.unknown_real_labels)):
            self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
            
    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())

        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # out = self.model(sp_tensor)
        # out_novel = self.model.forward_discover(sp_tensor)
        out = self.model_t.forward_discover(sp_tensor)
        
        for pcd_idx in range(pcd_indexes.shape[0]):
            logits = out['logits'][coords[:,0] == pcd_idx]
            # novel_logits = out_novel['logits'][coords[:,0] == pcd_idx][:,-self.hparams.num_unlabeled_classes:]
            probs_all = F.softmax(logits, dim=1)
            probs = probs_all[:,:self.hparams.num_labeled_classes]
            probs_novel = probs_all[:, self.hparams.num_labeled_classes:-1]

            # for seperate evaluation
            # rc_probs = probs_all[:,-1]
            # novel_cand_mask = rc_probs > self.novel_candidate_thld
            
            # preds = torch.argmax(probs[~novel_cand_mask], dim=1)
            # novel_preds = torch.argmax(probs_novel[novel_cand_mask], dim=1)
            # novel_preds += self.hparams.num_labeled_classes

            # pcd_labels = real_labels[coords[:,0] == pcd_idx]
            # total_preds = torch.cat((preds, novel_preds), dim=-1)
            # total_labels = torch.cat((pcd_labels[~novel_cand_mask], pcd_labels[novel_cand_mask]), dim=-1)
            # # logits_novel = out_novel['logits_novel']
            # # total_logits = torch.cat((logits, logits_novel), dim=1)
            # # preds = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)

            # for key, value in self.label_mapping_inv.items():
            #     total_preds[total_preds == key] = -value
            # total_preds = -total_preds

            # pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
            # pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

            # idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            # np.add.at(self.cost_matrix_sep, tuple(idxs), 1)
            
            # for classifier only
            total_preds = torch.argmax(probs_all[:,:-1], dim=1)
            total_labels = real_labels[coords[:,0] == pcd_idx]
            
            for key, value in self.label_mapping_inv.items():
                total_preds[total_preds == key] = -value
            total_preds = -total_preds
            
            pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
            pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            
            
            # # For Greedy-Hungarian
            # pcd_mask_lab = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]].cpu().numpy()
            # idxs_old = np.stack([pcd_preds[pcd_mask_lab], pcd_labels[pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_old, tuple(idxs_old), 1)
            
            # idxs_new = np.stack([pcd_preds[~pcd_mask_lab], pcd_labels[~pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_new, tuple(idxs_new), 1)

        return 
    
    def validation_epoch_end(self, valid_step_outputs):
        unknown_real_labels = np.array(self.unknown_real_labels)
        
        conf_matrix = self.cost_matrix[self.unknown_real_labels, :][:, self.unknown_real_labels]
        row_ind, col_ind = linear_sum_assignment(conf_matrix.max() - conf_matrix)
        self.cost_matrix[:,self.unknown_real_labels] = self.cost_matrix[:,np.array(self.unknown_real_labels)[col_ind]]
        include = np.arange(self.hparams.num_classes)
        include[self.unknown_real_labels] = np.array(self.unknown_real_labels)[np.argsort(col_ind)]
        
        
        # For Seperate Evaluation
        # conf_matrix_sep = self.cost_matrix_sep[self.unknown_real_labels, :][:, self.unknown_real_labels]
        # row_ind_sep, col_ind_sep = linear_sum_assignment(conf_matrix_sep.max() - conf_matrix_sep)
        # self.cost_matrix_sep[:,self.unknown_real_labels] = self.cost_matrix_sep[:,np.array(self.unknown_real_labels)[col_ind_sep]]
        
        # include_sep = np.arange(self.hparams.num_classes)
        # include_sep[self.unknown_real_labels] = np.array(self.unknown_real_labels)[np.argsort(col_ind_sep)]
        
        
        # for i, j in zip(unknown_real_labels[row_ind], unknown_real_labels[col_ind]):
        #     if i != j:
        #         self.cost_matrix[:, [i, j]] = self.cost_matrix[:, [j, i]]
        #         original_index = np.where(include == i)[0][0]
        #         matched_index = np.where(include == j)[0][0]
        
        #         include[original_index], include[matched_index] = include[matched_index], include[original_index]
        
        # for i, j in zip(unknown_real_labels[row_ind_sep], unknown_real_labels[col_ind_sep]):
        #     if i != j:
        #         self.cost_matrix_sep[:, [i, j]] = self.cost_matrix_sep[:, [j, i]]
        #         original_index = np.where(include == i)[0][0]
        #         matched_index = np.where(include == j)[0][0]
        
        #         include[original_index], include[matched_index] = include[matched_index], include[original_index]
        
        # for original, matched in zip(self.unknown_real_labels, matched_indices):
        #     # original과 matched가 다를 때만 교환
        #     if original != matched:
        #         # original과 matched의 위치를 찾아서 swap
        #         original_index = np.where(include == original)[0][0]
        #         matched_index = np.where(include == matched)[0][0]

        # ind = linear_sum_assignment(self.cost_matrix.max() - self.cost_matrix)
        # ind = np.vstack(ind).T
        # conf_matrix = self.cost_matrix[:, ind[:,1]]
        # include_ = np.argsort(ind[:,1])[:self.hparams.num_classes]

        iou = self.get_iou(self.cost_matrix, include)
        
        miou = iou.mean()
        miou_old = iou[self.known_real_labels].mean()
        miou_new = iou[self.unknown_real_labels].mean()

        _, output_dict = self.log_metric(miou, iou)

        self.log(f'valid/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'valid/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
            
        # iou_sep = self.get_iou(self.cost_matrix_sep, include_sep)
        
        # miou_sep = iou_sep.mean()
        # miou_old_sep = iou_sep[self.known_real_labels].mean()
        # miou_new_sep = iou_sep[self.unknown_real_labels].mean()

        # _, output_dict_sep = self.log_metric(miou_sep, iou_sep)

        # self.log(f'valid_sep/mIoU', miou_sep, on_step=False, rank_zero_only=True)
        # self.log(f'valid_sep/mIoU/old', miou_old_sep, on_step=False, rank_zero_only=True)
        # self.log(f'valid_sep/mIoU/new', miou_new_sep, on_step=False, rank_zero_only=True)
        
        # for class_, iou_ in output_dict_sep.items():
        #     if class_ == 'all': continue
        #     self.log(f'valid_sep/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        self.visualization = visualization
        D = int(self.hparams.num_classes)
        self.cost_matrix = np.zeros((D, D), dtype=np.int64)
        # self.novel_thresholds = [0.1, 0.2, ]
        # self.novel_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        # self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)
        if self.visualization:
            self.save_dir = os.path.join(self.hparams.log_dir,'visualize')
            os.makedirs(self.save_dir, exist_ok=True)
            
        for i in range(len(self.unknown_real_labels)):
            self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
        breakpoint()
        return
    
    def test_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())

        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # out = self.model(sp_tensor)
        # out_novel = self.model.forward_discover(sp_tensor)
        out = self.model_t.forward_discover(sp_tensor)
        
        for pcd_idx in range(pcd_indexes.shape[0]):
            logits = out['logits'][coords[:,0] == pcd_idx]
            # novel_logits = out_novel['logits'][coords[:,0] == pcd_idx][:,-self.hparams.num_unlabeled_classes:]
            probs_all = F.softmax(logits, dim=1)
            probs = probs_all[:,:self.hparams.num_labeled_classes]
            probs_novel = probs_all[:, self.hparams.num_labeled_classes:-1]
        
            # for classifier only
            total_preds = torch.argmax(probs_all[:,:-1], dim=1)
            total_labels = real_labels[coords[:,0] == pcd_idx]
            
            for key, value in self.label_mapping_inv.items():
                total_preds[total_preds == key] = -value
            total_preds = -total_preds
            
            if self.visualization:
                save_filename = os.path.join(self.save_dir, 
                                            f"{pcd_indexes[pcd_idx]:06}")
                self.visualize(save_filename, coords[:,1:][coords[:,0]==pcd_idx].detach().cpu().numpy(), total_preds.detach().cpu().numpy(), real_labels[coords[:,0] == pcd_idx].detach().cpu().numpy())
            
            pcd_labels = total_labels[inverse_maps[pcd_idx]].cpu().numpy()
            pcd_preds = total_preds[inverse_maps[pcd_idx]].cpu().numpy()

            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            
            


        return 
    
    def test_epoch_end(self, _):
        # for m, cost_matrix in enumerate(self.cost_matrices):
        #     # Strict-Hungarian assignment
        #     # conf_base_base = conf_matrix[:-1,:-1].sum()
        #     unknown_real_labels = np.array(self.unknown_real_labels)
        #     conf_matrix = cost_matrix[self.unknown_real_labels, :][:, self.unknown_real_labels]
        #     row_ind, col_ind = linear_sum_assignment(conf_matrix)
            #     include = np.arange(self.hparams.num_classes)
        
        #     for i, j in zip(unknown_real_labels[row_ind], unknown_real_labels[col_ind]):
        #         if i != j:
        #             cost_matrix[:, [i, j]] = cost_matrix[:, [j, i]]
        #             original_index = np.where(include == i)[0][0]
        #             matched_index = np.where(include == j)[0][0]
            
        #             include[original_index], include[matched_index] = include[matched_index], include[original_index]
                        
        #     # ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
        #     # ind = np.vstack(ind).T

        #     # 새로운 conf_matrix에 대해 IoU 계산
        #     # conf_matrix = cost_matrix[:, ind[:, 1]]
        #     # include = np.argsort(ind[:, 1])[:int(self.hparams.num_labeled_classes + 1)]
        #     include = np.arange(self.hparams.num_classes)

        #     iou = self.get_iou(cost_matrix, include)
        #     miou = iou.mean()

        #     # known classes와 unknown (novel) classes에 대한 mIoU

        #     miou_old = iou[self.known_real_labels].mean()
        #     miou_new = iou[self.unknown_real_labels].mean()

        #     _   , output_dict = self.log_metric(miou, iou)
        #     # 각각의 threshold에 따른 mIoU 기록
        #     # self.log(f'valid/miou_threshold_{self.novel_thresholds[i]}', miou, on_step=False, rank_zero_only=True)
        #     self.log(f'threshold_{self.novel_thresholds[m]}/miou/all', miou, on_step=False, rank_zero_only=True)
        #     self.log(f'threshold_{self.novel_thresholds[m]}/miou/known', miou_old, on_step=False, rank_zero_only=True)
        #     self.log(f'threshold_{self.novel_thresholds[m]}/miou/novel', miou_new, on_step=False, rank_zero_only=True)
            
        #     for class_, iou_ in output_dict.items():
                # if class_ == 'all': continue
                # self.log(f'threshold_{self.novel_thresholds[m]}/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
        
        
        conf_matrix = self.cost_matrix[self.unknown_real_labels, :][:, self.unknown_real_labels]
        row_ind, col_ind = linear_sum_assignment(conf_matrix.max() - conf_matrix)
        breakpoint()
        self.cost_matrix[:,self.unknown_real_labels] = self.cost_matrix[:,np.array(self.unknown_real_labels)[col_ind]]
        include = np.arange(self.hparams.num_classes)
        include[self.unknown_real_labels] = np.array(self.unknown_real_labels)[np.argsort(col_ind)]

        iou = self.get_iou(self.cost_matrix, include)
        miou = iou.mean()
        miou_old = iou[self.known_real_labels].mean()
        miou_new = iou[self.unknown_real_labels].mean()

        _, output_dict = self.log_metric(miou, iou)

        self.log(f'test/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'test/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        self.log(f'test/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'test/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(20, 20))
        plt.title("Confusion Matrix")
        sns.heatmap(self.cost_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=include, yticklabels=include)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        # if hasattr(self, 'class_names'):
        # tick_labels = self.class_names
        # plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=90)
        # plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0)
        # 저장할 경로를 지정 (log_dir 안에 'confusion_matrix.png' 로 저장)
        breakpoint()
        save_path = os.path.join(self.hparams.log_dir, self.hparams.name, 'ours_confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        # For Greedy-Hungarian
        # ind_old = linear_sum_assignment(self.cost_matrix_old.max() - self.cost_matrix_old)
        # ind_old = np.vstack(ind_old).T
        
        # conf_matrix_old = self.cost_matrix_old[:, ind_old[:,1]]
        # include_old = np.argsort(ind_old[:,1])[:self.hparams.num_classes]
        # iou_old = self.get_iou(conf_matrix_old, include_old)
        
        # ind_new = linear_sum_assignment(self.cost_matrix_new.max() - self.cost_matrix_new)
        # ind_new = np.vstack(ind_new).T
        
        # conf_matrix_new = self.cost_matrix_new[:, ind_new[:,1]]
        # include_new = np.argsort(ind_new[:,1])[:self.hparams.num_classes]
        # iou_new = self.get_iou(conf_matrix_new, include_new)
        
        # iou_greedy = iou_old.copy()
        # iou_greedy[self.unknown_real_labels] = iou_new[self.unknown_real_labels]
        
        # miou_greedy = iou_greedy.mean()
        # miou_old_greedy = iou_greedy[self.known_real_labels].mean()
        # miou_new_greedy = iou_greedy[self.unknown_real_labels].mean()
        
        # _, output_dict_greedy = self.log_metric(miou_greedy, iou_greedy)
        # self.log(f'valid_greedy/mIoU', miou_greedy, on_step=False, rank_zero_only=True)
        # self.log(f'valid_greedy/mIoU/old', miou_old_greedy, on_step=False, rank_zero_only=True)
        # self.log(f'valid_greedy/mIoU/new', miou_new_greedy, on_step=False, rank_zero_only=True)
        
        # for class_, iou_ in output_dict_greedy.items():
        #     if class_ == 'all': continue
        #     self.log(f'valid_greedy/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
    
    def log_metric(self, class_IoU, class_all_IoU, classes=None):
        
        # now make a nice dictionary
        output_dict = {}

        # make python variables
        class_IoU = class_IoU.item()
        class_all_IoU = class_all_IoU.flatten().tolist()

        output_dict["all"] = {}
        output_dict["all"]["IoU"] = class_IoU
        for idx, iou in enumerate(class_all_IoU):
            if classes is not None:
                if idx in classes:
                    class_str = self.label_dict[idx]
                    output_dict[class_str] = {}
                    output_dict[class_str]["IoU"] = iou
            else:
                class_str = self.label_dict[idx]
                output_dict[class_str] = {}
                output_dict[class_str]["IoU"] = iou

        mIoU = output_dict["all"]["IoU"]
        return mIoU, output_dict
    
    def get_iou(self, conf_matrix, include):
        conf = conf_matrix.copy().astype(np.double)
        tp = conf.diagonal()
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        
        intersection = tp
        union = tp + fp + fn
        union = np.maximum(union, 1e-15)
        
        iou = intersection.astype(np.double) / union.astype(np.double)
        return iou[include]
    
    def visualize(self, filename, coords, 
                  preds, 
                  labels):
            color_labels = get_color(labels, self.hparams.dataset_config)
            color_preds = get_color(preds, self.hparams.dataset_config)
            
            write_ply(filename + '-gt.ply', [coords, color_labels], ['x','y','z','red','green','blue'])
            write_ply(filename + '-pd.ply', [coords, color_preds], ['x','y','z','red','green','blue'])
       
    def mix_features(self, feats, labels, mixing_ratio):
        with torch.no_grad():            
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            diff_mask = (labels[permutation_idx1] != labels[permutation_idx2])
                        
            feats1 = feats[permutation_idx1][diff_mask]
            feats2 = feats[permutation_idx2][diff_mask]
            
            labels1, labels2 = labels[permutation_idx1][diff_mask], labels[permutation_idx2][diff_mask]
            
            ### mixing features ###
            if self.beta_sampling:
                mixing_ratio = np.random.beta(self.beta_coeff, self.beta_coeff)
                
            ## For random mixing ratio
            # mixing_ratio = ((self.mixing_ratio_max - self.mixing_ratio_min) * torch.rand(feats1.shape[0], 1) + self.mixing_ratio_min).to(self.device)
            # mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            ## For static mixing ratio
            # mix_coords = mixing_ratio * coords1 + (1- mixing_ratio) * coords2
            # mix_coords.to(torch.int32)
            mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2

            num_classes = self.hparams.num_labeled_classes + 1
            mix_probs = torch.zeros((mix_feats.shape[0], num_classes), device=self.device)
            
            # mix_probs.scatter_add_(1, labels1.unsqueeze(1), (mixing_ratio).unsqueeze(1))
            # mix_probs.scatter_add_(1, labels2.unsqueeze(1), ((1 - mixing_ratio)).unsqueeze(1))
            mix_probs.scatter_add_(1, labels1.unsqueeze(1).long(), torch.full(mix_probs.shape, mixing_ratio, device=self.device))
            mix_probs.scatter_add_(1, labels2.unsqueeze(1).long(), torch.full(mix_probs.shape, 1 - mixing_ratio, device=self.device))
            
            mix_probs = mix_probs / mix_probs.sum(dim=1, keepdim=True)
            
        return mix_feats, mix_probs
    
    def mix_unsup_centroid(self, feats):
        with torch.no_grad():
            # mask unknown labels
            feats = feats
                        
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx3 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
                        
            feats1 = feats[permutation_idx1]
            feats2 = feats[permutation_idx2]
            feats3 = feats[permutation_idx3]
            
            
            mix_feats = (feats1 + feats2 + feats3) / 3.0
            mix_labels = torch.ones(len(mix_feats), dtype= torch.int32, device=self.device)
            mix_labels *= self.unknown_label
            
            
        
        return mix_feats, mix_labels
    
    def mix_unsup_features(self, feats, mixing_ratio):
        with torch.no_grad():
            # mask unknown labels
            feats = feats
                        
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
                        
            feats1 = feats[permutation_idx1]
            feats2 = feats[permutation_idx2]
            
            ### mixing features ###
            if self.beta_sampling:
                mixing_ratio = np.random.beta(self.beta_coeff, self.beta_coeff)
                
            ## For random mixing ratio
            # mixing_ratio = ((self.mixing_ratio_max - self.mixing_ratio_min) * torch.rand(feats1.shape[0], 1) + self.mixing_ratio_min).to(self.device)
            # mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            ## For static mixing ratio
            # mix_coords = mixing_ratio * coords1 + (1- mixing_ratio) * coords2
            # mix_coords.to(torch.int32)
            mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            ## For mixing ratio scheduling
            # mixing_ratio = self.ramp_up_mixing_ratio()
            # mix_feats = mixing_ratio * feats1 + (1 - mixing_ratio) * feats2
            
            # mix_labels = torch.full((mix_feats.shape[0],), self.unknown_label, device=self.device)
            
            # Labels
            # labels1 = labels[permutation_idx1][diff_mask]
            # labels2 = labels[permutation_idx2][diff_mask]
            # mix_labels= labels1 * len(self.known_real_labels) + labels2
            mix_labels = torch.ones(len(mix_feats), dtype= torch.int32, device=self.device)
            mix_labels *= self.unknown_label
        
        return mix_feats, mix_labels

class ExpMergeDiscover_LaserMix_MeanTeacher_NCCAdaptive(ExpMergeDiscover_LaserMix_MeanTeacher):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)

        self.adaptive_logit_threshold = nn.Parameter(torch.tensor([0.0]))
        self.alpha = 5
        self.kmeans = KMeans(n_clusters=self.hparams.num_unlabeled_classes + self.alpha, init_method=self.init_method ,mode='cosine', verbose=0)
        # self.adaptive_logit_threshold = nn.Parameter(torch.tensor([-1.4])) # 
        if self.hparams.dataset == 'nuScenes':
            self.threshold_loss_weight = 0.5 # NUSC
        # NOTE: KITTI
        else:            
            self.threshold_loss_weight = 0.2 # KITTI

    def configure_optimizers(self):
        trainable_params = list(self.model_s.parameters()) + [self.adaptive_logit_threshold]
        
        optimizer = optim.SGD(
            params=trainable_params,
            lr=self.hparams.train_lr,
            momentum=self.hparams.momentum_for_optim,
            weight_decay=self.hparams.weight_decay_for_optim,
        )

        if self.hparams.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.epochs,
                warmup_start_lr=self.hparams.min_lr,
                eta_min=self.hparams.min_lr,
            )
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, data, batch_idx, dataloader_idx=0):

        sup_data = data[0]
        unsup_data = data[1]
        sup_coords = sup_data['voxel']['coords']
        sup_feats = sup_data['voxel']['feats']
        sup_selected_idx = sup_data['voxel']['selected_idx']
        sup_mapped_labels = sup_data['voxel']['mapped_labels']
        
        sup_data_points = sup_data['points']
        
        unsup_coords = unsup_data['voxel']['coords']
        unsup_feats = unsup_data['voxel']['feats']
        unsup_inverse_map = unsup_data['voxel']['inverse_maps']
        
        if len(unsup_inverse_map) == 2:
            unsup_inverse_map = torch.cat((unsup_inverse_map[0], unsup_inverse_map[1]))
        else:
            unsup_inverse_map = unsup_inverse_map[0]
        
        unsup_data_points = unsup_data['points']
        unsup_pcd_indexes = unsup_data['voxel']['pcd_indexes']
        
        unsup_pcd_masks = []
        for i in range(len(unsup_data['voxel']['pcd_indexes'])):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)
        unsup_coords[:,0] += len(sup_data['voxel']['pcd_indexes'])
        
        coords_cat = torch.cat((sup_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((sup_feats, unsup_feats), axis=0)
        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        
        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']
        
        logits_s = out_s['logits']
        feats_s = out_s['feats']
        sup_data_num = len(sup_selected_idx)
        
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]
        
        feats_unsup_s = feats_s[sup_data_num:, :]
        feats_unsup_t = feats_t[sup_data_num:, :]
        
        loss = 0
        
        final_layer_kernel = self.model_s.encoder.final.kernel
        final_layer_bias = self.model_s.encoder.final.bias
        # final_layer2_kernel = self.model_s.encoder.final2.kernel
        # final_layer2_bias = self.model_s.encoder.final2.bias
        
        mix_seg_loss = self.criterion(logits_sup_s, sup_mapped_labels.long())
        loss += mix_seg_loss
        self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # MSE loss
        logits_unsup_s = logits_s[sup_data_num:, :]
        logits_unsup_t = logits_t[sup_data_num:, :]
        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)
        
        mse_loss = F.mse_loss(prob_unsup_s, prob_unsup_t.detach()) * self.mse_coeff
        loss += mse_loss
        self.log("train/mse_loss", mse_loss.detach(), on_step=True)

        # NOTE: LaserMix Augmentation
        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        
        points_max_probs_unsup_t = max_probs_unsup_t[unsup_inverse_map]
        points_target_labels_unsup_t = target_labels_unsup_t[unsup_inverse_map] 

        # # Filter only confident points based on softmax probability
        unconfident_mask = points_max_probs_unsup_t < 0.9
        points_target_labels_unsup_t[unconfident_mask] = -1
        
        lasermix_points, lasermix_features, lasermix_labels = self.mix_transform(sup_data_points, unsup_data_points, points_target_labels_unsup_t)
        
        if lasermix_points is not None:
            # Make LaserMix points to voxel 
            lasermix_coords, lasermix_unique_map, lasermix_inverse_map = ME.utils.sparse_quantize(
                coordinates=lasermix_points,
                return_index=True,
                return_inverse=True,
                quantization_size=self.hparams.voxel_size,
            )
            
            lasermix_coords = lasermix_coords.to(self.device)
            lasermix_features = lasermix_features[lasermix_unique_map].to(self.device)
            lasermix_labels = lasermix_labels[lasermix_unique_map].to(self.device)
            mix_sp_tensor = ME.SparseTensor(features=lasermix_features.float(), coordinates=lasermix_coords.int())        
                    
            # out_lasermix_t = self.model_t(mix_sp_tensor)
            out_lasermix_s = self.model_s(mix_sp_tensor)
            # lasermix_logits_s = torch.argmax(torch.softmax(out_lasermix_s['logits'], dim=1),dim=1)
            lasermix_loss = self.criterion(out_lasermix_s['logits'], lasermix_labels.long()) * self.lasermix_coeff
            loss += lasermix_loss
            self.log("train/lasermix_loss", lasermix_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("lm", lasermix_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # NOTE: Calibration Loss 
        if self.label_calib:
            target_tmp = sup_mapped_labels.clone()
            # target_tmp[target_tmp==self.unknown_label] = -1
            index_tmp = torch.arange(0, logits_sup_s.shape[0]).unsqueeze(0).to(self.device)
            index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

            # preds_dummy 초기화
            preds_dummy = logits_sup_s.clone()
            preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            
            # labels_dummy 생성
            labels_dummy = torch.ones(logits_sup_s.shape[0]).long().to(self.device) * self.unknown_label
            labels_dummy[target_tmp == self.unknown_label] = -1
            
            # Calibration loss 계산
            calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += calib_loss
            self.log("train/sup_calib_loss", calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("s_cal", calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # NOTE: ===== Learnable Adaptive Threshold =====

        ncc_logits_sup_s = logits_s[:sup_data_num, -1]
        known_mask = (sup_mapped_labels != self.unknown_label)
        unknown_mask = (sup_mapped_labels == self.unknown_label)
        
        threshold_loss = 0

        # NOTE: Known 클래스 패널티: NCC Logit이 임계값을 넘지 않도록 학습
        if known_mask.sum() > 0:
            known_ncc_logits = ncc_logits_sup_s[known_mask]
            loss_known = torch.mean(torch.relu(known_ncc_logits - self.adaptive_logit_threshold))
            threshold_loss += loss_known

        # NOTE: Unknown 클래스 패널티: NCC Logit이 임계값을 넘도록 학습
        if unknown_mask.sum() > 0:
            unknown_ncc_logits = ncc_logits_sup_s[unknown_mask]
            loss_unknown = torch.mean(torch.relu(self.adaptive_logit_threshold - unknown_ncc_logits))
            threshold_loss += loss_unknown
        
        threshold_loss *= self.threshold_loss_weight
        loss += threshold_loss 
        
        # 학습 진행 상황 모니터링을 위한 로깅
        self.log("train/threshold_loss", threshold_loss.detach(), on_step=True, sync_dist=True)
        self.log("train/adaptive_logit_threshold", self.adaptive_logit_threshold.detach(), on_step=True, sync_dist=True)
        self.log("thr", self.adaptive_logit_threshold.detach(), prog_bar=True, logger=False)
        self.log("thr_l", threshold_loss.detach(), prog_bar=True, logger=False)

        with torch.no_grad():
            ncc_logits_unsup_t = logits_unsup_t[:, -1]
            novel_candidate_mask = ncc_logits_unsup_t > self.adaptive_logit_threshold.detach()
            
        # ncc_unsup = prob_unsup_t[:,-1]
        # # novel_candidate_mask = ncc_unsup > threshold
        # novel_candidate_mask = ncc_unsup > self.novel_candidate_thld
        num_novel_candidate = novel_candidate_mask.sum()
        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup_t[novel_candidate_mask]
            
            if self.novel_queue is not None:
                novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
            
            # if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            if len(novel_cand_feats) > int(self.hparams.num_unlabeled_classes + self.alpha):
                # Run Kmeans clustering to the novel candidate
                # kmeans = self.kmeans.fit(novel_cand_feats.detach().cpu())
                # cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
                # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats.detach())
                cluster_centers = self.kmeans.centroids
                
                # NOTE: base classifier / ncc?, metric: euclidean / cosine?
                cluster_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                # cluster_logits.max(dim=1)[0]
                
                unrealiable_labels = torch.argsort(cluster_logits.max(dim=1)[0], descending=True)[:self.alpha]
                
                # discard queue features
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = torch.isin(cluster_curr_labels, unrealiable_labels)
                
                if (~unrealiable_mask).sum() == 0:
                    return loss
                
                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]
                
                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels
                
                # label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

                # label_mapping을 사용해 reliable_cluster_labels 재정렬
                # reliable_cluster_labels = torch.tensor(
                #     [label_mapping[label.item()] for label in reliable_cluster_labels],
                #     device=reliable_cluster_labels.device
                # )
                
                novel_kernel = self.model_s.encoder.final3.kernel
                novel_bias = self.model_s.encoder.final3.bias
                
                # realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_kernel) + novel_bias
                realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

                unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                try:
                    np.add.at(unsup_cost_matrix, tuple(idxs), 1)
                except:
                    breakpoint()

                row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                for row, col in zip(row_ind, col_ind):
                    mapping_label[col] = row
                mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                
                # mapping = {row: col for row, col in zip(row_ind, col_ind)} # pred 2 label
                # mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

                # for pseudo_label in range(self.hparams.num_unlabeled_classes):
                #     mask_pseudo = realiable_novel_preds == pseudo_label
                #     for j, real_label in enumerate(self.unknown_real_labels):
                #         mask_real = real_labels[mask_unknown] == real_label
                #         cost_matrix[pseudo_label, j] += torch.logical_and(
                #             mask_pseudo, mask_real
                #         ).sum()
                
                # novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
                
                ### (optional)
                if self.use_sup_logit:
                    realiable_novel_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                    mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                
                novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                loss += novel_unsup_loss
                self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_sup_logit:
                    novel_logits_sup = torch.matmul(feats_sup_s, novel_kernel) + novel_bias
                    novel_logits_sup = torch.cat((logits_sup_s[:,:-1], novel_logits_sup), dim=1)
                
                    novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, sup_mapped_labels.long())
                    loss += novel_sup_loss
                    self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_unsup_ncc_loss:
                    ncc_kernel = self.model_s.encoder.final2.kernel
                    ncc_bias = self.model_s.encoder.final2.bias
                    
                    # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                    realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                    realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                    realiable_ncc_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                    ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                    
                    ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                    loss += ncc_unsup_loss
                    self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ## update novel queue
                # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                # self.update_queue(mean_reliable_feats.detach())
                # 
                self.update_queue(reliable_feats.detach())
                self.log("utils/novel_feats", len(reliable_feats), batch_size=len(unsup_pcd_indexes))
                self.log("utils/novel_queue", len(self.novel_queue), batch_size=len(unsup_pcd_indexes))
        
        return loss

class ExpMergeDiscover_LaserMix_MeanTeacher_HybridAdaptive(ExpMergeDiscover_LaserMix_MeanTeacher_NCCAdaptive):

    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)

        # self.adaptive_logit_threshold = nn.Parameter(torch.tensor([0.0]))
        self.adaptive_logit_threshold = nn.Parameter(torch.tensor([-1.4]))
        
        self.threshold_loss_weight = 0.1

        self.threshold_offset = nn.Parameter(torch.tensor([0.0]))

        self.offset_loss_weight = 0.1

        self.margin = 5

    def configure_optimizers(self):
        trainable_params = list(self.model_s.parameters()) + [self.adaptive_logit_threshold] + [self.threshold_offset]
        
        optimizer = optim.SGD(
            params=trainable_params,
            lr=self.hparams.train_lr,
            momentum=self.hparams.momentum_for_optim,
            weight_decay=self.hparams.weight_decay_for_optim,
        )

        if self.hparams.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.hparams.warmup_epochs,
                max_epochs=self.hparams.epochs,
                warmup_start_lr=self.hparams.min_lr,
                eta_min=self.hparams.min_lr,
            )
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, data, batch_idx, dataloader_idx=0):

        sup_data = data[0]
        unsup_data = data[1]
        sup_coords = sup_data['voxel']['coords']
        sup_feats = sup_data['voxel']['feats']
        sup_selected_idx = sup_data['voxel']['selected_idx']
        sup_mapped_labels = sup_data['voxel']['mapped_labels']
        
        sup_data_points = sup_data['points']
        
        unsup_coords = unsup_data['voxel']['coords']
        unsup_feats = unsup_data['voxel']['feats']
        unsup_inverse_map = unsup_data['voxel']['inverse_maps']
        
        if len(unsup_inverse_map) == 2:
            unsup_inverse_map = torch.cat((unsup_inverse_map[0], unsup_inverse_map[1]))
        else:
            unsup_inverse_map = unsup_inverse_map[0]
        
        unsup_data_points = unsup_data['points']
        unsup_pcd_indexes = unsup_data['voxel']['pcd_indexes']
        
        unsup_pcd_masks = []
        for i in range(len(unsup_data['voxel']['pcd_indexes'])):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)
        unsup_coords[:,0] += len(sup_data['voxel']['pcd_indexes'])
        
        coords_cat = torch.cat((sup_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((sup_feats, unsup_feats), axis=0)
        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        
        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']
        
        logits_s = out_s['logits']
        feats_s = out_s['feats']
        sup_data_num = len(sup_selected_idx)
        
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]
        
        # feats_unsup_s = feats_s[sup_data_num:, :]
        feats_unsup_t = feats_t[sup_data_num:, :]
        
        loss = 0
        
        final_layer_kernel = self.model_s.encoder.final.kernel
        final_layer_bias = self.model_s.encoder.final.bias
        # final_layer2_kernel = self.model_s.encoder.final2.kernel
        # final_layer2_bias = self.model_s.encoder.final2.bias
        
        mix_seg_loss = self.criterion(logits_sup_s, sup_mapped_labels.long())
        loss += mix_seg_loss
        self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # MSE loss
        logits_unsup_s = logits_s[sup_data_num:, :]
        logits_unsup_t = logits_t[sup_data_num:, :]
        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)
        
        mse_loss = F.mse_loss(prob_unsup_s, prob_unsup_t.detach()) * self.mse_coeff
        loss += mse_loss
        self.log("train/mse_loss", mse_loss.detach(), on_step=True)

        # NOTE: LaserMix Augmentation
        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        
        points_max_probs_unsup_t = max_probs_unsup_t[unsup_inverse_map]
        points_target_labels_unsup_t = target_labels_unsup_t[unsup_inverse_map] 

        # # Filter only confident points based on softmax probability
        unconfident_mask = points_max_probs_unsup_t < 0.9
        points_target_labels_unsup_t[unconfident_mask] = -1
        
        lasermix_points, lasermix_features, lasermix_labels = self.mix_transform(sup_data_points, unsup_data_points, points_target_labels_unsup_t)
        
        if lasermix_points is not None:
            # Make LaserMix points to voxel 
            lasermix_coords, lasermix_unique_map, lasermix_inverse_map = ME.utils.sparse_quantize(
                coordinates=lasermix_points,
                return_index=True,
                return_inverse=True,
                quantization_size=self.hparams.voxel_size,
            )
            
            lasermix_coords = lasermix_coords.to(self.device)
            lasermix_features = lasermix_features[lasermix_unique_map].to(self.device)
            lasermix_labels = lasermix_labels[lasermix_unique_map].to(self.device)
            mix_sp_tensor = ME.SparseTensor(features=lasermix_features.float(), coordinates=lasermix_coords.int())        
                    
            # out_lasermix_t = self.model_t(mix_sp_tensor)
            out_lasermix_s = self.model_s(mix_sp_tensor)
            # lasermix_logits_s = torch.argmax(torch.softmax(out_lasermix_s['logits'], dim=1),dim=1)
            lasermix_loss = self.criterion(out_lasermix_s['logits'], lasermix_labels.long()) * self.lasermix_coeff
            loss += lasermix_loss
            self.log("train/lasermix_loss", lasermix_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("lm", lasermix_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # NOTE: Calibration Loss 
        if self.label_calib:
            target_tmp = sup_mapped_labels.clone()
            # target_tmp[target_tmp==self.unknown_label] = -1
            index_tmp = torch.arange(0, logits_sup_s.shape[0]).unsqueeze(0).to(self.device)
            index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

            # preds_dummy 초기화
            preds_dummy = logits_sup_s.clone()
            preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            
            # labels_dummy 생성
            labels_dummy = torch.ones(logits_sup_s.shape[0]).long().to(self.device) * self.unknown_label
            labels_dummy[target_tmp == self.unknown_label] = -1
            
            # Calibration loss 계산
            calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += calib_loss
            self.log("train/sup_calib_loss", calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("s_cal", calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # NOTE: ===== Learnable Adaptive Threshold =====
        ncc_logits_sup_s = logits_s[:sup_data_num, -1]
        known_mask = (sup_mapped_labels != self.unknown_label)
        unknown_mask = (sup_mapped_labels == self.unknown_label)
        threshold_loss = 0
        
        # NOTE: Known 클래스 패널티: NCC Logit이 임계값을 넘지 않도록 학습
        if known_mask.sum() > 0:
            known_ncc_logits = ncc_logits_sup_s[known_mask]
            loss_known = torch.mean(torch.relu(known_ncc_logits - self.adaptive_logit_threshold))
            threshold_loss += loss_known

        # NOTE: Unknown 클래스 패널티: NCC Logit이 임계값을 넘도록 학습
        if unknown_mask.sum() > 0:
            unknown_ncc_logits = ncc_logits_sup_s[unknown_mask]
            loss_unknown = torch.mean(torch.relu(self.adaptive_logit_threshold - unknown_ncc_logits))
            threshold_loss += loss_unknown
        
        loss += threshold_loss * self.threshold_loss_weight
        
        # 학습 진행 상황 모니터링을 위한 로깅
        self.log("train/threshold_loss", threshold_loss.detach(), on_step=True, sync_dist=True)
        self.log("train/adaptive_logit_threshold", self.adaptive_logit_threshold.detach(), on_step=True, sync_dist=True)
        self.log("thr", self.adaptive_logit_threshold.detach(), prog_bar=True, logger=False)
        
        with torch.no_grad():
            logits_unsup_t = logits_t[sup_data_num:, :]
            ncc_logits_unsup_t = logits_unsup_t[:, -1]
            final_thld = self.adaptive_logit_threshold.detach() + self.threshold_offset
            novel_candidate_mask = ncc_logits_unsup_t > final_thld
            
        num_novel_candidate = novel_candidate_mask.sum()

        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup_t[novel_candidate_mask]
            
            if num_novel_candidate > self.hparams.num_unlabeled_classes + self.alpha:
                
                # --- '진짜 Novel' 선별 ---
                # novel_cand_feats = feats_unsup_t[sup_data_num:][novel_candidate_mask]

                if self.novel_queue is not None:
                    novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
                
                # ⭐️ 안전장치: 피쳐를 분리하여 이후 연산이 메인 인코더에 영향을 주지 않도록 함
                novel_cand_feats_detached = novel_cand_feats.detach()
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats_detached)
                cluster_centers = self.kmeans.centroids
                
                # 현재 사용하고 계신 'Known 유사도' 기반 필터링 로직                
                cluster_known_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                
                # Known 유사도가 가장 높은 상위 alpha개를 '신뢰할 수 없는' 라벨로 지정
                unreliable_labels = torch.argsort(cluster_known_logits.max(dim=1)[0], descending=True)[:self.alpha]
                
                # 신뢰할 수 없는 클러스터에 속한 포인트들을 식별                
                # queue feature 비우기 
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = torch.isin(cluster_curr_labels, unreliable_labels)

                if (~unrealiable_mask).sum() == 0:
                    return loss
                
                # unlabel의 reliable logits을 통한 threshold 학습
                
                reliable_novel_ncc_logits = ncc_logits_unsup_t[novel_candidate_mask][~unrealiable_mask]
                offset_loss = torch.mean(torch.relu(
                    (self.adaptive_logit_threshold.detach() + self.threshold_offset + self.margin) - reliable_novel_ncc_logits
                )) * self.offset_loss_weight
                loss += offset_loss 
                self.log("train/offset_loss", offset_loss.detach(), on_step=True)
                self.log("throff", offset_loss.detach(), prog_bar=True, logger=False)

                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]

                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels

                if reliable_feats.shape[0] > 0:
                    # --- '진짜 Novel'로 Novel Classifier 학습 ---
                    # (Hungarian Matching, loss 계산 등 부모 클래스의 로직을 여기에 적용)
                    # 이 과정에서 사용되는 모든 피쳐는 `reliable_feats` (detached) 이므로 안전합니다.
                    
                    # 예시: Novel Classifier(`final3`) 학습
                    novel_kernel = self.model_s.encoder.final3.kernel
                    novel_bias = self.model_s.encoder.final3.bias
                    
                    realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                    realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)
                    
                    unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                    idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                    try:
                        np.add.at(unsup_cost_matrix, tuple(idxs), 1)
                    except:
                        breakpoint()

                    row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                    mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                    for row, col in zip(row_ind, col_ind):
                        mapping_label[col] = row
                    mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                    if self.use_sup_logit:
                        realiable_novel_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                        mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                    
                    novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                    loss += novel_unsup_loss
                    self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                    
                    ### (optional)
                    if self.use_sup_logit:
                        novel_logits_sup = torch.matmul(feats_sup_s, novel_kernel) + novel_bias
                        novel_logits_sup = torch.cat((logits_sup_s[:,:-1], novel_logits_sup), dim=1)
                    
                        novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, sup_mapped_labels.long())
                        loss += novel_sup_loss
                        self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                        self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                    
                    ### (optional)
                    if self.use_unsup_ncc_loss:
                        ncc_kernel = self.model_s.encoder.final2.kernel
                        ncc_bias = self.model_s.encoder.final2.bias
                        
                        # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                        realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                        realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                        realiable_ncc_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                        ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                        
                        ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                        loss += ncc_unsup_loss
                        self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                        self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                    
                    ## update novel queue
                    # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                    # self.update_queue(mean_reliable_feats.detach())
                    # 
                    self.update_queue(reliable_feats.detach())
                    self.log("utils/novel_feats", len(reliable_feats), batch_size=len(unsup_pcd_indexes))
                    self.log("utils/novel_queue", len(self.novel_queue), batch_size=len(unsup_pcd_indexes))

        return loss

class ExpMergeDiscover_LaserMix_MeanTeacher_Oracle_threshold(ExpMergeDiscover_LaserMix_MeanTeacher_NCCAdaptive):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.threshold = 0.2052
        return
   
    def test_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, 'labeled')(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                dataset.nusc_infos = dataset.nusc_infos[::100]
            else:
                dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return dataloader
    
    def training_step(self, data, batch_idx, dataloader_idx=0):

        sup_data = data[0]
        unsup_data = data[1]
        sup_coords = sup_data['voxel']['coords']
        sup_feats = sup_data['voxel']['feats']
        sup_selected_idx = sup_data['voxel']['selected_idx']
        sup_mapped_labels = sup_data['voxel']['mapped_labels']
        
        sup_data_points = sup_data['points']
        
        unsup_coords = unsup_data['voxel']['coords']
        unsup_feats = unsup_data['voxel']['feats']
        unsup_inverse_map = unsup_data['voxel']['inverse_maps']
        
        if len(unsup_inverse_map) == 2:
            unsup_inverse_map = torch.cat((unsup_inverse_map[0], unsup_inverse_map[1]))
        else:
            unsup_inverse_map = unsup_inverse_map[0]
        
        unsup_data_points = unsup_data['points']
        unsup_pcd_indexes = unsup_data['voxel']['pcd_indexes']
        
        unsup_pcd_masks = []
        for i in range(len(unsup_data['voxel']['pcd_indexes'])):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)
        unsup_coords[:,0] += len(sup_data['voxel']['pcd_indexes'])
        
        coords_cat = torch.cat((sup_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((sup_feats, unsup_feats), axis=0)
        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        
        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']
        
        logits_s = out_s['logits']
        feats_s = out_s['feats']
        sup_data_num = len(sup_selected_idx)
        
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]
        
        feats_unsup_s = feats_s[sup_data_num:, :]
        feats_unsup_t = feats_t[sup_data_num:, :]
        
        loss = 0
        
        final_layer_kernel = self.model_s.encoder.final.kernel
        final_layer_bias = self.model_s.encoder.final.bias
        # final_layer2_kernel = self.model_s.encoder.final2.kernel
        # final_layer2_bias = self.model_s.encoder.final2.bias
        
        mix_seg_loss = self.criterion(logits_sup_s, sup_mapped_labels.long())
        loss += mix_seg_loss
        self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # MSE loss
        logits_unsup_s = logits_s[sup_data_num:, :]
        logits_unsup_t = logits_t[sup_data_num:, :]
        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)
        
        mse_loss = F.mse_loss(prob_unsup_s, prob_unsup_t.detach()) * self.mse_coeff
        loss += mse_loss
        self.log("train/mse_loss", mse_loss.detach(), on_step=True)

        # NOTE: LaserMix Augmentation
        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        
        points_max_probs_unsup_t = max_probs_unsup_t[unsup_inverse_map]
        points_target_labels_unsup_t = target_labels_unsup_t[unsup_inverse_map] 

        # # Filter only confident points based on softmax probability
        unconfident_mask = points_max_probs_unsup_t < 0.9
        points_target_labels_unsup_t[unconfident_mask] = -1
        
        lasermix_points, lasermix_features, lasermix_labels = self.mix_transform(sup_data_points, unsup_data_points, points_target_labels_unsup_t)
        
        if lasermix_points is not None:
            # Make LaserMix points to voxel 
            lasermix_coords, lasermix_unique_map, lasermix_inverse_map = ME.utils.sparse_quantize(
                coordinates=lasermix_points,
                return_index=True,
                return_inverse=True,
                quantization_size=self.hparams.voxel_size,
            )
            
            lasermix_coords = lasermix_coords.to(self.device)
            lasermix_features = lasermix_features[lasermix_unique_map].to(self.device)
            lasermix_labels = lasermix_labels[lasermix_unique_map].to(self.device)
            mix_sp_tensor = ME.SparseTensor(features=lasermix_features.float(), coordinates=lasermix_coords.int())        
                    
            # out_lasermix_t = self.model_t(mix_sp_tensor)
            out_lasermix_s = self.model_s(mix_sp_tensor)
            # lasermix_logits_s = torch.argmax(torch.softmax(out_lasermix_s['logits'], dim=1),dim=1)
            lasermix_loss = self.criterion(out_lasermix_s['logits'], lasermix_labels.long()) * self.lasermix_coeff
            loss += lasermix_loss
            self.log("train/lasermix_loss", lasermix_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("lm", lasermix_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # NOTE: Calibration Loss 
        if self.label_calib:
            target_tmp = sup_mapped_labels.clone()
            # target_tmp[target_tmp==self.unknown_label] = -1
            index_tmp = torch.arange(0, logits_sup_s.shape[0]).unsqueeze(0).to(self.device)
            index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

            # preds_dummy 초기화
            preds_dummy = logits_sup_s.clone()
            preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            
            # labels_dummy 생성
            labels_dummy = torch.ones(logits_sup_s.shape[0]).long().to(self.device) * self.unknown_label
            labels_dummy[target_tmp == self.unknown_label] = -1
            
            # Calibration loss 계산
            calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
            loss += calib_loss
            self.log("train/sup_calib_loss", calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("s_cal", calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        

        with torch.no_grad():
            ncc_logits_unsup_t = logits_unsup_t[:, -1]
            novel_candidate_mask = ncc_logits_unsup_t > self.threshold
            
        # ncc_unsup = prob_unsup_t[:,-1]
        # # novel_candidate_mask = ncc_unsup > threshold
        # novel_candidate_mask = ncc_unsup > self.novel_candidate_thld
        num_novel_candidate = novel_candidate_mask.sum()
        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup_t[novel_candidate_mask]
            
            if self.novel_queue is not None:
                novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
            
            # if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            if len(novel_cand_feats) > int(self.hparams.num_unlabeled_classes + self.alpha):
                # Run Kmeans clustering to the novel candidate
                # kmeans = self.kmeans.fit(novel_cand_feats.detach().cpu())
                # cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
                # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats.detach())
                cluster_centers = self.kmeans.centroids
                
                # NOTE: base classifier / ncc?, metric: euclidean / cosine?
                cluster_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                # cluster_logits.max(dim=1)[0]
                
                unrealiable_labels = torch.argsort(cluster_logits.max(dim=1)[0], descending=True)[:self.alpha]
                
                # discard queue features
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = torch.isin(cluster_curr_labels, unrealiable_labels)
                
                if (~unrealiable_mask).sum() == 0:
                    return loss
                
                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]
                
                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels
                
                # label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

                # label_mapping을 사용해 reliable_cluster_labels 재정렬
                # reliable_cluster_labels = torch.tensor(
                #     [label_mapping[label.item()] for label in reliable_cluster_labels],
                #     device=reliable_cluster_labels.device
                # )
                
                novel_kernel = self.model_s.encoder.final3.kernel
                novel_bias = self.model_s.encoder.final3.bias
                
                # realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_kernel) + novel_bias
                realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

                unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                try:
                    np.add.at(unsup_cost_matrix, tuple(idxs), 1)
                except:
                    breakpoint()

                row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                for row, col in zip(row_ind, col_ind):
                    mapping_label[col] = row
                mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                
                # mapping = {row: col for row, col in zip(row_ind, col_ind)} # pred 2 label
                # mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

                # for pseudo_label in range(self.hparams.num_unlabeled_classes):
                #     mask_pseudo = realiable_novel_preds == pseudo_label
                #     for j, real_label in enumerate(self.unknown_real_labels):
                #         mask_real = real_labels[mask_unknown] == real_label
                #         cost_matrix[pseudo_label, j] += torch.logical_and(
                #             mask_pseudo, mask_real
                #         ).sum()
                
                # novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
                
                ### (optional)
                if self.use_sup_logit:
                    realiable_novel_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                    mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                
                novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                loss += novel_unsup_loss
                self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_sup_logit:
                    novel_logits_sup = torch.matmul(feats_sup_s, novel_kernel) + novel_bias
                    novel_logits_sup = torch.cat((logits_sup_s[:,:-1], novel_logits_sup), dim=1)
                
                    novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, sup_mapped_labels.long())
                    loss += novel_sup_loss
                    self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_unsup_ncc_loss:
                    ncc_kernel = self.model_s.encoder.final2.kernel
                    ncc_bias = self.model_s.encoder.final2.bias
                    
                    # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                    realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                    realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                    realiable_ncc_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                    ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                    
                    ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                    loss += ncc_unsup_loss
                    self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ## update novel queue
                # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                # self.update_queue(mean_reliable_feats.detach())
                # 
                self.update_queue(reliable_feats.detach())
                self.log("utils/novel_feats", len(reliable_feats), batch_size=len(unsup_pcd_indexes))
                self.log("utils/novel_queue", len(self.novel_queue), batch_size=len(unsup_pcd_indexes))
        
        return loss
    
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        self.visualization = visualization
        D = int(self.hparams.num_classes)
        self.my_known_sum = 0.0
        self.my_known_cnt = 0
        self.my_novel_sum = 0.0
        self.my_novel_cnt = 0

        self.msp_known_sum = 0.0
        self.msp_known_cnt = 0
        self.msp_novel_sum = 0.0
        self.msp_novel_cnt = 0
        if self.visualization:
            self.save_dir = os.path.join(self.hparams.log_dir,'visualize')
            os.makedirs(self.save_dir, exist_ok=True)
            
        
        # for i in range(len(self.unknown_real_labels)):
        #     self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
        
        return
    
    def test_step(self, data, _):
        coords, feats, _, _, mapped_labels, _, _ = data
        
        # 1. 마스크 준비
        novel_mask = (mapped_labels == self.unknown_label)
        known_mask = ~novel_mask

        # 2. 모델 예측
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()
        out = self.model(sp_tensor)

        # --- 3. 이상치 점수 계산 ---
        # 3-1. 제안 방식의 이상치 점수 (Novel Candidate Logit)
        my_anomaly_scores = out['logits'][:, -1]

        # 3-2. MSP 방식의 이상치 점수
        known_logits = out['logits'][:, :-1]
        known_probs = torch.softmax(known_logits, dim=1)
        msp_scores, _ = torch.max(known_probs, dim=1)
        msp_anomaly_scores = 1 - msp_scores

        # --- 4. 통계 누적 ---
        # 각 마스크에 해당하는 점수들을 추출하여 합계와 개수를 누적합니다.
        # 제안 방식
        my_known_scores = my_anomaly_scores[known_mask]
        my_novel_scores = my_anomaly_scores[novel_mask]
        if len(my_known_scores) > 0:
            self.my_known_sum += my_known_scores.sum()
            self.my_known_cnt += len(my_known_scores)
        if len(my_novel_scores) > 0:
            self.my_novel_sum += my_novel_scores.sum()
            self.my_novel_cnt += len(my_novel_scores)
            
        # MSP 방식
        msp_known_scores = msp_anomaly_scores[known_mask]
        msp_novel_scores = msp_anomaly_scores[novel_mask]
        if len(msp_known_scores) > 0:
            self.msp_known_sum += msp_known_scores.sum()
            self.msp_known_cnt += len(msp_known_scores)
        if len(msp_novel_scores) > 0:
            self.msp_novel_sum += msp_novel_scores.sum()
            self.msp_novel_cnt += len(msp_novel_scores)

        return

    def test_epoch_end(self, _):
        print("\n" + "="*40)
        print("Ablation Study Statistics")
        print("="*40)

        # --- 제안 방식(My Method) 통계 ---
        my_known_mean = self.my_known_sum / self.my_known_cnt
        my_novel_mean = self.my_novel_sum / self.my_novel_cnt
        print(f"[My Method] Known scores mean: {my_known_mean:.4f}")
        print(f"[My Method] Novel scores mean: {my_novel_mean:.4f}")
        
        # --- MSP 베이스라인 통계 ---
        msp_known_mean = self.msp_known_sum / self.msp_known_cnt
        msp_novel_mean = self.msp_novel_sum / self.msp_novel_cnt
        print(f"[MSP Method] Known scores mean: {msp_known_mean:.4f}")
        print(f"[MSP Method] Novel scores mean: {msp_novel_mean:.4f}")
        print("-"*40)

        # --- 최종 Oracle 임계값 계산 ---
        # '나의 이상치 점수'에 대한 Oracle 임계값
        oracle_threshold = (my_known_mean + my_novel_mean) / 2
        print(f"🎯 Recommended Oracle Fixed Threshold: {oracle_threshold:.4f}")
        print("="*40 + "\n")

class ExpMergeDiscover_LaserMix_MeanTeacher_MSP_threshold(ExpMergeDiscover_LaserMix_MeanTeacher_NCCAdaptive):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.msp_threshold = 0.0883
        return
   
    def test_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, 'labeled')(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                dataset.nusc_infos = dataset.nusc_infos[::100]
            else:
                dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return dataloader
    
    def training_step(self, data, batch_idx, dataloader_idx=0):

        sup_data = data[0]
        unsup_data = data[1]
        sup_coords = sup_data['voxel']['coords']
        sup_feats = sup_data['voxel']['feats']
        sup_selected_idx = sup_data['voxel']['selected_idx']
        sup_mapped_labels = sup_data['voxel']['mapped_labels']
        
        sup_data_points = sup_data['points']
        
        unsup_coords = unsup_data['voxel']['coords']
        unsup_feats = unsup_data['voxel']['feats']
        unsup_inverse_map = unsup_data['voxel']['inverse_maps']
        
        if len(unsup_inverse_map) == 2:
            unsup_inverse_map = torch.cat((unsup_inverse_map[0], unsup_inverse_map[1]))
        else:
            unsup_inverse_map = unsup_inverse_map[0]
        
        unsup_data_points = unsup_data['points']
        unsup_pcd_indexes = unsup_data['voxel']['pcd_indexes']
        
        unsup_pcd_masks = []
        for i in range(len(unsup_data['voxel']['pcd_indexes'])):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)
        unsup_coords[:,0] += len(sup_data['voxel']['pcd_indexes'])
        
        coords_cat = torch.cat((sup_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((sup_feats, unsup_feats), axis=0)
        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        
        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']
        
        logits_s = out_s['logits']
        feats_s = out_s['feats']
        sup_data_num = len(sup_selected_idx)
        
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]
        
        feats_unsup_s = feats_s[sup_data_num:, :]
        feats_unsup_t = feats_t[sup_data_num:, :]
        
        loss = 0
        
        final_layer_kernel = self.model_s.encoder.final.kernel
        final_layer_bias = self.model_s.encoder.final.bias
        # final_layer2_kernel = self.model_s.encoder.final2.kernel
        # final_layer2_bias = self.model_s.encoder.final2.bias
        
        mix_seg_loss = self.criterion(logits_sup_s, sup_mapped_labels.long())
        loss += mix_seg_loss
        self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # MSE loss
        logits_unsup_s = logits_s[sup_data_num:, :]
        logits_unsup_t = logits_t[sup_data_num:, :]
        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)
        
        mse_loss = F.mse_loss(prob_unsup_s, prob_unsup_t.detach()) * self.mse_coeff
        loss += mse_loss
        self.log("train/mse_loss", mse_loss.detach(), on_step=True)

        # NOTE: LaserMix Augmentation
        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        
        points_max_probs_unsup_t = max_probs_unsup_t[unsup_inverse_map]
        points_target_labels_unsup_t = target_labels_unsup_t[unsup_inverse_map] 

        # # Filter only confident points based on softmax probability
        unconfident_mask = points_max_probs_unsup_t < 0.9
        points_target_labels_unsup_t[unconfident_mask] = -1
        
        lasermix_points, lasermix_features, lasermix_labels = self.mix_transform(sup_data_points, unsup_data_points, points_target_labels_unsup_t)
        
        if lasermix_points is not None:
            # Make LaserMix points to voxel 
            lasermix_coords, lasermix_unique_map, lasermix_inverse_map = ME.utils.sparse_quantize(
                coordinates=lasermix_points,
                return_index=True,
                return_inverse=True,
                quantization_size=self.hparams.voxel_size,
            )
            
            lasermix_coords = lasermix_coords.to(self.device)
            lasermix_features = lasermix_features[lasermix_unique_map].to(self.device)
            lasermix_labels = lasermix_labels[lasermix_unique_map].to(self.device)
            mix_sp_tensor = ME.SparseTensor(features=lasermix_features.float(), coordinates=lasermix_coords.int())        
                    
            # out_lasermix_t = self.model_t(mix_sp_tensor)
            out_lasermix_s = self.model_s(mix_sp_tensor)
            # lasermix_logits_s = torch.argmax(torch.softmax(out_lasermix_s['logits'], dim=1),dim=1)
            lasermix_loss = self.criterion(out_lasermix_s['logits'], lasermix_labels.long()) * self.lasermix_coeff
            loss += lasermix_loss
            self.log("train/lasermix_loss", lasermix_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("lm", lasermix_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # # NOTE: Calibration Loss 
        # if self.label_calib:
        #     target_tmp = sup_mapped_labels.clone()
        #     # target_tmp[target_tmp==self.unknown_label] = -1
        #     index_tmp = torch.arange(0, logits_sup_s.shape[0]).unsqueeze(0).to(self.device)
        #     index_tmp = torch.cat([index_tmp, target_tmp.unsqueeze(0)], dim=0)

        #     # preds_dummy 초기화
        #     preds_dummy = logits_sup_s.clone()
        #     preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
            
        #     # labels_dummy 생성
        #     labels_dummy = torch.ones(logits_sup_s.shape[0]).long().to(self.device) * self.unknown_label
        #     labels_dummy[target_tmp == self.unknown_label] = -1
            
        #     # Calibration loss 계산
        #     calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        #     loss += calib_loss
        #     self.log("train/sup_calib_loss", calib_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        #     self.log("s_cal", calib_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        with torch.no_grad():
            unsup_known_logits = logits_unsup_t[:, :-1]
            unsup_known_probs = torch.softmax(unsup_known_logits, dim=1)
            msp_scores, _  = torch.max(unsup_known_probs, dim=1)
            msp_anomaly_scores = 1 - msp_scores
            novel_candidate_mask = msp_anomaly_scores > self.msp_threshold
            
        # ncc_unsup = prob_unsup_t[:,-1]
        # # novel_candidate_mask = ncc_unsup > threshold
        # novel_candidate_mask = ncc_unsup > self.novel_candidate_thld
        num_novel_candidate = novel_candidate_mask.sum()
        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup_t[novel_candidate_mask]
            
            if self.novel_queue is not None:
                novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
            
            # if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            if len(novel_cand_feats) > int(self.hparams.num_unlabeled_classes + self.alpha):
                # Run Kmeans clustering to the novel candidate
                # kmeans = self.kmeans.fit(novel_cand_feats.detach().cpu())
                # cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
                # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats.detach())
                cluster_centers = self.kmeans.centroids
                
                # NOTE: base classifier / ncc?, metric: euclidean / cosine?
                cluster_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                # cluster_logits.max(dim=1)[0]
                
                unrealiable_labels = torch.argsort(cluster_logits.max(dim=1)[0], descending=True)[:self.alpha]
                
                # discard queue features
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = torch.isin(cluster_curr_labels, unrealiable_labels)
                
                if (~unrealiable_mask).sum() == 0:
                    return loss
                
                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]
                
                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels
                
                # label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

                # label_mapping을 사용해 reliable_cluster_labels 재정렬
                # reliable_cluster_labels = torch.tensor(
                #     [label_mapping[label.item()] for label in reliable_cluster_labels],
                #     device=reliable_cluster_labels.device
                # )
                
                novel_kernel = self.model_s.encoder.final3.kernel
                novel_bias = self.model_s.encoder.final3.bias
                
                # realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_kernel) + novel_bias
                realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

                unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                try:
                    np.add.at(unsup_cost_matrix, tuple(idxs), 1)
                except:
                    breakpoint()

                row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                for row, col in zip(row_ind, col_ind):
                    mapping_label[col] = row
                mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                
                # mapping = {row: col for row, col in zip(row_ind, col_ind)} # pred 2 label
                # mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

                # for pseudo_label in range(self.hparams.num_unlabeled_classes):
                #     mask_pseudo = realiable_novel_preds == pseudo_label
                #     for j, real_label in enumerate(self.unknown_real_labels):
                #         mask_real = real_labels[mask_unknown] == real_label
                #         cost_matrix[pseudo_label, j] += torch.logical_and(
                #             mask_pseudo, mask_real
                #         ).sum()
                
                # novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
                
                ### (optional)
                if self.use_sup_logit:
                    realiable_novel_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                    mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                
                novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                loss += novel_unsup_loss
                self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_sup_logit:
                    novel_logits_sup = torch.matmul(feats_sup_s, novel_kernel) + novel_bias
                    novel_logits_sup = torch.cat((logits_sup_s[:,:-1], novel_logits_sup), dim=1)
                
                    novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, sup_mapped_labels.long())
                    loss += novel_sup_loss
                    self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                # if self.use_unsup_ncc_loss:
                #     ncc_kernel = self.model_s.encoder.final2.kernel
                #     ncc_bias = self.model_s.encoder.final2.bias
                    
                #     # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                #     realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                #     realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                #     realiable_ncc_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                #     ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                    
                #     ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                #     loss += ncc_unsup_loss
                #     self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                #     self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ## update novel queue
                # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                # self.update_queue(mean_reliable_feats.detach())
                # 
                self.update_queue(reliable_feats.detach())
                self.log("utils/novel_feats", len(reliable_feats), batch_size=len(unsup_pcd_indexes))
                self.log("utils/novel_queue", len(self.novel_queue), batch_size=len(unsup_pcd_indexes))
        
        return loss
    
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        self.visualization = visualization
        D = int(self.hparams.num_classes)
        self.my_known_sum = 0.0
        self.my_known_cnt = 0
        self.my_novel_sum = 0.0
        self.my_novel_cnt = 0

        self.msp_known_sum = 0.0
        self.msp_known_cnt = 0
        self.msp_novel_sum = 0.0
        self.msp_novel_cnt = 0
        if self.visualization:
            self.save_dir = os.path.join(self.hparams.log_dir,'visualize')
            os.makedirs(self.save_dir, exist_ok=True)
            
        
        # for i in range(len(self.unknown_real_labels)):
        #     self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
        
        return
    
    def test_step(self, data, _):
        coords, feats, _, _, mapped_labels, _, _ = data
        
        # 1. 마스크 준비
        novel_mask = (mapped_labels == self.unknown_label)
        known_mask = ~novel_mask

        # 2. 모델 예측
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()
        out = self.model(sp_tensor)

        # --- 3. 이상치 점수 계산 ---
        # 3-1. 제안 방식의 이상치 점수 (Novel Candidate Logit)
        my_anomaly_scores = out['logits'][:, -1]

        # 3-2. MSP 방식의 이상치 점수
        known_logits = out['logits'][:, :-1]
        known_probs = torch.softmax(known_logits, dim=1)
        msp_scores, _ = torch.max(known_probs, dim=1)
        msp_anomaly_scores = 1 - msp_scores

        # --- 4. 통계 누적 ---
        # 각 마스크에 해당하는 점수들을 추출하여 합계와 개수를 누적합니다.
        # 제안 방식
        my_known_scores = my_anomaly_scores[known_mask]
        my_novel_scores = my_anomaly_scores[novel_mask]
        if len(my_known_scores) > 0:
            self.my_known_sum += my_known_scores.sum()
            self.my_known_cnt += len(my_known_scores)
        if len(my_novel_scores) > 0:
            self.my_novel_sum += my_novel_scores.sum()
            self.my_novel_cnt += len(my_novel_scores)
            
        # MSP 방식
        msp_known_scores = msp_anomaly_scores[known_mask]
        msp_novel_scores = msp_anomaly_scores[novel_mask]
        if len(msp_known_scores) > 0:
            self.msp_known_sum += msp_known_scores.sum()
            self.msp_known_cnt += len(msp_known_scores)
        if len(msp_novel_scores) > 0:
            self.msp_novel_sum += msp_novel_scores.sum()
            self.msp_novel_cnt += len(msp_novel_scores)

        return

    def test_epoch_end(self, _):
        print("\n" + "="*40)
        print("Ablation Study Statistics")
        print("="*40)

        # --- 제안 방식(My Method) 통계 ---
        my_known_mean = self.my_known_sum / self.my_known_cnt
        my_novel_mean = self.my_novel_sum / self.my_novel_cnt
        print(f"[My Method] Known scores mean: {my_known_mean:.4f}")
        print(f"[My Method] Novel scores mean: {my_novel_mean:.4f}")
        
        # --- MSP 베이스라인 통계 ---
        msp_known_mean = self.msp_known_sum / self.msp_known_cnt
        msp_novel_mean = self.msp_novel_sum / self.msp_novel_cnt
        print(f"[MSP Method] Known scores mean: {msp_known_mean:.4f}")
        print(f"[MSP Method] Novel scores mean: {msp_novel_mean:.4f}")
        print("-"*40)

        # --- 최종 Oracle 임계값 계산 ---
        # '나의 이상치 점수'에 대한 Oracle 임계값
        oracle_threshold = (my_known_mean + my_novel_mean) / 2
        print(f"🎯 Recommended Oracle Fixed Threshold: {oracle_threshold:.4f}")
        print("="*40 + "\n")


class ExpMergeDiscover_LaserMix_LiON_MeanTeacher(ExpMergeDiscover_LaserMix_MeanTeacher):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)        
        self.alpha = 5
        self.kmeans = KMeans(n_clusters=self.hparams.num_unlabeled_classes + self.alpha, mode='cosine', verbose=0)

        self.novel_queue = None
        self.novel_indices= []
        self.novel_queue_batches = 20
        self.use_first_dataloader = False 
        self.use_uncertainty_queue = True
        self.queue_percentage = 0.05
        
        # Novel Candidates Classifier
        self.warmup_epoch = 0 # 1
        self.mixing_ratio_feat = 0.1
        self.label_mixing = False
        self.label_calib = False # Default True
        self.unlabel_mixing = False
        self.unlabel_calib = False
        
        self.label_training = True # Sup Training
        

        # Novel Classifier
        self.use_sup_logit = True
        self.use_unsup_ncc_loss = True
        
        # For LaserMix
        self.semi_train_cfg = dict(
                freeze_teacher=True, pseudo_thr=0.9, ignore_label=19,
                pitch_angles=[-25, 3], num_areas=[3, 4, 5, 6],
                sup_weight=1, unsup_weight=1,
            )
        
        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 0.1
            self.unsup_calib_coeff = 0.1
            self.mix_unsup_seg_coeff = 1
            self.novel_coeff = 0.1 # 0~1
            self.sup_novel_coeff = 1
            self.ncc_coeff = 0.1
            self.mse_coeff = 200
            self.lasermix_coeff = 0.1

            self.centroid_sampling = False
            self.beta_sampling = True
            self.beta_coeff = 0.5
            
            self.novel_queue_thld = 0.2 # default: 0.2
            self.novel_candidate_thld = 0.2 # default: 0.2
        
        # NOTE: KITTI
        else:            
            self.nusc = None
            self.calib_coeff = 0.05 
            self.unsup_calib_coeff = 0.05
            self.mix_unsup_seg_coeff = 0.1
            self.novel_coeff = 0.1
            self.ncc_coeff = 0.1
            self.mse_coeff = 200
            self.lasermix_coeff = 0.1
            self.sup_novel_coeff = 1

            self.centroid_sampling = False
            self.beta_sampling = True
            self.beta_coeff = 0.5
            
            self.novel_queue_thld = 0.2 # default: 0.2
            self.novel_candidate_thld = 0.2 # default: 0.2

        return
    
    def on_train_start(self):
        # TODO: LiON loss porting
        self.gambler_loss = Gambler(reward=[4.5],
                                    device=self.device,
                                    valid_class_num=16, novel_class_num=self.hparams.num_unlabeled_classes, unknown_cls_idx=1,
                                    novel_class_list = self.unknown_real_labels)
        return 
    
    def training_step(self, data, batch_idx, dataloader_idx=0):
        sup_data = data[0]
        unsup_data = data[1]

        sup_coords = sup_data['voxel']['coords']
        sup_feats = sup_data['voxel']['feats']
        sup_selected_idx = sup_data['voxel']['selected_idx']
        sup_mapped_labels = sup_data['voxel']['mapped_labels']

        sup_data_points = sup_data['points']

        unsup_coords = unsup_data['voxel']['coords']
        unsup_feats = unsup_data['voxel']['feats']
        unsup_inverse_map = unsup_data['voxel']['inverse_maps']
        
        if len(unsup_inverse_map) == 2:
            unsup_inverse_map = torch.cat((unsup_inverse_map[0], unsup_inverse_map[1])) # For batch size 4 only
        else:
            unsup_inverse_map = unsup_inverse_map[0]
            
        unsup_data_points = unsup_data['points']
        unsup_pcd_indexes = unsup_data['voxel']['pcd_indexes']
        
        unsup_pcd_masks = []
        # for i in range(len(pcd_indexes)):
        for i in range(len(unsup_data['voxel']['pcd_indexes'])):
            unsup_pcd_masks.append(unsup_coords[:,0] == i)

        # unsup_coords[:,0] += len(pcd_indexes)
        unsup_coords[:,0] += len(sup_data['voxel']['pcd_indexes'])
        
        coords_cat = torch.cat((sup_coords, unsup_coords), axis=0)
        feats_cat = torch.cat((sup_feats, unsup_feats), axis=0)

        # sup and unsup sparsetensor 
        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        min_coord = torch.abs(sp_tensor.coordinates.min(dim=0).values)
        min_coord = min_coord.int() # remove batch index
        den_tensor = sp_tensor.dense(min_coordinate=min_coord)
        breakpoint()
        
        out_t = self.model_t(sp_tensor)
        out_s = self.model_s(sp_tensor)

        logits_t = out_t['logits']
        feats_t = out_t['feats']

        logits_s = out_s['logits']
        feats_s = out_s['feats']

        sup_data_num = len(sup_selected_idx)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup_s = logits_s[:sup_data_num, :]
        feats_sup_s = feats_s[:sup_data_num, :]

        # logits_sup_t = logits_t[:sup_data_num, :]
        feats_unsup_s = feats_s[sup_data_num:, :]   
        feats_unsup_t = feats_t[sup_data_num:, :]
        
        # NOTE: labeled dataset
        loss = 0
        
        # if self.label_mixing:
        #     mix_feats_s, mix_targets_s = self.mix_features(feats=feats_sup_s, labels=sup_mapped_labels, mixing_ratio=self.mixing_ratio_feat)
            
        #     final_layer_kernel = self.model_s.encoder.final.kernel
        #     final_layer_bias = self.model_s.encoder.final.bias
        #     final_layer2_kernel = self.model_s.encoder.final2.kernel
        #     final_layer2_bias = self.model_s.encoder.final2.bias
            
    
        #     mix_in_normal = torch.matmul(mix_feats_s.detach(), final_layer_kernel) + final_layer_bias
        #     mix_out_normal = torch.matmul(mix_feats_s.detach(), final_layer2_kernel) + final_layer2_bias
        #     mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        #     mix_logits_s = torch.cat([mix_in_normal, mix_out_normal_2], dim =1)

        #     logits_mix_sup_s = torch.cat([logits_sup_s, mix_logits_s], dim=0)
            
        #     targets_sup_s = (
        #     F.one_hot(
        #         sup_mapped_labels.long(),
        #         num_classes=self.hparams.num_labeled_classes + 1,
        #     )
        #     .float()
        #     .to(self.device)
        #     )
        #     target_mix_sup_s = torch.cat([targets_sup_s, mix_targets_s])
        #     mix_seg_loss = self.criterion(logits_mix_sup_s, target_mix_sup_s.long())
        #     loss += mix_seg_loss
        #     self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        #     self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        # else:
        #     if self.label_training:
        #         final_layer_kernel = self.model_s.encoder.final.kernel
        #         final_layer_bias = self.model_s.encoder.final.bias
        #         final_layer2_kernel = self.model_s.encoder.final2.kernel
        #         final_layer2_bias = self.model_s.encoder.final2.bias
                
        #         mix_seg_loss = self.criterion(logits_sup_s, sup_mapped_labels.long())
        #         loss += mix_seg_loss
        #         self.log("train/sup_mix_seg_loss", mix_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        #         self.log("s_mix", mix_seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)\
            
        #     else:
        #         final_layer_kernel = self.model_s.encoder.final.kernel
        #         final_layer_bias = self.model_s.encoder.final.bias
        #         final_layer2_kernel = self.model_s.encoder.final2.kernel
        #         final_layer2_bias = self.model_s.encoder.final2.bias
        
        # NOTE: LiON Loss Porting 
        e_loss, _ = crude_dynamic_energy_loss(logits_sup_s, 
                                    sup_mapped_labels,
                                    ood_ind=self.unknown_label,
                                    details_targets = sup_mapped_labels,
                                    m_out_max = 6,
                                    resized_point_label = self.unknown_label,
                                    resize_m_out= -6)
        self.log("train/energy_loss", e_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("E", e_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        breakpoint()
        # NOTE: unlabeled dataset
        logits_unsup_s = logits_s[sup_data_num:, :]
        # feats_unsup_s = feats_s[sup_data_num:, :]
        
        logits_unsup_t = logits_t[sup_data_num:, :]
        # feats_unsup_t = feats_t[sup_data_num:, :]
        prob_unsup_s = F.softmax(logits_unsup_s, dim=1)
        prob_unsup_t = F.softmax(logits_unsup_t, dim=1)
        
        # LaserMix MSE Loss
        mse_loss = F.mse_loss(prob_unsup_s, prob_unsup_t.detach()) * self.mse_coeff 
        loss += mse_loss
        self.log("train/mse_loss", mse_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        self.log("mse", mse_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        max_probs_unsup_t, target_labels_unsup_t = torch.max(prob_unsup_t, dim=1)
        
        # Voxel to Points 
        points_max_probs_unsup_t = max_probs_unsup_t[unsup_inverse_map]
        points_target_labels_unsup_t = target_labels_unsup_t[unsup_inverse_map] 

        # # Filter only confident points based on softmax probability
        unconfident_mask = points_max_probs_unsup_t < 0.9
        points_target_labels_unsup_t[unconfident_mask] = -1
        
        # NOTE: Lasermix Augmentation for teacher model
        
        lasermix_points, lasermix_features, lasermix_labels = self.mix_transform(sup_data_points, unsup_data_points, points_target_labels_unsup_t)
        
        if lasermix_points is not None:
            # Make LaserMix points to voxel 
            lasermix_coords, lasermix_unique_map, lasermix_inverse_map = ME.utils.sparse_quantize(
                coordinates=lasermix_points,
                return_index=True,
                return_inverse=True,
                quantization_size=self.hparams.voxel_size,
            )
            
            lasermix_coords = lasermix_coords.to(self.device)
            lasermix_features = lasermix_features[lasermix_unique_map].to(self.device)
            lasermix_labels = lasermix_labels[lasermix_unique_map].to(self.device)
            mix_sp_tensor = ME.SparseTensor(features=lasermix_features.float(), coordinates=lasermix_coords.int())        
                    
            # out_lasermix_t = self.model_t(mix_sp_tensor)
            out_lasermix_s = self.model_s(mix_sp_tensor)
            # lasermix_logits_s = torch.argmax(torch.softmax(out_lasermix_s['logits'], dim=1),dim=1)
            lasermix_loss = self.criterion(out_lasermix_s['logits'], lasermix_labels.long()) * self.lasermix_coeff
            loss += lasermix_loss
            self.log("train/lasermix_loss", lasermix_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("lm", lasermix_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
        
        
        """
        TODO
        1. clustering
        - over clustering 해서 metric에 따라 novel classifier 학습
        """
        
        # NOTE: unlabeled dataset
        # threshold = 0.2
        ncc_unsup = prob_unsup_t[:,-1]
        # novel_candidate_mask = ncc_unsup > threshold
        novel_candidate_mask = ncc_unsup > self.novel_candidate_thld
        num_novel_candidate = novel_candidate_mask.sum()
        if num_novel_candidate > 0:
            # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]
            novel_cand_feats = feats_unsup_t[novel_candidate_mask]
            
            if self.novel_queue is not None:
                novel_cand_feats = torch.vstack((novel_cand_feats, self.novel_queue))
            
            # if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            if len(novel_cand_feats) > int(self.hparams.num_unlabeled_classes + self.alpha):
                # Run Kmeans clustering to the novel candidate
                # kmeans = self.kmeans.fit(novel_cand_feats.detach().cpu())
                # cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
                # cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
                
                cluster_labels = self.kmeans.fit_predict(novel_cand_feats.detach())
                cluster_centers = self.kmeans.centroids
                
                # NOTE: base classifier / ncc?, metric: euclidean / cosine?
                cluster_logits = torch.mm(cluster_centers, final_layer_kernel) + final_layer_bias
                # cluster_logits.max(dim=1)[0]
                
                unrealiable_labels = torch.argsort(cluster_logits.max(dim=1)[0], descending=True)[:self.alpha]
                
                # discard queue features
                cluster_curr_labels = cluster_labels[:num_novel_candidate]
                unrealiable_mask = torch.isin(cluster_curr_labels, unrealiable_labels)
                
                if (~unrealiable_mask).sum() == 0:
                    return loss
                
                novel_curr_feats = novel_cand_feats[:num_novel_candidate]
                reliable_feats = novel_curr_feats[~unrealiable_mask]
                reliable_cluster_labels = cluster_curr_labels[~unrealiable_mask]
                
                # final_layer_kernel = base prototypes
                # base_prototypes = final_layer_kernel.double().detach()
                # distances = torch.tensor([
                #     torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
                #     for center in cluster_centers
                # ], device=self.device)
                
                # unrealiable_labels = torch.argmin(distances).item()
                # unrealiable_mask = cluster_labels == unrealiable_labels
                # unrealiable_feats = novel_feats[unrealiable_mask]
                # reliable_feats = novel_feats[~unrealiable_mask]
                # reliable_cluster_labels = cluster_labels[~unrealiable_mask]

                unique_labels = torch.unique(reliable_cluster_labels)
                label_mapping = torch.ones(len(cluster_centers))*(-100)
                for new_label, old_label in enumerate(unique_labels):
                    label_mapping[old_label] = new_label
                
                reliable_cluster_labels = label_mapping[reliable_cluster_labels]
                assert -100 not in reliable_cluster_labels
                
                # label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

                # label_mapping을 사용해 reliable_cluster_labels 재정렬
                # reliable_cluster_labels = torch.tensor(
                #     [label_mapping[label.item()] for label in reliable_cluster_labels],
                #     device=reliable_cluster_labels.device
                # )
                
                novel_kernel = self.model_s.encoder.final3.kernel
                novel_bias = self.model_s.encoder.final3.bias
                
                # realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_kernel) + novel_bias
                realiable_novel_logits = torch.matmul(reliable_feats, novel_kernel) + novel_bias
                realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

                unsup_cost_matrix = np.zeros(
                        (
                            self.hparams.num_unlabeled_classes,
                            self.hparams.num_unlabeled_classes,
                        ), dtype=np.int64
                    )
                idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0).astype(np.int32)
                try:
                    np.add.at(unsup_cost_matrix, tuple(idxs), 1)
                except:
                    breakpoint()

                row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max() - unsup_cost_matrix) # pred, label

                mapping_label = torch.ones(len(row_ind), device=self.device)*(-100)
                for row, col in zip(row_ind, col_ind):
                    mapping_label[col] = row
                mapped_novel_labels = mapping_label[reliable_cluster_labels.long()]
                
                ### (optional)
                if self.use_sup_logit:
                    realiable_novel_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_novel_logits), dim=1)
                    mapped_novel_labels = mapped_novel_labels + self.hparams.num_labeled_classes
                
                novel_unsup_loss = self.novel_coeff * self.novel_criterion(realiable_novel_logits, mapped_novel_labels.long())
                loss += novel_unsup_loss
                self.log("train/unsup_novel_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("u_nov", novel_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_sup_logit:
                    novel_logits_sup = torch.matmul(feats_sup_s, novel_kernel) + novel_bias
                    novel_logits_sup = torch.cat((logits_sup_s[:,:-1], novel_logits_sup), dim=1)
                
                    novel_sup_loss = self.sup_novel_coeff * self.novel_criterion(novel_logits_sup, sup_mapped_labels.long())
                    loss += novel_sup_loss
                    self.log("train/sup_novel_loss", novel_sup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("s_nov", novel_sup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ### (optional)
                if self.use_unsup_ncc_loss:
                    ncc_kernel = self.model_s.encoder.final2.kernel
                    ncc_bias = self.model_s.encoder.final2.bias
                    
                    # realiable_ncc_logits = torch.matmul(reliable_feats.detach(), ncc_kernel) + ncc_bias
                    realiable_ncc_logits = torch.matmul(reliable_feats, ncc_kernel) + ncc_bias
                    realiable_ncc_logits, _ = torch.max(realiable_ncc_logits, dim=1, keepdim=True)
                    realiable_ncc_logits = torch.cat((logits_unsup_s[novel_candidate_mask][~unrealiable_mask][:,:-1], realiable_ncc_logits), dim=1)
                    ncc_labels = torch.ones(realiable_ncc_logits.shape[0], device=self.device) * self.unknown_label
                    
                    ncc_unsup_loss = self.ncc_coeff * self.novel_criterion(realiable_ncc_logits, ncc_labels.long())
                    loss += ncc_unsup_loss
                    self.log("train/unsup_ncc_loss", ncc_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                    self.log("u_ncc", ncc_unsup_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)
                
                ## update novel queue
                # mean_reliable_feats = torch.mean(reliable_feats, dim=0)
                # self.update_queue(mean_reliable_feats.detach())
                # 
                self.update_queue(reliable_feats.detach())
                self.log("utils/novel_feats", len(reliable_feats), batch_size=len(unsup_pcd_indexes))
                self.log("utils/novel_queue", len(self.novel_queue), batch_size=len(unsup_pcd_indexes))
        
        return loss