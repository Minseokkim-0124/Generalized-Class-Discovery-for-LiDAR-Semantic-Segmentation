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
import copy
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import jaccard_index
from tqdm import tqdm


# Optimizer
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from utils.scheduler import LinearWarmupCosineAnnealingLR

# Clustering
from fast_pytorch_kmeans import KMeans
# from torch_kmeans import KMeans
import hdbscan 
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans as KMeansSklearn
from torch_scatter import scatter_mean

# from torch.utils.data.dataset import ConcatDataset
from models.multiheadminkunet import MinkUNetBase, MinkUNetRC, MinkUNetBaseCosine, MinkUNetRCCosine
from utils.collation import (
    collation_fn_restricted_dataset,
    collation_fn_restricted_dataset_two_samples_ssl,
    collation_fn_ssl_dataset,
    collation_fn_restricted_dataset_two_samples,
)
from mmdet.datasets.samplers import MultiSourceSampler
from torch.utils.data import ConcatDataset, DataLoader, Subset, SequentialSampler
from utils.dataset_remission import get_dataset as get_dataset_remission
from utils.dataset import swav_wrapper_ssl, get_dataset
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.eval import SemanticEval
from utils.np_ioueval import iouEval
# from utils.ioueval import iouEval

# Visualization
from utils.visualize_ply import *
from utils.visualize import get_color, get_color_cluster

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


### (Stage-1) Pre-train
class ExpPretrain(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )
        
        self.model = MinkUNetBase(
            num_classes=self.hparams.num_labeled_classes,
        ) #feature와 prototypes 생성
        
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
        
        # wCE as loss
        self.criterion = torch.nn.CrossEntropyLoss()
        # weights = torch.ones(int(self.hparams.num_labeled_classes))/int(self.hparams.num_labeled_classes) 
        # self.criterion.weight = weights

        self.valid_criterion = torch.nn.CrossEntropyLoss()
        # weights = torch.ones(int(self.hparams.num_labeled_classes))/int(self.hparams.num_labeled_classes)
        # self.valid_criterion.weight = weights

        # if self.hparams.pretrained is not None:
        #     self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
        # Mapping numeric_label -> word_label
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

        if self.hparams.dataset == 'nuScenes':
            from nuscenes import NuScenes
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=dataset_config['dataset_path'], verbose=True)
        else:
            self.nusc = None
        
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
    
    @staticmethod
    def freeze(model: nn.Module) -> None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
    def configure_optimizers(self):
        optimizer = optim.SGD(
            params=self.model.parameters(),
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
        dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        
        if self.debug:
            if self.hparams.dataset == 'nuScenes':
                dataset.nusc_infos = dataset.nusc_infos[:50]
            else:
                dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader
    
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

    def test_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, 'labeled')(
            config_file=self.hparams.dataset_config,
            split="valid",
            voxel_size=self.hparams.voxel_size,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        
        if self.debug:
            dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return dataloader
    
    def training_step(self, data, _):
        (
            coords,
            feats,
            real_labels,
            selected_idx,
            mapped_labels,
            _,
            pcd_indexes,
        ) = data
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
    
        out = self.model(sp_tensor)
        logits = out['logits']

        loss = self.criterion(logits, mapped_labels.long())
        self.log("train/loss", loss, sync_dist=True, rank_zero_only=True)

        return loss
    
    def on_validation_epoch_start(self):
        D = self.hparams.num_classes
        self.cost_matrix = np.zeros((D, D), dtype=np.int64)
        self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)

        return

    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())

        mask_lab = mapped_labels != self.unknown_label
                    
        out = self.model(sp_tensor)

        logits = out['logits']
        
        loss = self.valid_criterion(logits[mask_lab], mapped_labels[mask_lab].long())
        
        # Change mIoU metric
        pred_labels = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)
        # Transform predictions
        for key, value in self.label_mapping_inv.items():
            pred_labels[pred_labels == key] = -value
        pred_labels = -pred_labels
        
        # Add point-wise predicitons to cost matrix
        for pcd_idx in range(pcd_indexes.shape[0]):
            mask_inv_map = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]
                                                            ]
            pcd_preds = pred_labels[coords[:,0] == pcd_idx]
            pcd_preds = pcd_preds[inverse_maps[pcd_idx][mask_inv_map]].cpu().numpy()
            
            pcd_labels = real_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx][mask_inv_map]].cpu().numpy()
            
            # For Strict-Hungarian
            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            

        return {'loss': loss.item()}
    
    def validation_epoch_end(self, valid_step_outputs):
        loss = 0
        for output in valid_step_outputs:
            loss += output['loss']
        
        # logging
        self.log(f'valid/loss', loss / len(valid_step_outputs), on_step=False, rank_zero_only=True)
    
        # For Strict-Hungarian
        ind = linear_sum_assignment(self.cost_matrix.max() - self.cost_matrix)
        ind = np.vstack(ind).T
        
        conf_matrix = self.cost_matrix[:, ind[:,1]]
        include = np.argsort(ind[:,1])[:self.hparams.num_classes]
        
        iou = self.get_iou(conf_matrix, include)
        miou = iou.mean()
        miou_old = iou[self.known_real_labels].mean()
        # miou_new = iou[self.unknown_real_labels].mean()
        
        _, output_dict = self.log_metric(miou, iou)
        self.log(f'valid/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'valid/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        # self.log(f'valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'valid/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
        if self.hparams.test:
            # confusion matrix
            import matplotlib.pyplot as plt
            from mlxtend.plotting import plot_confusion_matrix
            import copy
            class_names = copy.deepcopy(self.label_dict)
            # del class_names[-1]
            exclude = np.argsort(ind[:,1])[self.hparams.num_classes:]
            conf_matrix_with_unlab = np.zeros((len(include)+1, len(include)+1), dtype=np.int64)
            conf_matrix_with_unlab[1:, 1:] = conf_matrix[include][:,include]
            conf_matrix_with_unlab[0, 1:] = conf_matrix[exclude].sum(0)[include]
            checkpoint_name = self.hparams.checkpoint_dir.split('/')[-1]
            # fig, ax = plot_confusion_matrix(conf_mat=conf_matrix[include][:,include].T, colorbar=True, show_absolute=False, show_normed=True, class_names=list(class_names.values()))
            
            fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_with_unlab.T, colorbar=True, show_absolute=False, show_normed=True, class_names=list(class_names.values()), figsize=(12,12))
            plt.savefig(f'{self.hparams.log_dir}/{checkpoint_name}-conf_mat.png', dpi=300)
            
        return

    def get_precision_and_recall_for_novel(self, conf_matrix):
            conf = conf_matrix.copy().astype(np.double)
            tp = conf.diagonal()
            fp = conf.sum(axis=1) - tp
            fn = conf.sum(axis=0) - tp
            
            precision = tp / np.maximum((tp + fp), 1e-15)
            recall = tp / np.maximum((tp + fn), 1e-15)
            
            return precision, recall

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

    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        
        D = self.hparams.num_classes
        self.cost_matrix = np.zeros((D, D), dtype=np.int64)
        self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)

        return
    
    def test_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())

        mask_lab = mapped_labels != self.unknown_label
                    
        out = self.model(sp_tensor)

        logits = out['logits']
        
        # loss = self.valid_criterion(logits[mask_lab], mapped_labels[mask_lab].long())
        
        # Change mIoU metric
        pred_labels = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)
        # Transform predictions
        for key, value in self.label_mapping_inv.items():
            pred_labels[pred_labels == key] = -value
        pred_labels = -pred_labels
        
        # Add point-wise predicitons to cost matrix
        for pcd_idx in range(pcd_indexes.shape[0]):
            mask_inv_map = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]]
            pcd_preds = pred_labels[coords[:,0] == pcd_idx]
            pcd_preds = pcd_preds[inverse_maps[pcd_idx][mask_inv_map]].cpu().numpy()
            
            pcd_labels = real_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx][mask_inv_map]].cpu().numpy()
            
            # For Strict-Hungarian
            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            

        return 
    
    def test_epoch_end(self, _):
        # loss = 0
        # for output in test_step_outputs:
        #     loss += output['loss']
        
        # logging
        # self.log(f'test/loss', loss / len(test_step_outputs), on_step=False, rank_zero_only=True)
    
        # For Strict-Hungarian
        ind = linear_sum_assignment(self.cost_matrix.max() - self.cost_matrix)
        ind = np.vstack(ind).T
        
        conf_matrix = self.cost_matrix[:, ind[:,1]]
        include = np.argsort(ind[:,1])[:self.hparams.num_classes]
        
        iou = self.get_iou(conf_matrix, include)
        miou = iou.mean()
        miou_old = iou[self.known_real_labels].mean()
        # miou_new = iou[self.unknown_real_labels].mean()
        
        _, output_dict = self.log_metric(miou, iou)
        self.log(f'test/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'test/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        # self.log(f'valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'test/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
        return
    
class ExpCosinePretrain(ExpPretrain):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model = MinkUNetBaseCosine(
            num_classes=self.hparams.num_labeled_classes,
        )
        




### (Stage-2) NCC Fine-tuning
class ExpFineTuning(ExpPretrain):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model = MinkUNetRC(
            num_labeled=self.hparams.num_labeled_classes,
        )
        
        if self.hparams.resume_checkpoint is not None:
            print("##############################Resume Training##############################")
            self.model.encoder.final2 = ME.MinkowskiConvolution(
                            self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                            out_channels=3,
                            kernel_size=1,
                            bias=True,
                            dimension=3)

        elif self.hparams.pretrained is not None:
            self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
            self.model.encoder.final2 = ME.MinkowskiConvolution(
                            self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                            out_channels=3,
                            kernel_size=1,
                            bias=True,
                            dimension=3)
        else:
            raise FileNotFoundError("Checkpoint is needed for fine-tuning RC")

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        weights = torch.ones(int(self.hparams.num_labeled_classes + 1))/int(self.hparams.num_labeled_classes + 1) 
        self.criterion.weight = weights

        self.valid_criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(int(self.hparams.num_labeled_classes + 1))/int(self.hparams.num_labeled_classes + 1)
        self.valid_criterion.weight = weights

        self.hungarian_at_each_step = True 

        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 0.15
        # NOTE: KITTI
        else:
            self.calib_coeff = 0.05 
            
    def configure_evaluator(self):
        self.evaluators = []
        
        target = copy.deepcopy(self.target)
        for t in target:
            if t == self.source:
                self.evaluators.append(SemanticEval(self.class_num, None, [0]))
            else:
                self.evaluators.append(SemanticEval(len(self.class_str_common), None, [0]))
        # self.evaluators = [SemanticEval(self.class_num, None, [0]) for _ in target]
        
        for evaluator in self.evaluators:
            evaluator.reset()
        return
    
    def configure_ood_evaluator(self):
        # overriding previous evaluators
        self.evaluators = iouEval(self.hparams.num_labeled_classes + 1 , ignore=-1, unknown=self.unknown_label)
        
        
        self.evaluators.reset()
        return
    
    def configure_optimizer(self):
        if self.hparams.pretrained is not None:
            if True:
                encoder_params = list(self.model.encoder.parameters())  # 인코더의 모든 파라미터를 가져옴

                optimizer = optim.SGD(
                    [
                        {"params": encoder_params, "lr": self.hparams.finetune_lr},  # 모든 인코더 파라미터에 대해 finetune_lr 적용
                    ],
                    momentum=self.hparams.momentum_for_optim,
                    weight_decay=self.hparams.weight_decay_for_optim,
                )
            
            # final2_params = list(self.model.encoder.final2.parameters())
            # encoder_params = [
            # param for param in self.model.encoder.parameters()
            # if param not in final2_params
            # ]
            
            # optimizer = optim.SGD(
            # [
            #     {"params": final2_params, "lr": self.hparams.train_lr},  # final2에 대해 train_lr 적용
            #     {"params": encoder_params, "lr": self.hparams.finetune_lr},  # 나머지 인코더에 대해 finetune_lr 적용
            # ],
            # momentum=self.hparams.momentum_for_optim,
            # weight_decay=self.hparams.weight_decay_for_optim,
            # )
        else: 
            raise FileNotFoundError("Checkpoint is needed for fine-tuning RC")
        
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
        dataset = get_dataset_remission(self.hparams.dataset, "finetuning")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            resize_aug=True,
            nusc=self.nusc,
        )
        
        if self.debug:
            dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader
    
    def training_step(self, data, _):
        (
            coords,
            feats,
            _,
            _,
            mapped_labels,
            _,
            _,
        ) = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
    
        out = self.model(sp_tensor)
        
        logits = out['logits']
        
        loss = 0

        seg_loss = self.criterion(logits, mapped_labels.long())
        loss += seg_loss
        self.log("seg", seg_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)

        # calibration loss
        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits.shape[0]).unsqueeze(0).to(logits.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
        
        labels_dummy = torch.ones(logits.shape[0]).long().to(logits.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calibration_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calibration_loss
        self.log("calib", calibration_loss.detach(), prog_bar=True, logger=False, on_epoch=False, sync_dist=True)

        results = {
            "train/loss": loss.detach(),
            "train/seg_loss" : seg_loss.detach(),
            "train/calibration_loss" : calibration_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_validation_epoch_start(self):
        D = int(self.hparams.num_labeled_classes + 1)
        
        # 여러 개의 cost_matrix를 하나의 리스트로 처리
        # self.novel_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        self.novel_thresholds = [0.2, 0.3, 0.4, 0.5]
        # self.novel_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        
        return
    
    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        mask_lab = mapped_labels != self.unknown_label
                    
        loss = self.valid_criterion(out['logits'], mapped_labels.long())
        # Change mIoU metric
                # Change mIoU metric
        preds = F.softmax(out['logits'], dim=1)
        rc_prob = preds[:,-1]
        pred_labels = torch.argmax(preds, dim=1)
        
        pred_novel_masks = [(rc_prob > thresh) for thresh in self.novel_thresholds]
        pred_labels_list = [pred_labels.clone() for _ in self.novel_thresholds]
        
        for i, pred_labels in enumerate(pred_labels_list):
            pred_labels[pred_novel_masks[i]] = self.unknown_label
        
        # # Transform predictions
        # gt_labels = real_labels
        # for key, value in self.label_mapping_inv.items():
        #     pred_labels[pred_labels == key] = -value
        # pred_labels = -pred_labels
        
        # Add point-wise predicitons to cost matrix
        for pcd_idx in range(pcd_indexes.shape[0]):
            pcd_labels = mapped_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
            for i, pred_labels in enumerate(pred_labels_list):
                pcd_preds = pred_labels[coords[:,0] == pcd_idx]
                pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
                
                # For Strict-Hungarian
                idxs = np.stack([pcd_preds, pcd_labels], axis=0)
                np.add.at(self.cost_matrices[i], tuple(idxs), 1)            
            

        return {'loss': loss.item()}
    
    def validation_epoch_end(self, valid_step_outputs):
        loss = 0
        for output in valid_step_outputs:
            loss += output['loss']
        
        # logging
        self.log(f'valid/loss', loss / len(valid_step_outputs), on_step=False, rank_zero_only=True)

        # mious = []
        for i, cost_matrix in enumerate(self.cost_matrices):
            # Strict-Hungarian assignment
            # conf_base_base = conf_matrix[:-1,:-1].sum()
            conf_base_novel = cost_matrix[:-1,-1].sum()
            conf_novel_base = cost_matrix[-1,:-1].sum()
            conf_novel_novel = cost_matrix[-1,-1].sum()
            
            recall_novel = conf_novel_novel / np.maximum((conf_novel_novel + conf_base_novel), 1e-4)
            precision_novel = conf_novel_novel / np.maximum((conf_novel_novel + conf_novel_base), 1e-4)
            
            # precision, recall = self.get_precision_and_recall_for_novel(conf_precision_recall)
            self.log(f'valid/precision/threshold_{self.novel_thresholds[i]}', precision_novel, on_step=False, rank_zero_only=True)
            self.log(f'valid/recall/threshold_{self.novel_thresholds[i]}', recall_novel, on_step=False, rank_zero_only=True)
            
            # ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
            # ind = np.vstack(ind).T

            # 새로운 conf_matrix에 대해 IoU 계산
            # conf_matrix = cost_matrix[:, ind[:, 1]]
            # include = np.argsort(ind[:, 1])[:int(self.hparams.num_labeled_classes + 1)]
            include = np.arange(self.hparams.num_labeled_classes+1)

            iou = self.get_iou(cost_matrix, include)
            miou = iou.mean()

            # known classes와 unknown (novel) classes에 대한 mIoU
            miou_new = iou[self.unknown_label].mean()
            miou_old = iou[:self.unknown_label].mean()
            # 각각의 threshold에 따른 mIoU 기록
            # self.log(f'valid/miou_threshold_{self.novel_thresholds[i]}', miou, on_step=False, rank_zero_only=True)
            self.log(f'valid/miou_known_threshold_{self.novel_thresholds[i]}', miou_old, on_step=False, rank_zero_only=True)
            self.log(f'valid/miou_novel_threshold_{self.novel_thresholds[i]}', miou_new, on_step=False, rank_zero_only=True)
            
        
        # # For Strict-Hungarian
        # ind = linear_sum_assignment(self.cost_matrix.max() - self.cost_matrix)
        # ind = np.vstack(ind).T
        
        # conf_matrix = self.cost_matrix[:, ind[:,1]]
        # include = np.argsort(ind[:,1])[:self.hparams.num_classes]
        
        # iou = self.get_iou(conf_matrix, include)
        # miou = iou.mean()
        # miou_old = iou[self.known_real_labels].mean()
        # miou_new = iou[self.unknown_real_labels].mean()
        
        # _, output_dict = self.log_metric(miou, iou)
        # self.log(f'valid/mIoU', miou, on_step=False, rank_zero_only=True)
        # self.log(f'valid/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        # self.log(f'valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        # for class_, iou_ in output_dict.items():
        #     if class_ == 'all': continue
        #     self.log(f'valid/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
        # # Precison and Recall
        # precision, recall = self.get_precision_and_recall(conf_matrix, include)
        # print("precision old mean: {:.4f}".format(precision[self.known_real_labels].mean().item()))
        # print("recall old mean: {:.4f}".format(recall[self.known_real_labels].mean().item()))   
        # self.log(f'valid/precision/old', precision[self.known_real_labels].mean().item(), on_step=False, rank_zero_only=True)
        # self.log(f'valid/precision/new', precision[self.unknown_real_labels].mean().item(), on_step=False, rank_zero_only=True)
        # self.log(f'valid/recall/old', recall[self.known_real_labels].mean().item(), on_step=False, rank_zero_only=True)
        # self.log(f'valid/recall/new', recall[self.unknown_real_labels].mean().item(), on_step=False, rank_zero_only=True)
         
        
       
        
        # if self.hparams.test:
        #     # confusion matrix
        #     import matplotlib.pyplot as plt
        #     from mlxtend.plotting import plot_confusion_matrix
        #     import copy
        #     class_names = copy.deepcopy(self.label_dict)
        #     # del class_names[-1]
        #     # exclude = np.argsort(ind[:,1])[self.hparams.num_classes:]
        #     conf_matrix_with_unlab = np.zeros((len(include)+1, len(include)+1), dtype=np.int64)
        #     conf_matrix_with_unlab[1:, 1:] = cost_matrix[include][:,include]
        #     conf_matrix_with_unlab[0, 1:] = cost_matrix[exclude].sum(0)[include]
        #     checkpoint_name = self.hparams.checkpoint_dir.split('/')[-1]
        #     # fig, ax = plot_confusion_matrix(conf_mat=conf_matrix[include][:,include].T, colorbar=True, show_absolute=False, show_normed=True, class_names=list(class_names.values()))
            
        #     fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_with_unlab.T, colorbar=True, show_absolute=False, show_normed=True, class_names=list(class_names.values()), figsize=(12,12))
        #     plt.savefig(f'{self.hparams.log_dir}/{checkpoint_name}-conf_mat.png', dpi=300)
            
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
            dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
        return dataloader
    
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        
        self.visualization = visualization
        self.save_result = save_result
        
        if ood_evaluation:
            assert not save_result, "cannot use ood evaluation when save the results"
            assert evaluation_method in ['maxlogit', 'msp', 'entropy']

            self.configure_ood_evaluator()
            self.ood_evaluation_method = evaluation_method
        else:
            self.configure_evaluator()
        return
    
    def test_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        # if self.global_step % self.hparams.clear_cache_int == 0:
        #     torch.cuda.empty_cache()

        with torch.no_grad():
            out = self.model(sp_tensor)
            # mask_lab = mapped_labels != self.unknown_label
                        
            # Change mIoU metric
            logits = out['logits']
            preds = logits[:,:-1] #RC 제외하고 꺼냄
            preds = F.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1).detach().cpu().numpy().astype(np.int32)
            labels = mapped_labels.cpu().numpy()
            # preds - /
            
            temperature = 1
            uncertainty_scores_softmax = torch.softmax(logits / temperature, dim=1)[:,-1]
            uncertainty_scores_softmax = uncertainty_scores_softmax.cpu().detach().numpy()
            self.evaluators.addBatch(preds, labels, uncertainty_scores_softmax)
            
        return 
    
    def test_epoch_end(self, test_step_outputs):
        output_list = []
        average = 0
        num = 0

        iou, output_dict = self.log_metric_with_ood(self.evaluators)
        output_list.append(output_dict)
        
        self.log('metric/mIoU', iou, rank_zero_only=True)
        self.log('metric/AUPR', output_dict['aupr']['IoU'], rank_zero_only=True)
        self.log('metric/AUROC', output_dict['auroc']['IoU'], rank_zero_only=True)
        self.log('metric/FPR95', output_dict['fpr95']['IoU'], rank_zero_only=True)
        
        for k, v in output_dict.items():
            if k in ['all', 'aupr', 'auroc', 'fpr95', 'unlabeled', 'noise']:
                continue
            self.log(f'IoU/{k}', v['IoU'], rank_zero_only=True) 
        
        self.evaluators.reset()
        average += iou
        num += 1
        
        assert num > 0, "need at least one test set"
        
        self.log(f'metric/mIoU/average', average/num, rank_zero_only=True)
        # self.write_txt(output_list, average=average/num)
        return
    
    def log_metric_with_ood(self, evaluator):
        assert isinstance(evaluator, iouEval)
            
        class_IoU, class_all_IoU = evaluator.getIoU()
        aupr, auroc, fpr95 = evaluator.get_unknown_indices(save_dir=self.logger.log_dir)

        # now make a nice dictionary
        output_dict = {}

        # make python variables
        class_IoU = class_IoU.item()
        class_all_IoU = class_all_IoU.flatten().tolist()
        aupr, auroc, fpr95 = aupr.item(), auroc.item(), fpr95.item()

        output_dict["aupr"] = {}
        output_dict["aupr"]["IoU"] = aupr
        
        output_dict["auroc"] = {}
        output_dict["auroc"]["IoU"] = auroc
        
        output_dict["fpr95"] = {}
        output_dict["fpr95"]["IoU"] = fpr95
        
        output_dict["all"] = {}
        output_dict["all"]["IoU"] = class_IoU

        self.label_mapping_inv[len(self.label_mapping_inv.keys())] = 999
        self.label_dict[999] = 'novel'
        for idx, iou in enumerate(class_all_IoU):
            label_idx = self.label_mapping_inv[idx]
            c_str = self.label_dict[label_idx]
            output_dict[c_str] = {}
            output_dict[c_str]["IoU"] = iou
        
        mIoU = output_dict["all"]["IoU"]
        return mIoU, output_dict

class ExpRCExtra(ExpFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model.encoder.final2 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=3,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        if self.hparams.pretrained is not None:
            self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
            
    def train_dataloader(self):
        # aug 된 데이터 셋
        lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            labeled=True,
            nusc=self.nusc,
        )
        # aug 된 데이터 셋
        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                downsampling=self.hparams.downsampling,
                voxel_size=self.hparams.voxel_size,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
            )
        
        if self.debug:
            lab_dataset.num_files = 30
            unlab_dataset.num_files = 30

        dataset = ConcatDataset([lab_dataset, unlab_dataset])

        # weights = [1 / len(lab_dataset)] * len(lab_dataset) + [1 / len(unlab_dataset)] * len(unlab_dataset)
        # sampler = WeightedRandomSampler(weights, num_samples= len(dataset), replacement=True)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_ssl_dataset,
            num_workers=self.hparams.num_workers,
            sampler=MultiSourceSampler(dataset=dataset, batch_size=self.hparams.batch_size, source_ratio=[1,1]),
            shuffle=False,
        )

        return dataloader
    
    def training_step(self, data, _):
        sup_data = data['sup']
        unsup_data = data['unsup']

        coords = sup_data['coords']
        unsup_coords = unsup_data['coords']

        feats = sup_data['feats']
        unsup_feats = unsup_data['feats']

        unsup_coords[:,0] += len(sup_data['pcd_idx'])
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        out = self.model(sp_tensor)

        logits = out['logits']

        sup_data_num = 0
        for i in range(len(sup_data['selected_idx'])):
            sup_data_num += len(sup_data['selected_idx'][i])
        
        logits_sup = logits[:sup_data_num, :]
        logits_unsup = logits[sup_data_num:, :]

        mapped_labels = sup_data['mapped_label']
        if self.unknown_label in mapped_labels.unique():
            raise ValueError
        unsup_mapped_labels = unsup_data['mapped_label']

        unlab_mask_lab = unsup_mapped_labels == self.unknown_label

        loss = 0
        # Supervised Loss
        sup_seg_loss =self.criterion(logits_sup, mapped_labels.long())
        loss += sup_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = 0.01 * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        # Unsupervised Loss for novel points
        novel_unsup_logits = logits_unsup[unlab_mask_lab]
        novel_unsup_probs = F.softmax(novel_unsup_logits, dim=1)
        novel_rc_probs = novel_unsup_probs[:, -1]
        novel_pred_labels = torch.argmax(novel_unsup_probs, dim=1)

        threshold = 0.21
        threshold_mask = novel_rc_probs > threshold 
        novel_pred_labels[threshold_mask] = self.unknown_label
        novel_pred_labels[~threshold_mask] = -1 
        
        if len(novel_unsup_logits) != 0:
            unsup_seg_loss = 0.2 * self.criterion(novel_unsup_logits, novel_pred_labels.long())
            loss += unsup_seg_loss
            self.log("train/unsup_seg_loss", unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        results = {
            "train/loss" : loss.detach(),
            "train/sup_seg_loss" : sup_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }


        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_validation_epoch_start(self):
        D = int(self.hparams.num_labeled_classes + 1)
        
        # 여러 개의 cost_matrix를 하나의 리스트로 처리
        # self.novel_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        self.novel_thresholds = [0.21]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        
        return

class ExpClusterFineTuning(ExpFineTuning):
    def train_dataloader(self):
        lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            labeled=True,
            nusc=self.nusc,
        )
        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                downsampling=self.hparams.downsampling,
                voxel_size=self.hparams.voxel_size,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
            )
        
        if self.debug:
            lab_dataset.num_files = 30
            unlab_dataset.num_files = 30

        dataset = ConcatDataset([lab_dataset, unlab_dataset])

        # weights = [1 / len(lab_dataset)] * len(lab_dataset) + [1 / len(unlab_dataset)] * len(unlab_dataset)
        # sampler = WeightedRandomSampler(weights, num_samples= len(dataset), replacement=True)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_ssl_dataset,
            num_workers=self.hparams.num_workers,
            sampler=MultiSourceSampler(dataset=dataset, batch_size=self.hparams.batch_size, source_ratio=[1,1]),
            shuffle=False,
        )

        return dataloader
    
    def on_train_start(self):
        # self.kmeans = KMeans(n_clusters=self.hparams.num_labeled_classes + 1, verbose=0, mode='cosine')
        self.kmeans = KMeansSklearn(n_clusters=self.hparams.num_labeled_classes + 1, verbose=0, n_init='auto')
        self.dbscan = DBSCAN(eps=3, min_samples=2)

    def training_step(self, data, _):
        sup_data = data['sup']
        unsup_data = data['unsup']

        coords = sup_data['coords']
        unsup_coords = unsup_data['coords']

        feats = sup_data['feats']
        unsup_feats = unsup_data['feats']

        unsup_coords[:,0] += len(sup_data['pcd_idx'])
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        out = self.model(sp_tensor)

        logits = out['logits']

        sup_data_num = 0
        for i in range(len(sup_data['selected_idx'])):
            sup_data_num += len(sup_data['selected_idx'][i])
        
        logits_sup = logits[:sup_data_num, :]
        logits_unsup = logits[sup_data_num:, :]

        mapped_labels = sup_data['mapped_label']

        # Supervised 
        loss = 0 
        sup_seg_loss = self.criterion(logits_sup, mapped_labels.long())
        loss += sup_seg_loss

        # Unsupervised 
        with torch.no_grad():
            mask_unknown = []
            for i in range(len(sup_data['pcd_idx']), (len(sup_data['pcd_idx']) + len(unsup_data['pcd_idx']))):
                batch_mask = unsup_coords[:,0] == i
                coords_batch = unsup_coords[batch_mask][:, 1:].detach().cpu().numpy()
                feats_batch = unsup_feats[batch_mask].detach()
                logits_batch = logits_unsup[:,:self.unknown_label][batch_mask].detach()

                db_labels = self.dbscan.fit_predict(coords_batch)
                n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

                clustered_avg_feats = torch.zeros(n_clusters, feats_batch.size(1), device=self.device)
                db_labels = torch.from_numpy(db_labels).to(self.device)
                db_cluster_indices = {i: torch.where(db_labels == i)[0] for i in db_labels.cpu().numpy()}

                valid_mask = db_labels != -1 
                cluster_ids = db_labels[valid_mask]
                feats_valid = feats_batch[valid_mask]

                clustered_avg_feats = scatter_mean(feats_valid, cluster_ids, dim=0, dim_size=n_clusters)
                kmeans_feats = self.kmeans.fit_predict(clustered_avg_feats.detach().cpu())
                kmeans_feats = torch.from_numpy(kmeans_feats).to(self.device)
                kmeans_cluster_indices = {i: torch.where(kmeans_feats == i)[0] for i in kmeans_feats.cpu().numpy()}
                kmeans_to_coords_indices = {}

                for kmeans_key, dbscan_cluster_ids in kmeans_cluster_indices.items():
                    coord_indices = []
                    for dbscan_cluster_id in dbscan_cluster_ids:
                        # coord_indices.extend(db_cluster_indices[dbscan_cluster_id.item()].cpu().numpy())
                        coord_indices.append(db_cluster_indices[dbscan_cluster_id.item()].cpu())
                    # kmeans_to_coords_indices[kmeans_key] = coord_indices
                    if coord_indices:
                        kmeans_to_coords_indices[kmeans_key] = torch.cat(coord_indices).to(self.device)
                
                probs = F.softmax(logits_batch, dim=1)
                cluster_class_prob = torch.zeros((self.hparams.num_labeled_classes + 1, self.hparams.num_labeled_classes), device=self.device)
                dummy_matrix = torch.zeros((self.hparams.num_labeled_classes + 1, 1), device=self.device)

                # NOTE : 얘는 각 noise point가 하나의 cluster라고 여기고 kmeans를 하던지 해야함. 위에서 적은 3번에 해당
                # noise_ids = db_labels[~valid_mask]
                feats_noise = feats_batch[~valid_mask]

                kmeans_noise_feats = self.kmeans.fit_predict(feats_noise.detach().cpu())
                kmeans_noise_feats = torch.from_numpy(kmeans_noise_feats).to(self.device)

                # kmeans_noise_feats = kmeans_noise_feats.to(self.device)
                kmeans_noise_indices = {i: torch.where(kmeans_noise_feats == i)[0] for i in kmeans_noise_feats.cpu().numpy()}
                
                # noise key = -1 in db_cluster_indices
                
                for kmeans_key, dbscan_cluster_ids in kmeans_noise_indices.items():
                    noise_indices = db_cluster_indices[-1][dbscan_cluster_ids]
                    if kmeans_key in kmeans_to_coords_indices:
                        # kmeans_to_coords_indices[kmeans_key].extend(noise_indices)
                        kmeans_to_coords_indices[kmeans_key] = torch.cat([kmeans_to_coords_indices[kmeans_key], noise_indices])    
                    else:
                        # breakpoint()
                        raise ValueError()
                
                for cluster_id, indices in kmeans_to_coords_indices.items():
                    cluster_probs = probs[indices]
                    cluster_class_prob[cluster_id] = torch.mean(cluster_probs,axis=0)

                cluster_class_prob = torch.hstack((cluster_class_prob, dummy_matrix))

                if torch.any(torch.isnan(cluster_class_prob)) or torch.any(torch.isinf(cluster_class_prob)):            
                    cluster_class_prob = torch.nan_to_num(cluster_class_prob, nan=0.0, posinf=0.0, neginf=0.0)  

                # hungarian matching for mapping clusters and classes
                row_ind, col_ind = linear_sum_assignment(cost_matrix=cluster_class_prob.cpu(), maximize=True)
        
                # novel_class_indices = torch.arange(self.hparams.num_labeled_classes, self.hparams.num_labeled_classes + 1, device=self.device)
                mapping_novel_classes = torch.empty((0,), dtype=torch.int32, device=self.device)
                for cluster_idx, class_idx in zip(row_ind, col_ind):
                    if class_idx == self.unknown_label:
                        mapping_novel_classes = torch.cat((mapping_novel_classes, kmeans_to_coords_indices[cluster_idx]))
                
                view_idx = torch.arange(coords_batch.shape[0], device=self.device)
                batch_mask_unknown = torch.isin(view_idx, mapping_novel_classes)
                mask_unknown.append(batch_mask_unknown)
            mask_unknown = torch.cat(mask_unknown)

        unsup_mapped_labels = torch.zeros(mask_unknown.shape[0], device=self.device)
        unsup_mapped_labels[mask_unknown] = self.unknown_label
        unsup_loss = 0.1 * self.criterion(logits_unsup, unsup_mapped_labels.long())
        
        loss += unsup_loss

        results = {
            "train/loss" : loss.detach(),
            "train/sup_seg_loss" : sup_seg_loss.detach(),
            "train/unsup_seg_loss" : unsup_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss

class ExpMixFineTuning(ExpFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)

        self.temperature = 0.07
        self.mixing_ratio_feat = 0.1

        self.centroid_sampling = False
        self.beta_sampling = True
        self.beta_coeff = 0.5
        
        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 1
            self.entropy_minimize = False
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-3
        
        # NOTE: KITTI
        else:
            self.calib_coeff = 0.05 
            self.entropy_minimize = False
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-6

    def train_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            resize_aug=False,
            nusc=self.nusc,
        )
        
        if self.debug:
            dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader 
    
    def training_step(self, data, _):
        # breakpoint()/
        (
            coords,
            feats,
            _,
            _,
            mapped_labels,
            _,
            _,
        ) = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
    
        out = self.model(sp_tensor)
        
        logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        if self.centroid_sampling:
            mix_feats, mix_labels = self.mix_centroid_features(feats=out['feats'], labels=mapped_labels, mask=mask_lab)
        else:
            mix_feats, mix_labels = self.mix_features(feats=out['feats'], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        # mix_in_normal = final_layer(mix_feats).F
        # mix_out_normal = final_layer2(mix_feats).F


        # mix_sparse_tensor = ME.SparseTensor(features=mix_feats.float(), coordinates=coords.int())
        # mix_out = self.model(mix_sparse_tensor)
        # mix_logits = mix_out['logits']                                          
        # mix_logits = torch.cat([mix_logits_in_normal, mix_logits_out_normal_2], dim=1)

        logits = torch.cat([logits, mix_logits], dim=0)
        mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits, mapped_labels.long())
        loss += mix_seg_loss

        # calibration loss 
        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits.shape[0]).unsqueeze(0).to(logits.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
        
        labels_dummy = torch.ones(logits.shape[0]).long().to(logits.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calibration_loss = self.calib_coeff  * self.criterion(preds_dummy, labels_dummy)
        loss += calibration_loss

        
        if self.entropy_minimize:
            probs = F.softmax(logits, dim=1)
            rc_probs = probs[:, -1]
            probs = probs[:, :-1]
            id_entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1)) * self.id_entropy_coeff
            loss += id_entropy_loss 
            # NOTE: clamp version
            # log_rc_probs = torch.log(torch.clamp(rc_probs, min=1e-8))
            # ood_entropy_loss = torch.mean(torch.sum(rc_probs * log_rc_probs)) * self.ood_entropy_coeff
            
            # NOTE: non clamp version
            ood_entropy_loss = torch.mean(torch.sum(rc_probs * torch.log(rc_probs + 1e-8))) * self.ood_entropy_coeff
            loss += ood_entropy_loss 
            
            self.log("train/id_entropy_loss", id_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/ood_entropy_loss", ood_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            
        results = {
            "train/loss": loss.detach(),
            "train/mix_seg_loss" : mix_seg_loss.detach(),
            "train/calibration_loss" : calibration_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def mix_features(self, feats, labels, mask, mixing_ratio):
        with torch.no_grad():
            # mask unknown labels
            feats = feats[mask]
            labels = labels[mask]
                        
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            diff_mask = (labels[permutation_idx1] != labels[permutation_idx2])
                        
            feats1 = feats[permutation_idx1][diff_mask]
            feats2 = feats[permutation_idx2][diff_mask]
            
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
    
    def mix_centroid_features(self, feats, labels, mask):
        with torch.no_grad():
            feats = feats[mask]
            labels = labels[mask]
            
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

            mix_labels = torch.full((centroid_feats.shape[0],), self.unknown_label, device=self.device, dtype=torch.int32)

        return centroid_feats, mix_labels
    
    def on_validation_epoch_start(self):
        D = int(self.hparams.num_labeled_classes + 1)
        
        # 여러 개의 cost_matrix를 하나의 리스트로 처리
        # self.novel_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        # self.novel_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.novel_thresholds = [0.2,0.3,0.4,0.5]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        
        return

    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        for key, val in enumerate(self.label_dict) :
            if key in self.unknown_real_labels:
                self.label_dict[key] = 'novel'
        
        self.label_mapping_inv[self.unknown_label] = self.unknown_label

        self.visualization = visualization
        # self.save_result = save_result
        D = int(self.hparams.num_labeled_classes + 1)
        self.novel_threshold = 0.2 
        self.cost_matrix = np.zeros((D,D), dtype=np.int64)

        # D = int(self.hparams.num_labeled_classes + 1)
        # self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)
        return
    
    def test_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        out = self.model(sp_tensor)

        preds = F.softmax(out['logits'], dim=1)
        rc_prob = preds[:,-1]
        pred_labels = torch.argmax(preds, dim=1)

        pred_novel_masks = rc_prob > self.novel_threshold
        pred_labels[pred_novel_masks] = self.unknown_label
        
        # for key, value in self.label_mapping_inv.items():
        #     pred_labels[pred_labels == key] = -value
        # pred_labels = -pred_labels

        for pcd_idx in range(pcd_indexes.shape[0]):
            pcd_labels = mapped_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
            pcd_preds = pred_labels[coords[:,0] == pcd_idx]
            pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()

            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1) 

        # mask_lab = mapped_labels != self.unknown_label
    
    def test_epoch_end(self, _):
        include = np.arange(self.hparams.num_labeled_classes+1)
        
        iou = self.get_iou(self.cost_matrix, include)
        miou = iou.mean()
        miou_old = iou[:self.unknown_label].mean()
        miou_new = iou[self.unknown_label].mean()
        
        _, output_dict = self.log_metric(miou, iou)
        self.log(f'thershold{self.novel_threshold}-test/mIoU', miou, on_step=False, rank_zero_only=True)
        self.log(f'thershold{self.novel_threshold}-test/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
        self.log(f'thershold{self.novel_threshold}-test/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
        
        for class_, iou_ in output_dict.items():
            if class_ == 'all': continue
            self.log(f'thershold{self.novel_threshold}-test/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
    
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

class ExpBetaSchedulingFineTuning(ExpMixFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)

        self.temperature = 0.07
        self.mixing_ratio_feat = 0.1

        self.centroid_sampling = True
        self.beta_sampling = True
        self.start_beta_coeff = 1
        self.end_beta_coeff = 0.1
        
        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 0.1
            self.entropy_minimize = False
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-3
        
        # NOTE: KITTI
        else:
            self.calib_coeff = 0.05 
            self.entropy_minimize = False
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-6
    
    def on_train_start(self):
        steps_per_epoch = len(self.train_dataloader())  
        self.max_steps = self.trainer.max_epochs * steps_per_epoch  
        

    def training_step(self, data, _):
        # breakpoint()/
        (
            coords,
            feats,
            _,
            _,
            mapped_labels,
            _,
            _,
        ) = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
    
        out = self.model(sp_tensor)
        
        logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # Beta sampling scheduling
        current_step = self.global_step     
        
        if current_step < self.max_steps:
            mixing_ratio = 1 - (current_step / self.max_steps) * (self.start_beta_coeff - self.end_beta_coeff)  
        else:
            mixing_ratio = 0.1
    
        loss = 0
        if self.centroid_sampling:
            mix_feats, mix_labels = self.mix_centroid_features(feats=out['feats'], labels=mapped_labels, mask=mask_lab)
        else:
            mix_feats, mix_labels = self.mix_features(feats=out['feats'], labels=mapped_labels, mask=mask_lab, mixing_ratio=mixing_ratio)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        # mix_in_normal = final_layer(mix_feats).F
        # mix_out_normal = final_layer2(mix_feats).F


        # mix_sparse_tensor = ME.SparseTensor(features=mix_feats.float(), coordinates=coords.int())
        # mix_out = self.model(mix_sparse_tensor)
        # mix_logits = mix_out['logits']                                          
        # mix_logits = torch.cat([mix_logits_in_normal, mix_logits_out_normal_2], dim=1)

        logits = torch.cat([logits, mix_logits], dim=0)
        mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits, mapped_labels.long())
        loss += mix_seg_loss

        # calibration loss 
        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits.shape[0]).unsqueeze(0).to(logits.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
        
        labels_dummy = torch.ones(logits.shape[0]).long().to(logits.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calibration_loss = self.calib_coeff  * self.criterion(preds_dummy, labels_dummy)
        loss += calibration_loss

        
        if self.entropy_minimize:
            probs = F.softmax(logits, dim=1)
            rc_probs = probs[:, -1]
            probs = probs[:, :-1]
            id_entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1)) * self.id_entropy_coeff
            loss += id_entropy_loss 
            # NOTE: clamp version
            # log_rc_probs = torch.log(torch.clamp(rc_probs, min=1e-8))
            # ood_entropy_loss = torch.mean(torch.sum(rc_probs * log_rc_probs)) * self.ood_entropy_coeff
            
            # NOTE: non clamp version
            ood_entropy_loss = torch.mean(torch.sum(rc_probs * torch.log(rc_probs + 1e-8))) * self.ood_entropy_coeff
            loss += ood_entropy_loss 
            
            self.log("train/id_entropy_loss", id_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/ood_entropy_loss", ood_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            
        results = {
            "train/loss": loss.detach(),
            "train/mix_seg_loss" : mix_seg_loss.detach(),
            "train/calibration_loss" : calibration_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
        
class ExpMixCosineFineTuning(ExpCosinePretrain):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model = MinkUNetRCCosine(
            num_labeled=self.hparams.num_labeled_classes,
        )
        if self.hparams.resume_checkpoint is not None:
            print("##############################Resume Training##############################")
            # self.model.encoder.final2 = ME.MinkowskiConvolution(
            #                 self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
            #                 out_channels=3,
            #                 kernel_size=1,
            #                 bias=True,
            #                 dimension=3)
            

            self.model.encoder.final2 = CosinePrototypes(output_dim=self.model.feat_dim ,num_prototypes=3)

        elif self.hparams.pretrained is not None:
            self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
            self.model.encoder.final2 = CosinePrototypes(output_dim=self.model.feat_dim ,num_prototypes=3)
            
            # self.model.encoder.final2 = ME.MinkowskiConvolution(
            #                 self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
            #                 out_channels=3,
            #                 kernel_size=1,
            #                 bias=True,
            #                 dimension=3)
        else:
            raise FileNotFoundError("Checkpoint is needed for fine-tuning RC")
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        weights = torch.ones(int(self.hparams.num_labeled_classes + 1))/int(self.hparams.num_labeled_classes + 1) 
        self.criterion.weight = weights

        self.valid_criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(int(self.hparams.num_labeled_classes + 1))/int(self.hparams.num_labeled_classes + 1)
        self.valid_criterion.weight = weights

        self.hungarian_at_each_step = True
        
        # Mixing
        self.temperature = 0.07
        self.mixing_ratio_feat = 0.1

        self.beta_sampling = True
        self.beta_coeff = 0.5
        
        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 0.1
            self.entropy_minimize = False
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-3
        else:
        # Entropy minimize maximize
            self.calib_coeff = 0.1
            self.entropy_minimize = True
            self.id_entropy_coeff = 0.1
            self.ood_entropy_coeff = 1e-6

    def configure_optimizer(self):
        if self.hparams.pretrained is not None:
            if True:
                encoder_params = list(self.model.encoder.parameters())  # 인코더의 모든 파라미터를 가져옴

                optimizer = optim.SGD(
                    [
                        {"params": encoder_params, "lr": self.hparams.finetune_lr},  # 모든 인코더 파라미터에 대해 finetune_lr 적용
                    ],
                    momentum=self.hparams.momentum_for_optim,
                    weight_decay=self.hparams.weight_decay_for_optim,
                )

    def train_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            resize_aug=False,
            nusc=self.nusc,
        )
        
        if self.debug:
            dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader 
    
    def training_step(self, data, _):
        # breakpoint()/
        (
            coords,
            feats,
            _,
            _,
            mapped_labels,
            _,
            _,
        ) = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
    
        out = self.model(sp_tensor)
        
        logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=out['feats'], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        final_layer_kernel = self.model.head_lab.prototypes.weight.T
        final_layer2_kernel = self.model.encoder.final2.prototypes.weight.T

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        # mix_in_normal = final_layer(mix_feats).F
        # mix_out_normal = final_layer2(mix_feats).F


        # mix_sparse_tensor = ME.SparseTensor(features=mix_feats.float(), coordinates=coords.int())
        # mix_out = self.model(mix_sparse_tensor)
        # mix_logits = mix_out['logits']                                          
        # mix_logits = torch.cat([mix_logits_in_normal, mix_logits_out_normal_2], dim=1)

        logits = torch.cat([logits, mix_logits], dim=0)
        mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits, mapped_labels.long())
        loss += mix_seg_loss

        # calibration loss 
        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits.shape[0]).unsqueeze(0).to(logits.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9
        
        labels_dummy = torch.ones(logits.shape[0]).long().to(logits.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calibration_loss = self.calib_coeff  * self.criterion(preds_dummy, labels_dummy)
        loss += calibration_loss

        if self.entropy_minimize:
            probs = F.softmax(logits, dim=1)
            rc_probs = probs[:, -1]
            probs = probs[:, :-1]
            id_entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1)) * self.id_entropy_coeff
            loss += id_entropy_loss 
            
            log_rc_probs = torch.log(torch.clamp(rc_probs, min=1e-8))
            ood_entropy_loss = torch.mean(torch.sum(rc_probs * log_rc_probs)) * self.ood_entropy_coeff
            # ood_entropy_loss = torch.mean(torch.sum(rc_probs * torch.log(rc_probs + 1e-8))) * self.ood_entropy_coeff
            loss += ood_entropy_loss 
            self.log("train/id_entropy_loss", id_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/ood_entropy_loss", ood_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            
        results = {
            "train/loss": loss.detach(),
            "train/mix_seg_loss" : mix_seg_loss.detach(),
            "train/calibration_loss" : calibration_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def mix_features(self, feats, labels, mask, mixing_ratio):
        with torch.no_grad():
            # mask unknown labels
            feats = feats[mask]
            labels = labels[mask]
                        
            permutation_idx1 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            permutation_idx2 = np.random.choice(feats.shape[0], feats.shape[0], replace=False)
            diff_mask = (labels[permutation_idx1] != labels[permutation_idx2])
                        
            feats1 = feats[permutation_idx1][diff_mask]
            feats2 = feats[permutation_idx2][diff_mask]
            
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

    def on_validation_epoch_start(self):
        D = int(self.hparams.num_labeled_classes + 1)
        
        # 여러 개의 cost_matrix를 하나의 리스트로 처리
        # self.novel_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        self.novel_thresholds = [0.2, 0.3, 0.4, 0.5]
        # self.novel_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        
        return
    
    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        mask_lab = mapped_labels != self.unknown_label
                    
        loss = self.valid_criterion(out['logits'], mapped_labels.long())
        # Change mIoU metric
                # Change mIoU metric
        preds = F.softmax(out['logits'], dim=1)
        rc_prob = preds[:,-1]
        pred_labels = torch.argmax(preds, dim=1)
        
        pred_novel_masks = [(rc_prob > thresh) for thresh in self.novel_thresholds]
        pred_labels_list = [pred_labels.clone() for _ in self.novel_thresholds]
        
        for i, pred_labels in enumerate(pred_labels_list):
            pred_labels[pred_novel_masks[i]] = self.unknown_label
        
        # # Transform predictions
        # gt_labels = real_labels
        # for key, value in self.label_mapping_inv.items():
        #     pred_labels[pred_labels == key] = -value
        # pred_labels = -pred_labels
        
        # Add point-wise predicitons to cost matrix
        for pcd_idx in range(pcd_indexes.shape[0]):
            pcd_labels = mapped_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
            for i, pred_labels in enumerate(pred_labels_list):
                pcd_preds = pred_labels[coords[:,0] == pcd_idx]
                pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
                
                # For Strict-Hungarian
                idxs = np.stack([pcd_preds, pcd_labels], axis=0)
                np.add.at(self.cost_matrices[i], tuple(idxs), 1)            
            
        return {'loss': loss.item()}
    
    def validation_epoch_end(self, valid_step_outputs):
        loss = 0
        for output in valid_step_outputs:
            loss += output['loss']
        
        # logging
        self.log(f'valid/loss', loss / len(valid_step_outputs), on_step=False, rank_zero_only=True)

        # mious = []
        for i, cost_matrix in enumerate(self.cost_matrices):
            # Strict-Hungarian assignment
            # conf_base_base = conf_matrix[:-1,:-1].sum()
            conf_base_novel = cost_matrix[:-1,-1].sum()
            conf_novel_base = cost_matrix[-1,:-1].sum()
            conf_novel_novel = cost_matrix[-1,-1].sum()
            
            recall_novel = conf_novel_novel / np.maximum((conf_novel_novel + conf_base_novel), 1e-4)
            precision_novel = conf_novel_novel / np.maximum((conf_novel_novel + conf_novel_base), 1e-4)
            
            # precision, recall = self.get_precision_and_recall_for_novel(conf_precision_recall)
            self.log(f'valid/precision/threshold_{self.novel_thresholds[i]}', precision_novel, on_step=False, rank_zero_only=True)
            self.log(f'valid/recall/threshold_{self.novel_thresholds[i]}', recall_novel, on_step=False, rank_zero_only=True)
            
            # ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
            # ind = np.vstack(ind).T

            # 새로운 conf_matrix에 대해 IoU 계산
            # conf_matrix = cost_matrix[:, ind[:, 1]]
            # include = np.argsort(ind[:, 1])[:int(self.hparams.num_labeled_classes + 1)]
            include = np.arange(self.hparams.num_labeled_classes+1)

            iou = self.get_iou(cost_matrix, include)
            miou = iou.mean()

            # known classes와 unknown (novel) classes에 대한 mIoU
            miou_new = iou[self.unknown_label].mean()
            miou_old = iou[:self.unknown_label].mean()
            # 각각의 threshold에 따른 mIoU 기록
            # self.log(f'valid/miou_threshold_{self.novel_thresholds[i]}', miou, on_step=False, rank_zero_only=True)
            self.log(f'valid/miou_known_threshold_{self.novel_thresholds[i]}', miou_old, on_step=False, rank_zero_only=True)
            self.log(f'valid/miou_novel_threshold_{self.novel_thresholds[i]}', miou_new, on_step=False, rank_zero_only=True)
            
class ExpMixRealAugFineTuning(ExpMixFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)

        self.temperature = 0.07
        self.mixing_ratio_feat = 0.1

        self.beta_sampling = True
        self.beta_coeff = 0.5
        
        self.calib_coeff = 0.05 

    def train_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, "finetuning")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            labeled=False,
            resize_aug=True,
            nusc=self.nusc,
        )
        
        if self.debug:
            dataset.num_files = 50

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader

class ExpMixExtraFineTuning(ExpMixFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model.encoder.final2 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=3,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        self.calib_coeff = 0.05 
        self.temperature = 0.07
        self.mixing_ratio_feat = 0.1

        self.beta_sampling = True
        self.beta_coeff = 0.5
        self.entropy_minimize = True
        self.unsup_novel_only = False

        if self.hparams.pretrained is not None:
            self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
    
    def configure_optimizer(self):
        if self.hparams.pretrained is not None:
                if True:
                    encoder_params = list(self.model.encoder.parameters())  # 인코더의 모든 파라미터를 가져옴

                    optimizer = optim.SGD(
                        [
                            {"params": encoder_params, "lr": self.hparams.train_lr},  # 모든 인코더 파라미터에 대해 finetune_lr 적용
                        ],
                        momentum=self.hparams.momentum_for_optim,
                        weight_decay=self.hparams.weight_decay_for_optim,
                    )
                
                # final2_params = list(self.model.encoder.final2.parameters())
                # encoder_params = [
                # param for param in self.model.encoder.parameters()
                # if param not in final2_params
                # ]
                
                # optimizer = optim.SGD(
                # [
                #     {"params": final2_params, "lr": self.hparams.train_lr},  # final2에 대해 train_lr 적용
                #     {"params": encoder_params, "lr": self.hparams.finetune_lr},  # 나머지 인코더에 대해 finetune_lr 적용
                # ],
                # momentum=self.hparams.momentum_for_optim,
                # weight_decay=self.hparams.weight_decay_for_optim,
                # )
                else: 
                    raise FileNotFoundError("Checkpoint is needed for fine-tuning RC")
                
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
        # aug 된 데이터 셋
        lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            labeled=True,
            nusc=self.nusc,
        )
        # aug 된 데이터 셋
        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                downsampling=self.hparams.downsampling,
                voxel_size=self.hparams.voxel_size,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
            )
        
        if self.debug:
            lab_dataset.num_files = 30
            unlab_dataset.num_files = 30

        dataset = ConcatDataset([lab_dataset, unlab_dataset])

        # weights = [1 / len(lab_dataset)] * len(lab_dataset) + [1 / len(unlab_dataset)] * len(unlab_dataset)
        # sampler = WeightedRandomSampler(weights, num_samples= len(dataset), replacement=True)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_ssl_dataset,
            num_workers=self.hparams.num_workers,
            sampler=MultiSourceSampler(dataset=dataset, batch_size=self.hparams.batch_size, source_ratio=[1,1]),
            shuffle=False,
        )

        return dataloader
    
    def training_step(self, data, _):
        sup_data = data['sup']
        unsup_data = data['unsup']

        coords = sup_data['coords']
        unsup_coords = unsup_data['coords']

        feats = sup_data['feats']
        unsup_feats = unsup_data['feats']

        unsup_coords[:,0] += len(sup_data['pcd_idx'])
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        out = self.model(sp_tensor)

        logits = out['logits']

        sup_data_num = 0
        for i in range(len(sup_data['selected_idx'])):
            sup_data_num += len(sup_data['selected_idx'][i])
        
        logits_sup = logits[:sup_data_num, :]
        logits_unsup = logits[sup_data_num:, :]

        mapped_labels = sup_data['mapped_label']
        if self.unknown_label in mapped_labels.unique():
            raise ValueError
        
        # unsup_mapped_labels = unsup_data['mapped_label']

        # unlab_mask_lab = unsup_mapped_labels == self.unknown_label
        
        # logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=out['feats'][:sup_data_num, :], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        # Unsupervised Loss for novel points
        if self.unsup_novel_only:
            mapped_label_unsup = unsup_data['mapped_label']
            mask_novel = mapped_label_unsup == self.unknown_label
            novel_unsup_logits = logits_unsup[mask_novel]
        else:
            novel_unsup_logits = logits_unsup
            novel_unsup_probs = F.softmax(novel_unsup_logits, dim=1)
            novel_rc_probs = novel_unsup_probs[:, -1]
            novel_pred_labels = torch.argmax(novel_unsup_probs, dim=1)

        # threshold = 0.21 # REAL
        # threshold = 0.35 # no beta Mix
        # threshold = 0.21 # beta 1  Mix
        # threshold = 0.45 #  best precision in beta 0.5 coeff 0.05
        
        # NOTE: adaptive threshold is needed 
        threshold = 0.2
        threshold_mask = novel_rc_probs > threshold 
        novel_pred_labels[threshold_mask] = self.unknown_label
        # novel_pred_labels[~threshold_mask] = -1 # ignore for training
        
        # Open-set Entropy Minimization from NIPS OpenMatch
        if self.entropy_minimize:
            entropy_loss = -torch.mean(torch.sum(novel_unsup_probs * torch.log(novel_unsup_probs + 1e-8), dim=-1)) * 0.1
            loss += entropy_loss 
            self.log("train/unsup_entropy_loss", entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        if len(novel_unsup_logits) != 0:
            # unsup seg loss need to be checked
            unsup_seg_loss = 0.1 * self.criterion(novel_unsup_logits, novel_pred_labels.long())
            loss += unsup_seg_loss
            self.log("train/unsup_seg_loss", unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_validation_epoch_start(self):
        D = int(self.hparams.num_labeled_classes + 1)
        
        # 여러 개의 cost_matrix를 하나의 리스트로 처리
        # self.novel_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        self.novel_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        
        return      
    
    def visualize_novel(self, filename, coords, 
                  preds, 
                  labels):
            color_labels = get_color(labels, self.hparams.dataset_config)
            color_preds = get_color(preds, self.hparams.dataset_config)
            
            write_ply(filename + '-novel-gt.ply', [coords, color_labels], ['x','y','z','red','green','blue'])
            write_ply(filename + '-pd.ply', [coords, color_preds], ['x','y','z','red','green','blue'])

class ExpMixRealAugExtraFineTuning(ExpMixExtraFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.calib_coeff = 0.05 
        
        self.temperature = 0.07
        self.mixing_ratio_feat = 0.1

        self.beta_sampling = True
        self.beta_coeff = 0.5
        
    def train_dataloader(self):
        # aug 된 데이터 셋
        lab_dataset = get_dataset_remission(self.hparams.dataset, "finetuning")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            labeled=True,
            nusc=self.nusc,
        )
        # aug 된 데이터 셋
        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'finetuning')(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                downsampling=self.hparams.downsampling,
                voxel_size=self.hparams.voxel_size,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
            )
        
        if self.debug:
            lab_dataset.num_files = 30
            unlab_dataset.num_files = 30

        dataset = ConcatDataset([lab_dataset, unlab_dataset])

        # weights = [1 / len(lab_dataset)] * len(lab_dataset) + [1 / len(unlab_dataset)] * len(unlab_dataset)
        # sampler = WeightedRandomSampler(weights, num_samples= len(dataset), replacement=True)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_ssl_dataset,
            num_workers=self.hparams.num_workers,
            sampler=MultiSourceSampler(dataset=dataset, batch_size=self.hparams.batch_size, source_ratio=[1,1]),
            shuffle=False,
        )

        return dataloader

class ExpMixExtraStepSchedulingFineTuning(ExpMixExtraFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.threshold = 0.1
        self.end_threshold = 0.5 
        # self.total_steps = self.hparams.epochs * (len(self.train_dataloader().dataset) // self.hparams.batch_size)
        #     
    def training_step(self, data, _):
        sup_data = data['sup']
        unsup_data = data['unsup']

        coords = sup_data['coords']
        unsup_coords = unsup_data['coords']

        feats = sup_data['feats']
        unsup_feats = unsup_data['feats']

        unsup_coords[:,0] += len(sup_data['pcd_idx'])
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        out = self.model(sp_tensor)

        logits = out['logits']

        sup_data_num = 0
        for i in range(len(sup_data['selected_idx'])):
            sup_data_num += len(sup_data['selected_idx'][i])
        
        logits_sup = logits[:sup_data_num, :]
        logits_unsup = logits[sup_data_num:, :]

        mapped_labels = sup_data['mapped_label']
        if self.unknown_label in mapped_labels.unique():
            raise ValueError
        
        # unsup_mapped_labels = unsup_data['mapped_label']

        # unlab_mask_lab = unsup_mapped_labels == self.unknown_label
        
        # logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=out['feats'][:sup_data_num, :], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = 0.05 * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        # Unsupervised Loss for novel points
        novel_unsup_logits = logits_unsup
        novel_unsup_probs = F.softmax(novel_unsup_logits, dim=1)
        novel_rc_probs = novel_unsup_probs[:, -1]
        novel_pred_labels = torch.argmax(novel_unsup_probs, dim=1)

        # threshold = 0.21 # REAL
        # threshold = 0.35 # no beta Mix
        # threshold = 0.21 # beta 1  Mix
        # threshold = 0.45 #  best precision in beta 0.5 coeff 0.05
        
        # NOTE: adaptive threshold is needed 
        # current_step = self.global_step
        # if current_step <= self.total_steps:
        #     threshold = self.start_threshold + (self.end_threshold - self.start_threshold) * (current_step/ self.total_steps)
        # else:
        #     threshold = self.end_threshold
        # novel_pred_labels[~threshold_mask] = -1 # ignore for training
        novel_threshold = novel_rc_probs > self.threshold
        novel_pred_labels[novel_threshold] = self.unknown_label
        if len(novel_unsup_logits) != 0:
            # unsup seg loss need to be checked
            unsup_seg_loss = 0.1 * self.criterion(novel_unsup_logits, novel_pred_labels.long())
            loss += unsup_seg_loss
            self.log("train/unsup_seg_loss", unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        epoch = self.current_epoch 
        if epoch % 2 == 0:
            self.threshold = self.threshold + (self.end_threshold - self.threshold) * epoch / self.hparams.epochs

class ExpMixExtraPolySchedulingFineTuning(ExpMixExtraStepSchedulingFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.init_threshold = 0.1
        self.end_threshold = 0.5 
        self.total_steps = self.hparams.epochs * (len(self.train_dataloader().dataset) // self.hparams.batch_size)
    
    def training_step(self, data, _):
        sup_data = data['sup']
        unsup_data = data['unsup']

        coords = sup_data['coords']
        unsup_coords = unsup_data['coords']

        feats = sup_data['feats']
        unsup_feats = unsup_data['feats']

        unsup_coords[:,0] += len(sup_data['pcd_idx'])
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        out = self.model(sp_tensor)

        logits = out['logits']

        sup_data_num = 0
        for i in range(len(sup_data['selected_idx'])):
            sup_data_num += len(sup_data['selected_idx'][i])
        
        logits_sup = logits[:sup_data_num, :]
        logits_unsup = logits[sup_data_num:, :]

        mapped_labels = sup_data['mapped_label']
        if self.unknown_label in mapped_labels.unique():
            raise ValueError
        
        # unsup_mapped_labels = unsup_data['mapped_label']

        # unlab_mask_lab = unsup_mapped_labels == self.unknown_label
        
        # logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=out['feats'][:sup_data_num, :], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = 0.05 * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        # Unsupervised Loss for novel points
        novel_unsup_logits = logits_unsup
        novel_unsup_probs = F.softmax(novel_unsup_logits, dim=1)
        novel_rc_probs = novel_unsup_probs[:, -1]
        novel_pred_labels = torch.argmax(novel_unsup_probs, dim=1)

        # threshold = 0.21 # REAL
        # threshold = 0.35 # no beta Mix
        # threshold = 0.21 # beta 1  Mix
        # threshold = 0.45 #  best precision in beta 0.5 coeff 0.05
        
        # NOTE: adaptive threshold is needed 
        # current_step = self.global_step
        # if current_step <= self.total_steps:
        #     threshold = self.start_threshold + (self.end_threshold - self.start_threshold) * (current_step/ self.total_steps)
        # else:
        #     threshold = self.end_threshold
        # novel_pred_labels[~threshold_mask] = -1 # ignore for training
        current_step = self.global_step
        a = self.init_threshold 
        b = self.end_threshold
        progress = current_step / self.total_steps
        if current_step <= self.total_steps:
            threshold = a + (b-a) * (progress ** 2)
        else:
            threshold = self.end_threshold

        novel_threshold = novel_rc_probs > threshold
        novel_pred_labels[novel_threshold] = self.unknown_label
        if len(novel_unsup_logits) != 0:
            # unsup seg loss need to be checked
            unsup_seg_loss = 0.1 * self.criterion(novel_unsup_logits, novel_pred_labels.long())
            loss += unsup_seg_loss
            self.log("train/unsup_seg_loss", unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
         super().on_train_epoch_end()

class ExpMixExtraLinearSchedulingFineTuning(ExpMixExtraFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.start_threshold = 0.1
        self.end_threshold = 0.5
        self.total_steps = self.hparams.epochs * (len(self.train_dataloader().dataset) // self.hparams.batch_size)
    
    def training_step(self, data, _):
        sup_data = data['sup']
        unsup_data = data['unsup']

        coords = sup_data['coords']
        unsup_coords = unsup_data['coords']

        feats = sup_data['feats']
        unsup_feats = unsup_data['feats']

        unsup_coords[:,0] += len(sup_data['pcd_idx'])
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        out = self.model(sp_tensor)

        logits = out['logits']

        sup_data_num = 0
        for i in range(len(sup_data['selected_idx'])):
            sup_data_num += len(sup_data['selected_idx'][i])
        
        logits_sup = logits[:sup_data_num, :]
        logits_unsup = logits[sup_data_num:, :]

        mapped_labels = sup_data['mapped_label']
        if self.unknown_label in mapped_labels.unique():
            raise ValueError
        
        # unsup_mapped_labels = unsup_data['mapped_label']

        # unlab_mask_lab = unsup_mapped_labels == self.unknown_label
        
        # logits = out['logits']
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=out['feats'][:sup_data_num, :], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = 0.05 * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        # Unsupervised Loss for novel points
        novel_unsup_logits = logits_unsup
        novel_unsup_probs = F.softmax(novel_unsup_logits, dim=1)
        novel_rc_probs = novel_unsup_probs[:, -1]
        novel_pred_labels = torch.argmax(novel_unsup_probs, dim=1)

        # threshold = 0.21 # REAL
        # threshold = 0.35 # no beta Mix
        # threshold = 0.21 # beta 1  Mix
        # threshold = 0.45 #  best precision in beta 0.5 coeff 0.05
        
        # NOTE: adaptive threshold is needed 
        current_step = self.global_step
        if current_step <= self.total_steps:
            threshold = self.start_threshold + (self.end_threshold - self.start_threshold) * (current_step/ self.total_steps)
        else:
            threshold = self.end_threshold
        # novel_pred_labels[~threshold_mask] = -1 # ignore for training
        novel_threshold = novel_rc_probs > threshold
        novel_pred_labels[novel_threshold] = self.unknown_label
        if len(novel_unsup_logits) != 0:
            # unsup seg loss need to be checked
            unsup_seg_loss = 0.1 * self.criterion(novel_unsup_logits, novel_pred_labels.long())
            loss += unsup_seg_loss
            self.log("train/unsup_seg_loss", unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss

class ExpUncertaintyCheck(pl.LightningModule):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )
        self.model = MinkUNetRC(
            num_labeled=self.hparams.num_labeled_classes,
        )
        self.model.encoder.final2 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=3,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        self.save_hyperparameters(
            {k: v for (k, v) in kwargs.items() if not callable(v)}
        )
        if self.hparams.pretrained is not None:
            self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
            
        self.batch_idx_list = []
        self.uncertainty_list = []

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
        
        # wCE as loss
        self.criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(int(self.hparams.num_labeled_classes))/int(self.hparams.num_labeled_classes) 
        self.criterion.weight = weights

        self.valid_criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(int(self.hparams.num_labeled_classes))/int(self.hparams.num_labeled_classes)
        self.valid_criterion.weight = weights

        # if self.hparams.pretrained is not None:
        #     self.load_pretrained(self.model, pretrained=self.hparams.pretrained)
        # Mapping numeric_label -> word_label
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
        
        if self.hparams.dataset == 'nuScenes':
            from nuscenes import NuScenes
            self.nusc =  NuScenes(version='v1.0-trainval', dataroot=dataset_config['dataset_path'], verbose=True)
        else:
            self.nusc = None
        
    
    def configure_optimizers(self):
        optimizer = optim.SGD(
            params=self.model.parameters(),
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
    
    def train_dataloader(self):
        dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                    config_file=self.hparams.dataset_config,
                    split="train",
                    data_idx_file=self.hparams.data_idx_file,
                    downsampling=self.hparams.downsampling,
                    voxel_size=self.hparams.voxel_size,
                    augment=True,
                    label_mapping=self.label_mapping,
                    labeled=False,
                    store_unlabeled_label=True,
                    nusc=self.nusc,
                )
            
        if self.debug:
            dataset.num_files = 30
                
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=collation_fn_restricted_dataset,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader
    
    def training_step(self, data, _):
        def calculate_uncertainty(logits):
            """
            Calculate uncertainty based on classifier scores.
            :param scores: The output logits from the last classifier layer
            :return: Uncertainty score for each sample in the batch
            """
            probs = F.softmax(logits, dim=1)
            rc_probs = probs[:,-1].reshape(-1,1)
            entropy = -torch.sum(rc_probs * torch.log(rc_probs + 1e-10), dim=1)
            uncertainty = entropy.mean().item()
            return uncertainty
        
        (
            coords, 
            feats, 
            labels, 
            selected_idx, 
            mapped_labels, 
            inverse_maps, 
            pcd_indexes

        ) = data
        pcd_masks = []

        for i in range(len(pcd_indexes)):
            pcd_masks.append(coords[:,0] == i)

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        out = self.model(sp_tensor)
        logits = out['logits']

        for pcd_mask, pcd_idx in zip(pcd_masks, pcd_indexes):
            self.batch_idx_list.append(pcd_idx)
            logits_batch = logits[pcd_mask]
            uncertainty = calculate_uncertainty(logits_batch)
            self.uncertainty_list.append(uncertainty)
        
    def on_train_end(self):
        # Zip batch_idx_list with uncertainty_list and sort by uncertainty in descending order
        batch_uncertainty_pairs = list(zip(self.batch_idx_list, self.uncertainty_list))
        sorted_pairs = sorted(batch_uncertainty_pairs, key=lambda x: x[1], reverse=True)
        
        # Unzip the sorted pairs
        sorted_batch_idx_list, sorted_uncertainty_list = zip(*sorted_pairs)
        
        # Move each tensor in sorted_batch_idx_list to CPU and convert to numpy
        sorted_batch_idx_array = np.array([idx.cpu().numpy() for idx in sorted_batch_idx_list])
        
        # Convert sorted_uncertainty_list directly to numpy array
        sorted_uncertainty_array = np.array(sorted_uncertainty_list)

        # Save sorted arrays as .npy files
        np.save("sorted_batch_idx.npy", sorted_batch_idx_array)
        np.save("sorted_uncertainty.npy", sorted_uncertainty_array)

        print("Sorted uncertainty data saved as sorted_batch_idx.npy and sorted_uncertainty.npy")

    # def on_validation_epoch_start(self):
    #     pass

    # def validation_step(self, data, _):
    #     pass

    # def validation_epoch_end(self, _):
    #     pass

class ExpMixExtraTest(ExpMixExtraFineTuning):
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        
        self.visualization = visualization
        self.save_result = save_result
        D = self.hparams.num_classes
        
        self.novel_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        self.KMeans = KMeansSklearn(n_clusters=2, verbose=0, n_init='auto')
        return
    
    def test_step(self, data, _): 
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        # if self.global_step % self.hparams.clear_cache_int == 0:
        #     torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        mask_lab = mapped_labels != self.unknown_label
        feats = out['feats']

        # loss = self.valid_criterion(out['logits'], mapped_labels.long())
        
        preds = F.softmax(out['logits'], dim=1)
        rc_prob = preds[:,-1]
        pred_labels = torch.argmax(preds, dim=1)
        
        pred_novel_masks = [(rc_prob > thresh) for thresh in self.novel_thresholds]
        pred_labels_list = [pred_labels.clone() for _ in self.novel_thresholds]
        # to_delete = []
        for i, pred_labels in enumerate(pred_labels_list):
            # pred_labels[pred_novel_masks[i]] = self.unknown_label
            feats_novel = feats[pred_novel_masks[i]]
            # print(feats_novel.shape())
            if feats_novel.shape[0] < 2:
                # print(f'Not enough point to do KMeans Clustering -> feature shape = {feats_novel.shape}')/sel
                # to_delete.append(i)
                continue
            kmeans_labels = self.KMeans.fit_predict(feats_novel.detach().cpu())
            kmeans_labels = torch.from_numpy(kmeans_labels).to(self.device)
            kmeans_novel_mask1 = kmeans_labels == 0
            novel_indices = pred_novel_masks[i].nonzero(as_tuple=True)[0]  
            kmeans_indices = novel_indices[kmeans_novel_mask1]
            pred_labels[kmeans_indices] = self.unknown_label
            pred_labels[novel_indices[~kmeans_novel_mask1]] = self.unknown_label + 1
        
        # # for 루프가 끝난 후, 역순으로 삭제
        # for i in sorted(to_delete, reverse=True):
        #     print(f'Threshold to delete ! : {self.novel_thresholds[i]}')
        #     del self.novel_thresholds[i]
        #     del pred_labels_list[i]
        #     del self.cost_matrices[i]
        #         # label mapping inv 변경 필요 
        
        # Transform predictions
        self.label_mapping_inv[self.unknown_label] = self.unknown_real_labels[0]
        self.label_mapping_inv[self.unknown_label+1] = self.unknown_real_labels[1]

        for key, value in self.label_mapping_inv.items():
            for pred_labels in pred_labels_list:
                pred_labels[pred_labels == key] = -value
            pred_labels = -pred_labels
        
        # Add point-wise predicitons to cost matrix
        for pcd_idx in range(pcd_indexes.shape[0]):
            pcd_labels = real_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            for i, pred_labels in enumerate(pred_labels_list):
                pcd_preds = pred_labels[coords[:,0] == pcd_idx]
                pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
                
                # For Strict-Hungarian
                idxs = np.stack([pcd_preds, pcd_labels], axis=0)
                np.add.at(self.cost_matrices[i], tuple(idxs), 1)
            
            # For Greedy-Hungarian
            # pcd_mask_lab = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]].cpu().numpy()
            
            # idxs_old = np.stack([pcd_preds[pcd_mask_lab], pcd_labels[pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_old, tuple(idxs_old), 1)
            
            # idxs_new = np.stack([pcd_preds[~pcd_mask_lab], pcd_labels[~pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_new, tuple(idxs_new), 1)

        
        # # Add point-wise predicitons to cost matrix
        # for pcd_idx in range(pcd_indexes.shape[0]):
        #     pcd_labels = mapped_labels[coords[:,0] == pcd_idx]
        #     pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
        #     for i, pred_labels in enumerate(pred_labels_list):
        #         pcd_preds = pred_labels[coords[:,0] == pcd_idx]
        #         pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
                
        #         # For Strict-Hungarian
        #         idxs = np.stack([pcd_preds, pcd_labels], axis=0)
        #         np.add.at(self.cost_matrices[i], tuple(idxs), 1)  
    
    def test_epoch_end(self, _):
        # mious = []
        for i, cost_matrix in enumerate(self.cost_matrices):
            # For Strict-Hungarian
            ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
            ind = np.vstack(ind).T
            
            conf_matrix = cost_matrix[:, ind[:,1]]
            include = np.argsort(ind[:,1])[:self.hparams.num_classes]
            
            iou = self.get_iou(conf_matrix, include)
            miou = iou.mean()
            miou_old = iou[self.known_real_labels].mean()
            miou_new = iou[self.unknown_real_labels].mean()
            
            _, output_dict = self.log_metric(miou, iou)
            self.log(f'thershold{self.novel_thresholds[i]}-valid/mIoU', miou, on_step=False, rank_zero_only=True)
            self.log(f'thershold{self.novel_thresholds[i]}-valid/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
            self.log(f'thershold{self.novel_thresholds[i]}-valid/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
            
            for class_, iou_ in output_dict.items():
                if class_ == 'all': continue
                self.log(f'thershold{self.novel_thresholds[i]}-valid/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
            
        
        
        
        # if self.hparams.test:
        #     # confusion matrix
        #     import matplotlib.pyplot as plt
        #     from mlxtend.plotting import plot_confusion_matrix
        #     import copy
        #     class_names = copy.deepcopy(self.label_dict)
        #     # del class_names[-1]
        #     # exclude = np.argsort(ind[:,1])[self.hparams.num_classes:]
        #     conf_matrix_with_unlab = np.zeros((len(include)+1, len(include)+1), dtype=np.int64)
        #     conf_matrix_with_unlab[1:, 1:] = cost_matrix[include][:,include]
        #     conf_matrix_with_unlab[0, 1:] = cost_matrix[exclude].sum(0)[include]
        #     checkpoint_name = self.hparams.checkpoint_dir.split('/')[-1]
        #     # fig, ax = plot_confusion_matrix(conf_mat=conf_matrix[include][:,include].T, colorbar=True, show_absolute=False, show_normed=True, class_names=list(class_names.values()))
            
        #     fig, ax = plot_confusion_matrix(conf_mat=conf_matrix_with_unlab.T, colorbar=True, show_absolute=False, show_normed=True, class_names=list(class_names.values()), figsize=(12,12))
        #     plt.savefig(f'{self.hparams.log_dir}/{checkpoint_name}-conf_mat.png', dpi=300)

class ExpRCTest(ExpMixFineTuning):
    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        # self.target = target
        # self.target_mapping = {}
        # for i, target in enumerate(self.target):
        #     self.target_mapping[i] = target
        
        self.visualization = visualization
        self.save_result = save_result
        # D = len(self.unknown_real_labels)
        D = self.hparams.num_labeled_classes + 1
        
        self.novel_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        self.cost_matrices = [np.zeros((D, D), dtype=np.int64) for _ in range(len(self.novel_thresholds))]
        # self.KMeans = KMeansSklearn(n_clusters=2, verbose=0, n_init='auto')
        return
    
    def test_step(self, data, _): 

        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        # Forward
        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        if self.global_step % self.hparams.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(sp_tensor)
        mask_lab = mapped_labels != self.unknown_label
        feats = out['feats']

        # loss = self.valid_criterion(out['logits'], mapped_labels.long())
        
        preds = F.softmax(out['logits'], dim=1)
        rc_prob = preds[:,-1]
        pred_labels = torch.argmax(preds, dim=1)
        
        pred_novel_masks = [(rc_prob > thresh) for thresh in self.novel_thresholds]
        pred_novel_labels_list = [pred_labels.clone() for _ in self.novel_thresholds]

        rc_protoypes = self.model.encoder.final2.kernel.data


        # to_delete = []
        for i, pred_novel_labels in enumerate(pred_novel_labels_list):
            # pred_labels[pred_novel_masks[i]] = self.unknown_label
            pred_novel_labels[i] = self.unknown_label
            # feats_novel = feats[pred_novel_masks[i]]
            # rc_logits = torch.matmul(feats_novel, rc_protoypes)

            # rc_preds = F.softmax(rc_logits, dim=1)
            # rc_preds = torch.argmax(rc_preds, dim=1)
        for pcd_idx in range(pcd_indexes.shape[0]):
            pcd_labels = mapped_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
            for i, pred_labels in enumerate(pred_novel_labels_list):
                pcd_preds = pred_labels[coords[:,0] == pcd_idx]
                pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
                
                # For Strict-Hungarian
                idxs = np.stack([pcd_preds, pcd_labels], axis=0)
                np.add.at(self.cost_matrices[i], tuple(idxs), 1)

            
            
    
    def test_epoch_end(self, _):
        # mious = []
        for i, cost_matrix in enumerate(self.cost_matrices):
            # For Strict-Hungarian
            conf_base_novel = cost_matrix[:-1,-1].sum()
            conf_novel_base = cost_matrix[-1,:-1].sum()
            conf_novel_novel = cost_matrix[-1,-1].sum()
            
            recall_novel = conf_novel_novel / np.maximum((conf_novel_novel + conf_base_novel), 1e-4)
            precision_novel = conf_novel_novel / np.maximum((conf_novel_novel + conf_novel_base), 1e-4)
            
            # precision, recall = self.get_precision_and_recall_for_novel(conf_precision_recall)
            self.log(f'thershold{self.novel_thresholds[i]}/precision_novel', precision_novel, on_step=False, rank_zero_only=True)
            self.log(f'thershold{self.novel_thresholds[i]}/recall_novel', recall_novel, on_step=False, rank_zero_only=True)
            
            # ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
            # ind = np.vstack(ind).T
            
            # conf_matrix = cost_matrix[:, ind[:,1]]
            # include = np.argsort(ind[:,1])[:self.hparams.num_classes]
            
            # iou = self.get_iou(conf_matrix, include)
            # miou = iou.mean()
            # miou_old = iou[self.known_real_labels].mean()
            # miou_new = iou[self.unknown_real_labels].mean()
            include = np.arange(self.hparams.num_labeled_classes+1)

            iou = self.get_iou(cost_matrix, include)
            miou = iou.mean()

            # known classes와 unknown (novel) classes에 대한 mIoU
            miou_new = iou[self.unknown_label].mean()
            miou_old = iou[:self.unknown_label].mean()
            
            _, output_dict = self.log_metric(miou, iou)
            self.log(f'thershold{self.novel_thresholds[i]}/mIoU', miou, on_step=False, rank_zero_only=True)
            self.log(f'thershold{self.novel_thresholds[i]}/mIoU/old', miou_old, on_step=False, rank_zero_only=True)
            self.log(f'thershold{self.novel_thresholds[i]}/mIoU/new', miou_new, on_step=False, rank_zero_only=True)
            
            for class_, iou_ in output_dict.items():
                if class_ == 'all': continue
                self.log(f'thershold{self.novel_thresholds[i]}/IoU/{class_}', iou_['IoU'], on_step=False, rank_zero_only=True)
    
    def log_metric(self, class_IoU, class_all_IoU, classes=None):
        
        # now make a nice dictionary
        output_dict = {}

        # make python variables
        class_IoU = class_IoU.item()
        class_all_IoU = class_all_IoU.flatten().tolist()

        output_dict["all"] = {}
        output_dict["all"]["IoU"] = class_IoU
        
        for idx, iou in enumerate(class_all_IoU):
            if idx != self.unknown_label:
                if classes is not None:
                    if idx in classes:
                        class_str = self.label_dict[self.label_mapping_inv[idx]]
                        output_dict[class_str] = {}
                        output_dict[class_str]["IoU"] = iou
                else:
                    class_str = self.label_dict[self.label_mapping_inv[idx]]
                    output_dict[class_str] = {}
                    output_dict[class_str]["IoU"] = iou

        mIoU = output_dict["all"]["IoU"]
        return mIoU, output_dict


### (Stage-3) Discover

class ExpMixRealMeanTeacherDiscover(ExpMixExtraFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)

        del self.model

        self.teacher = MinkUNetRC(
            num_labeled=self.hparams.num_labeled_classes,
        )
        self.student =  MinkUNetRC(
            num_labeled=self.hparams.num_labeled_classes,
        )
        self.teacher.encoder.final2 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=3,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        self.student.encoder.final2 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=3,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        if self.hparams.pretrained is not None:
            self.load_pretrained(self.teacher, pretrained=self.hparams.pretrained)
            self.load_pretrained(self.student, pretrained=self.hparams.pretrained)

        self.teacher.encoder.final3 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=self.hparams.num_unlabeled_classes,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        self.student.encoder.final3 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=self.hparams.num_unlabeled_classes,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        
        # Sinkhorn-Knopp algorithm
        self.num_iter_sk = 3 
        self.initial_epsilon_sk = 0.3
        self.final_epsilon_sk = 0.05
        self.queue_start_epoch = 2
        self.hungarian_at_each_step = True 
        self.use_uncertainty_loss = True
        self.use_uncertainty_queue = True
        self.adapting_epsilon_sk = True
        self.uncertainty_percentile = 0.5
        self.hungarian_at_each_step = True 
        self.queue_percentage = 0.1
        self.queue_batches = 10

        self.sk = SinkhornKnopp(
            num_iters=int(self.initial_epsilon_sk), epsilon=self.initial_epsilon_sk
        )

        self.freeze(self.teacher)
    
    def train_dataloader(self):
        lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                config_file=self.hparams.dataset_config,
                split="train",
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                data_idx_file=self.hparams.data_idx_file,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
            )
        
        if self.debug:
            lab_dataset.num_files = 30
            unlab_dataset.num_files = 30

        dataset = swav_wrapper_ssl(lab_dataset, unlab_dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=int(self.hparams.batch_size/2),
            collate_fn=collation_fn_restricted_dataset_two_samples_ssl,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader

    def training_step(self, data, _):
        (
            coords,
            feats,
            _,
            selected_idx,
            mapped_labels,
            _,
            coords1,
            feats1,
            _,
            selected_idx1,
            _,
            _,
            pcd_indexes,

        ) = data['sup']
        
        (
            unsup_coords,
            unsup_feats,
            _,
            unsup_selected_idx,
            _,
            _,
            unsup_coords1,
            unsup_feats1,
            _,
            unsupp_selected_idx1,
            _,
            _,
            unsup_pcd_indexes,
        )= data['unsup']

        unsup_coords[:,0] += len(pcd_indexes)
        unsup_coords1[:,0] += len(pcd_indexes)

        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        coords_cat1 = torch.cat((coords1, unsup_coords1), axis=0)
        feats_cat1 = torch.cat((feats1, unsup_feats1), axis=0)


        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        sp_tensor1 = ME.SparseTensor(features=feats_cat1.float(), coordinates=coords_cat1.int())

        out_t = self.teacher(sp_tensor)
        out_s = self.student(sp_tensor)

        out_t1 = self.teacher(sp_tensor1)
        out_s1 = self.student(sp_tensor1)

        logits_t = out_t['logits']
        logits_s = out_s['logits']

        logits_t1 = out_t1['logits']
        logits_s1 = out_s1['logits']

        feats_t = out_t['feats']
        feats_s = out_s['feats']

        feats_t1 = out_t1['feats']
        feats_s1 = out_s1['feats']

        sup_data_num = 0
        sup_data_num1 = 0
        for i in range(len(selected_idx)):
            sup_data_num += len(selected_idx[i])
        
        for i in range(len(selected_idx1)):
            sup_data_num1 += len(selected_idx1[i])
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_t_unsup = logits_t[sup_data_num:, :]
        feats_t_unsup = feats_t[sup_data_num:, :]

        logits_t_unsup1 = logits_t1[sup_data_num1:, :]
        feats_t_unsup1 = feats_t1[sup_data_num1:, :]

        logits_s_sup = logits_s[:sup_data_num, :]

        logits_s_unsup = logits_s[sup_data_num:, :]
        feats_s_unsup = feats_s[sup_data_num:, :]
        logits_s_unsup1 = logits_s1[sup_data_num1:, :]
        feats_s_unsup1 = feats_s1[sup_data_num1:, :]

        mapped_labels = mapped_labels

        if self.unknown_label in mapped_labels.unique():
            raise ValueError
    
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=out_s['feats'][:sup_data_num, :], labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)
        
        # y_in_normal = self.final(out).F
        # y_out_normal = self.final2(out).F
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)
        
        final_layer_kernel = self.student.encoder.final.kernel
        final_layer2_kernel = self.student.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_s_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_s_sup.shape[0]).unsqueeze(0).to(logits_s_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_s_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_s_sup.shape[0]).long().to(logits_s_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        # Unsupervised Loss MeanTeacher Framework

        prob_t_unsup = F.softmax(logits_t_unsup, dim=1)
        prob_t_unsup1 = F.softmax(logits_t_unsup1, dim=1)
        
        rc_t_unsup = prob_t_unsup[:,-1]
        rc_t_unsup1 = prob_t_unsup1[:,-1]

        pred_t_unsup = torch.argmax(prob_t_unsup, dim=1)
        pred_t_unsup1 = torch.argmax(prob_t_unsup1, dim=1)

        threshold = 0.2
        novel_candidate_mask = rc_t_unsup > threshold
        novel_candidate_mask1 = rc_t_unsup1 > threshold

        pred_t_unsup[novel_candidate_mask] = self.unknown_label
        pred_t_unsup1[novel_candidate_mask1] = self.unknown_label

        # prob_s_unsup = F.softmax(logits_s_unsup, dim=1)
        # pred_s_unsup = torch.argmax(prob_s_unsup, dim=1)
        # novel_pred_labels[~threshold_mask] = -1 # ignore for training
        
        if len(logits_t_unsup) != 0:
            # RC 학습
            # Seg loss 말고 swav loss 줘야할듯
            # unsup_seg_loss = 0.1 * self.criterion(logits_s_unsup, pred_t_unsup.long())
            # loss += unsup_seg_loss
            # self.log("train/unsup_mean_teacher_seg_loss", unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            print("TBD...")
        

        novel_t_prototypes = self.teacher.final3.kernel
        novel_s_prototypes = self.student.final3.kernel 
        
        feats_t_novel = feats_t_unsup[novel_candidate_mask]
        feats_s_novel = feats_s_unsup[novel_candidate_mask]

        feats_t_novel1 = feats_t_unsup1[novel_candidate_mask]
        feats_s_novel1 = feats_s_unsup1[novel_candidate_mask]

        logits_t_novel = torch.matmul(feats_t_novel.detach(), novel_t_prototypes)
        logits_s_novel = torch.matmul(feats_s_novel.detach(), novel_s_prototypes)

        logits_t_novel1 = torch.matmul(feats_t_novel1.detach(), novel_t_prototypes)
        logits_s_novel1 = torch.matmul(feats_s_novel1.detach(), novel_s_prototypes)


        
        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx, _):
        self._update_ema_variables(momentum=0.01)
    
    def _update_ema_variables(self, momentum):
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.mul_(1 - momentum).add_(param_s.data, alpha=momentum)

class ExpMixDiscover(ExpMixExtraFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model.encoder.final3 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=self.hparams.num_unlabeled_classes,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        self.kmeans = KMeansSklearn(n_clusters=self.hparams.num_unlabeled_classes + 1 , verbose=0, n_init='auto')
        self.novel_queue = None
        self.novel_indices= []
        self.novel_queue_batches = 20
        self.use_first_dataloader = False 
        self.use_uncertainty_queue = True
        self.queue_percentage = 0.05
        self.novel_criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(self.hparams.num_unlabeled_classes) / self.hparams.num_unlabeled_classes
        self.novel_criterion.weight = weights
        # self.criterion = torch.nn.CrossEntropyLoss()
        # weights = torch.ones(self.hparams.num_unlabeled_classes) / self.hparams.num_unlabeled_classes
        # self.criterion.weight = weights
        
        
        if self.hparams.dataset == 'nuScenes':
            self.calib_coeff = 0.05 
            self.mix_unsup_seg_coeff = 0.1
            self.novel_coeff = 1

            self.centroid_sampling = True
            self.beta_sampling = True
            self.beta_coeff = 0.5

            self.entropy_minimize = False
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-3
        
        # NOTE: KITTI
        else:
            self.calib_coeff = 0.05 
            self.mix_unsup_seg_coeff = 0.1
            self.novel_coeff = 1

            self.centroid_sampling = True
            self.beta_sampling = True
            self.beta_coeff = 0.5

            self.entropy_minimize = True
            self.id_entropy_coeff = 1
            self.ood_entropy_coeff = 1e-3
    
    def train_dataloader(self):
        if self.use_first_dataloader:
            lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
                nusc=self.nusc,
            )
            unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                    config_file=self.hparams.dataset_config,
                    split="train",
                    voxel_size=self.hparams.voxel_size,
                    downsampling=self.hparams.downsampling,
                    data_idx_file=self.hparams.data_idx_file,
                    # data_idx_file=self.hparams.uncertain_idx_file,
                    augment=True,
                    label_mapping=self.label_mapping,
                    labeled=False,
                    store_unlabeled_label=True,
                    nusc=self.nusc,
                )
            
            if self.debug:
                lab_dataset.num_files = 50
                unlab_dataset.num_files = 50

            with open(self.hparams.uncertain_idx_file, "rb") as f:
                ordered_indices = np.load(f)
        
            unlab_ordered_subset = Subset(unlab_dataset, ordered_indices)

            # dataset = ConcatDataset([lab_dataset, unlab_ordered_subset])

            lab_dataloader = DataLoader(
                dataset=lab_dataset,
                batch_size=self.hparams.batch_size // 2,  # 예: 배치의 절반을 lab 데이터로 설정
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=True,  
            )

            unlab_dataloader = DataLoader(
                dataset=unlab_ordered_subset,
                batch_size=self.hparams.batch_size // 2,  # 나머지 절반을 unlab 데이터로 설정
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                sampler=SequentialSampler(unlab_ordered_subset),  # 순서대로 데이터를 불러옴
                shuffle=False,
            )

            self.lab_dataloader_iter = cycle(lab_dataloader)
            self.unlab_dataloader_iter = cycle(unlab_dataloader)
            
            return [lab_dataloader, unlab_dataloader]
        
        else:
            lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
                nusc=self.nusc,
            )
            unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                    config_file=self.hparams.dataset_config,
                    split="train",
                    voxel_size=self.hparams.voxel_size,
                    downsampling=self.hparams.downsampling,
                    data_idx_file=self.hparams.data_idx_file,
                    augment=True,
                    label_mapping=self.label_mapping,
                    labeled=False,
                    store_unlabeled_label=True,
                    nusc=self.nusc,
                )
            
            if self.debug:
                lab_dataset.num_files = 50
                unlab_dataset.num_files = 50

            lab_dataloader = DataLoader(
                dataset=lab_dataset,
                batch_size=self.hparams.batch_size // 2,  
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=True,  
            )

            unlab_dataloader = DataLoader(
                dataset=lab_dataset,
                batch_size=self.hparams.batch_size // 2,  
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=True,  
            )

            self.lab_dataloader_iter = cycle(lab_dataloader)
            self.unlab_dataloader_iter = cycle(unlab_dataloader)
            
            return [lab_dataloader, unlab_dataloader] 
        
    def on_train_epoch_start(self):
        if self.current_epoch > 0:
            self.use_first_dataloader = False

    def training_step(self, data, _, dataloader_idx=0):
        coords, feats, selected_idx, _, mapped_labels, inverse_maps, pcd_indexes = data[0]
        unsup_coords, unsup_feats, _, _, unsup_mapped_labels, unsup_inverse_maps, unsup_pcd_indexes = data[1]

        if self.unknown_label in mapped_labels:
            breakpoint()

        unsup_pcd_masks = []
        
        for i in range(len(pcd_indexes)):
            unsup_pcd_masks.append(unsup_coords[:, 0] == i)

        unsup_coords[:,0] += len(pcd_indexes)
        
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())

        out = self.model(sp_tensor)
        
        logits = out['logits']
        feats = out['feats']

        sup_data_num = len(selected_idx)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup = logits[:sup_data_num, :]
        feats_sup = feats[:sup_data_num, :]

        logits_unsup = logits[sup_data_num:, :]
        feats_unsup = feats[sup_data_num:, :]

        mapped_labels = mapped_labels

        if self.unknown_label in mapped_labels.unique():
            raise ValueError
    
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        if self.centroid_sampling:
            mix_feats, mix_labels = self.mix_centroid_features(feats=feats_sup, labels=mapped_labels, mask=mask_lab)
        else:
            mix_feats, mix_labels = self.mix_features(feats=feats_sup, labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)

        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        if self.entropy_minimize:
            probs_sup = F.softmax(logits_mix_sup, dim=1)
            rc_probs_sup = probs_sup[:, -1]
            probs_sup = probs_sup[:, :-1]
            id_entropy_loss = -torch.mean(torch.sum(probs_sup * torch.log(probs_sup + 1e-8), dim=1)) * self.id_entropy_coeff
            loss += id_entropy_loss 
            ood_entropy_loss = torch.mean(torch.sum(rc_probs_sup * torch.log(rc_probs_sup + 1e-8))) * self.ood_entropy_coeff
            loss += ood_entropy_loss
            self.log("train/sup_id_entropy_loss", id_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/sup_ood_entropy_loss", ood_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)

        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        # Unsupervised Manner
        if self.centroid_sampling:
            mix_unsup_feats, mix_unsup_labels = self.mix_unsup_centroid(feats=feats_unsup)
        else:
            mix_unsup_feats, mix_unsup_labels = self.mix_unsup_features(feats=feats_unsup, mixing_ratio=self.mixing_ratio_feat)

        prob_unsup = F.softmax(logits_unsup, dim=1)
        rc_unsup = prob_unsup[:,-1]

        pseudo_gt_unsup = torch.argmax(prob_unsup, dim=1)

        threshold = 0.2
        novel_candidate_mask = rc_unsup > threshold

        novel_feats = feats_unsup[novel_candidate_mask]
        pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]

        mix_in_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer_kernel)
        mix_out_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer2_kernel)
        mix_out_normal_2_unsup, _ = torch.max(mix_out_normal_unsup, dim=1, keepdim=True)
        mix_unsup_logits = torch.cat([mix_in_normal_unsup, mix_out_normal_2_unsup], dim =1 )
        logits_mix_unsup = torch.cat([logits_unsup[~novel_candidate_mask], mix_unsup_logits], dim=0)
        mix_mapped_labels_unsup = torch.cat([pseudo_gt_unsup, mix_unsup_labels], dim=0)
        
        mix_unsup_seg_loss = self.criterion(logits_mix_unsup, mix_mapped_labels_unsup.long()) * self.mix_unsup_seg_coeff
        self.log("train/unsup_mix_seg_loss", mix_unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        loss += mix_unsup_seg_loss 
        
        if self.novel_queue is not None:
            novel_feats = torch.vstack((novel_feats, self.novel_queue))
        else:
            self.novel_queue = None
        
        if novel_feats.shape[0] > int(self.hparams.num_unlabeled_classes + 1):
            # Run Kmeans clustering to the novel candidate
            kmeans = self.kmeans.fit(novel_feats.detach().cpu())

            cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
            
            # final_layer_kernel = base prototypes
            base_prototypes = final_layer_kernel.double().detach()
            distances = torch.tensor([
                torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
                for center in cluster_centers
            ], device=self.device)
            
            unrealiable_labels = torch.argmin(distances).item()
            unrealiable_mask = cluster_labels == unrealiable_labels
            # unrealiable_feats = novel_feats[unrealiable_mask]
            reliable_feats = novel_feats[~unrealiable_mask]
            reliable_cluster_labels = cluster_labels[~unrealiable_mask]

            unique_labels = torch.unique(reliable_cluster_labels)

            label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

            # label_mapping을 사용해 reliable_cluster_labels 재정렬
            reliable_cluster_labels = torch.tensor(
                [label_mapping[label.item()] for label in reliable_cluster_labels],
                device=reliable_cluster_labels.device
            )
            
            novel_prototypes = self.model.encoder.final3.kernel        
            
            realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_prototypes)
            realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

            unsup_cost_matrix = np.zeros(
                    (
                        self.hparams.num_unlabeled_classes,
                        self.hparams.num_unlabeled_classes,
                    ), dtype=np.int64
                )
            idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0)
            np.add.at(unsup_cost_matrix, tuple(idxs), 1)

            row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix.max()- unsup_cost_matrix)

            mapping = {row: col for row, col in zip(row_ind, col_ind)}
            mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

            
            # for pseudo_label in range(self.hparams.num_unlabeled_classes):
            #     mask_pseudo = realiable_novel_preds == pseudo_label
            #     for j, real_label in enumerate(self.unknown_real_labels):
            #         mask_real = real_labels[mask_unknown] == real_label
            #         cost_matrix[pseudo_label, j] += torch.logical_and(
            #             mask_pseudo, mask_real
            #         ).sum()
            
            novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
            self.log("train/unsup_novel_unsup_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            loss += novel_unsup_loss
            mean_reliable_feats = torch.mean(reliable_feats, dim=0)
            self.update_queue(mean_reliable_feats)

            if self.entropy_minimize:
                prob_unsup = prob_unsup[:, :-1]
                unsup_id_entropy_loss = -torch.mean(torch.sum(prob_unsup * torch.log(prob_unsup + 1e-8), dim=1)) * self.id_entropy_coeff
                loss += id_entropy_loss 
                unsup_ood_entropy_loss = torch.mean(torch.sum(rc_unsup * torch.log(rc_unsup + 1e-8))) * self.ood_entropy_coeff
                loss += ood_entropy_loss
                self.log("train/unsup_id_entropy_loss", unsup_id_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
                self.log("train/unsup_ood_entropy_loss", unsup_ood_entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
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
        # self.cost_matrix_old = np.zeros((D, D), dtype=np.int64)
        # self.cost_matrix_new = np.zeros((D, D), dtype=np.int64)
        for i in range(len(self.unknown_real_labels)):
            self.label_mapping_inv[self.unknown_label+i] = self.unknown_real_labels[i]
            
    def validation_step(self, data, _):
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
        row_ind, col_ind = linear_sum_assignment(conf_matrix.max()- conf_matrix)
        # breakpoint()
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

        # breakpoint()
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
        #     # breakpoint()
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
        row_ind, col_ind = linear_sum_assignment(conf_matrix)
        # breakpoint()
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

        # breakpoint()
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

class ExpMixDiscoverJoint(ExpMixDiscover):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.use_first_dataloader = False # False = vanilla
        self.calib_coeff = 0.05 
        self.mix_unsup_seg_coeff = 0.5
        self.novel_coeff = 0.002
        self.novel_criterion = torch.nn.CrossEntropyLoss()
        weights = torch.ones(self.hparams.num_classes) / self.hparams.num_classes
        self.novel_criterion.weight = weights
    
    def training_step(self, data, _, dataloader_idx=0):
        coords, feats, selected_idx, _, mapped_labels, inverse_maps, pcd_indexes = data[0]
        unsup_coords, unsup_feats, _, _, _, unsup_inverse_maps, unsup_pcd_indexes = data[1]

        if self.unknown_label in mapped_labels:
            breakpoint()

        unsup_pcd_masks = []
        
        for i in range(len(pcd_indexes)):
            unsup_pcd_masks.append(unsup_coords[:, 0] == i)

        unsup_coords[:,0] += len(pcd_indexes)
        
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())

        out = self.model(sp_tensor)
        
        logits = out['logits']
        feats = out['feats']

        sup_data_num = len(selected_idx)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup = logits[:sup_data_num, :]
        feats_sup = feats[:sup_data_num, :]

        logits_unsup = logits[sup_data_num:, :]
        feats_unsup = feats[sup_data_num:, :]

        mapped_labels = mapped_labels

        if self.unknown_label in mapped_labels.unique():
            raise ValueError
    
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=feats_sup, labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)

        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        # Unsupervised Manner

        # mix_unsup_feats, mix_unsup_labels = self.mix_unsup_features(feats=feats_unsup, mixing_ratio=self.mixing_ratio_feat)

        prob_unsup = F.softmax(logits_unsup, dim=1)
        rc_unsup = prob_unsup[:,-1]

        # pseudo_gt_unsup = torch.argmax(prob_unsup, dim=1)

        threshold = 0.2
        novel_candidate_mask = rc_unsup > threshold

        novel_feats = feats_unsup[novel_candidate_mask]
        # pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]

        # mix_in_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer_kernel)
        # mix_out_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer2_kernel)
        # mix_out_normal_2_unsup, _ = torch.max(mix_out_normal_unsup, dim=1, keepdim=True)
        # mix_unsup_logits = torch.cat([mix_in_normal_unsup, mix_out_normal_2_unsup], dim =1 )
        # logits_mix_unsup = torch.cat([logits_unsup[~novel_candidate_mask], mix_unsup_logits], dim=0)
        # mix_mapped_labels_unsup = torch.cat([pseudo_gt_unsup, mix_unsup_labels], dim=0)
        
        # mix_unsup_seg_loss = self.criterion(logits_mix_unsup, mix_mapped_labels_unsup.long()) * self.mix_unsup_seg_coeff
        # self.log("train/unsup_mix_seg_loss", mix_unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        # loss += mix_unsup_seg_loss 
        
        if self.novel_queue is not None:
            novel_feats = torch.vstack((novel_feats, self.novel_queue))
        else:
            self.novel_queue = None
        
        if novel_feats.shape[0] > self.hparams.num_unlabeled_classes:
            # Run Kmeans clustering to the novel candidate
            kmeans = self.kmeans.fit(novel_feats.detach().cpu())

            cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
            
            # final_layer_kernel = base prototypes
            base_prototypes = final_layer_kernel.double().detach()
            distances = torch.tensor([
                torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
                for center in cluster_centers
            ], device=self.device)
            
            unrealiable_labels = torch.argmin(distances).item()
            unrealiable_mask = cluster_labels == unrealiable_labels
            # unrealiable_feats = novel_feats[unrealiable_mask]
            reliable_feats = novel_feats[~unrealiable_mask]
            reliable_cluster_labels = cluster_labels[~unrealiable_mask]

            unique_labels = torch.unique(reliable_cluster_labels)

            label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

            # label_mapping을 사용해 reliable_cluster_labels 재정렬
            reliable_cluster_labels = torch.tensor(
                [label_mapping[label.item()] for label in reliable_cluster_labels],
                device=reliable_cluster_labels.device
            )
            
            novel_prototypes = self.model.encoder.final3.kernel        
            base_prototypes = self.model.encoder.final.kernel
            
            realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_prototypes)
            realiable_base_logits = torch.matmul(reliable_feats.detach(), base_prototypes)
            realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

            total_reliable_logits = torch.cat((realiable_base_logits, realiable_novel_logits), dim=1)

            unsup_cost_matrix = np.zeros(
                    (
                        self.hparams.num_unlabeled_classes,
                        self.hparams.num_unlabeled_classes,
                    ), dtype=np.int64
                )
            idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0)
            np.add.at(unsup_cost_matrix, tuple(idxs), 1)

            row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix)

            mapping = {row: col for row, col in zip(row_ind, col_ind)}
            mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

            
            # for pseudo_label in range(self.hparams.num_unlabeled_classes):
            #     mask_pseudo = realiable_novel_preds == pseudo_label
            #     for j, real_label in enumerate(self.unknown_real_labels):
            #         mask_real = real_labels[mask_unknown] == real_label
            #         cost_matrix[pseudo_label, j] += torch.logical_and(
            #             mask_pseudo, mask_real
            #         ).sum()
            mapped_preds += self.hparams.num_labeled_classes

            novel_unsup_loss = self.novel_criterion(total_reliable_logits, mapped_preds.long()) * self.novel_coeff
            self.log("train/unsup_novel_unsup_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            loss += novel_unsup_loss
            mean_reliable_feats = torch.mean(reliable_feats, dim=0)
            self.update_queue(mean_reliable_feats)

        return loss

    def validation_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        mask_lab = mapped_labels != self.unknown_label

        out_novel = self.model.forward_discover(sp_tensor)
        logits_novel = out_novel['logits']
        total_preds = torch.argmax(torch.softmax(logits_novel.detach(), dim=1), dim=1)

        for key, value in self.label_mapping_inv.items():
            total_preds[total_preds == key] = -value
        total_preds = -total_preds

        for pcd_idx in range(pcd_indexes.shape[0]):
            pcd_labels = real_labels[coords[:,0] == pcd_idx]
            pcd_labels = pcd_labels[inverse_maps[pcd_idx]].cpu().numpy()
            
            pcd_preds = total_preds[coords[:,0] == pcd_idx]
            pcd_preds = pcd_preds[inverse_maps[pcd_idx]].cpu().numpy()
            
            idxs = np.stack([pcd_preds, pcd_labels], axis=0)
            np.add.at(self.cost_matrix, tuple(idxs), 1)
            
            
            # # For Greedy-Hungarian
            # pcd_mask_lab = mask_lab[coords[:,0] == pcd_idx][inverse_maps[pcd_idx]].cpu().numpy()
            # idxs_old = np.stack([pcd_preds[pcd_mask_lab], pcd_labels[pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_old, tuple(idxs_old), 1)
            
            # idxs_new = np.stack([pcd_preds[~pcd_mask_lab], pcd_labels[~pcd_mask_lab]], axis=0)
            # np.add.at(self.cost_matrix_new, tuple(idxs_new), 1)

        return 

class ExpMixDiscoverSwaV(ExpMixExtraFineTuning):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.model.encoder.final3 = ME.MinkowskiConvolution(
                        self.model.encoder.PLANES[7] * self.model.encoder.BLOCK.expansion,
                        out_channels=self.hparams.num_unlabeled_classes,
                        kernel_size=1,
                        bias=True,
                        dimension=3)
        self.swav_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        weights = torch.ones(int(self.hparams.num_labeled_classes + 1))/int(self.hparams.num_labeled_classes + 1)
        self.swav_criterion.weight = weights
        
        self.kmeans = KMeansSklearn(n_clusters=self.hparams.num_unlabeled_classes + 1 , verbose=0, n_init='auto')
        self.kmeans1 = KMeansSklearn(n_clusters=self.hparams.num_unlabeled_classes + 1 , verbose=0, n_init='auto')
        self.novel_queue = None
        self.debug = self.hparams.debug
    
    def train_dataloader(self):
        lab_dataset = get_dataset_remission(self.hparams.dataset, "disjoint")(
            config_file=self.hparams.dataset_config,
            split="train",
            data_idx_file=self.hparams.data_idx_file,
            voxel_size=self.hparams.voxel_size,
            downsampling=self.hparams.downsampling,
            augment=True,
            label_mapping=self.label_mapping,
            nusc=self.nusc,
        )
        unlab_dataset = get_dataset_remission(self.hparams.dataset, 'disjoint')(
                config_file=self.hparams.dataset_config,
                split="train",
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                data_idx_file=self.hparams.data_idx_file,
                augment=True,
                label_mapping=self.label_mapping,
                labeled=False,
                store_unlabeled_label=True,
                nusc=self.nusc,
            )
        if self.debug:
            lab_dataset.num_files = 30
            unlab_dataset.num_files = 30

        dataset = swav_wrapper_ssl(lab_dataset, unlab_dataset)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=int(self.hparams.batch_size/2),
            collate_fn=collation_fn_restricted_dataset_two_samples_ssl,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

        return dataloader

    def training_step(self, data, _):
        (
            coords,
            feats,
            _,
            selected_idx,
            mapped_labels,
            _,
            coords1,
            feats1,
            _,
            selected_idx1,
            _,
            _,
            pcd_indexes,

        ) = data['sup']
        
        (
            unsup_coords,
            unsup_feats,
            _,
            unsup_selected_idx,
            _,
            unsup_inverse_map,
            unsup_coords1,
            unsup_feats1,
            _,
            unsup_selected_idx1,
            _,
            unsup_inverse_map1,
            unsup_pcd_indexes,
        )= data['unsup']
        unsup_pcd_masks = []
        unsup_pcd_masks1 = []
        for i in range(pcd_indexes.shape[0]):
            unsup_pcd_masks.append(unsup_coords[:, 0] == i)
            unsup_pcd_masks1.append(unsup_coords1[:, 0] == i)
            
        unsup_coords[:,0] += len(pcd_indexes)
        unsup_coords1[:,0] += len(pcd_indexes)

        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        coords_cat1 = torch.cat((coords1, unsup_coords1), axis=0)
        feats_cat1 = torch.cat((feats1, unsup_feats1), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())
        sp_tensor1 = ME.SparseTensor(features=feats_cat1.float(), coordinates=coords_cat1.int())

        out = self.model(sp_tensor)
        out1 = self.model(sp_tensor1)

        logits = out['logits']
        logits1 = out1['logits']

        feats = out['feats']
        feats1 = out1['feats']

        sup_data_num = len(selected_idx)
        sup_data_num1 = len(selected_idx1)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup = logits[:sup_data_num, :]
        feats_sup = feats[:sup_data_num, :]

        logits_unsup = logits[sup_data_num:, :]
        feats_unsup = feats[sup_data_num:, :]

        logits_unsup1 = logits1[sup_data_num1:, :]
        feats_unsup1 = feats1[sup_data_num1:, :]

        mapped_labels = mapped_labels

        if self.unknown_label in mapped_labels.unique():
            raise ValueError
    
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        mix_feats, mix_labels = self.mix_features(feats=feats_sup, labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)

        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        mix_seg_loss = self.criterion(logits_mix_sup, mix_mapped_labels.long())
        loss += mix_seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        results = {
            "train/loss" : loss.detach(),
            "train/sup_mix_loss" : mix_seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        # Unsupervised Manner
        logits_points_unsup = torch.cat([
            logits_unsup[unsup_pcd_masks[i]][unsup_inverse_map[i]]
            for i in range(len(unsup_pcd_masks))
        ], dim=0).to(self.device)
        logits_points_unsup1 = torch.cat([
            logits_unsup1[unsup_pcd_masks1[i]][unsup_inverse_map1[i]]
            for i in range(len(unsup_pcd_masks))
        ], dim=0).to(self.device)
        feats_points_unsup = torch.cat([
            feats_unsup[unsup_pcd_masks[i]][unsup_inverse_map[i]]
            for i in range(len(unsup_pcd_masks))
        ], dim=0).to(self.device)
        feats_points_unsup1 = torch.cat([
            feats_unsup1[unsup_pcd_masks1[i]][unsup_inverse_map1[i]]
            for i in range(len(unsup_pcd_masks1))
        ], dim=0).to(self.device)
        
        prob_unsup = F.softmax(logits_points_unsup, dim=1)
        prob_unsup1 = F.softmax(logits_points_unsup1, dim=1)
        
        rc_unsup = prob_unsup[:,-1]
        rc_unsup1 = prob_unsup1[:,-1]

        pred_unsup = torch.argmax(prob_unsup, dim=1)
        pred_unsup1 = torch.argmax(prob_unsup1, dim=1)

        threshold = 0.2
        novel_candidate_mask = rc_unsup > threshold
        novel_candidate_mask1 = rc_unsup1 > threshold

        # pred_unsup[novel_candidate_mask] = self.unknown_label
        # pred_unsup1[novel_candidate_mask1] = self.unknown_label
        
        novel_points_feats = feats_points_unsup[novel_candidate_mask]
        novel_points_feats1 = feats_points_unsup1[novel_candidate_mask1]
        
        if self.novel_queue is not None:
            novel_points_feats = torch.vstack((novel_points_feats, self.novel_queue))
            novel_points_feats1 = torch.vstack((novel_points_feats1, self.novel_queue))
        else:
            self.novel_queue = None
            
        # Run Kmeans clustering to the novel candidate
        kmeans = self.kmeans.fit(novel_points_feats.detach().cpu())
        kmeans1 = self.kmeans1.fit(novel_points_feats1.detach().cpu())
        
        cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
        # print("novel_feats shape:", novel_feats.shape)
        # print("cluster_labels shape:", cluster_labels.shape)

        cluster_labels1 = torch.tensor(kmeans1.labels_, dtype=torch.int32, device=self.device)
        cluster_centers1 = torch.tensor(kmeans1.cluster_centers_, dtype=torch.double, device=self.device)
        # print("novel_feats1 shape:", novel_feats1.shape)
        # print("cluster_labels1 shape:", cluster_labels1.shape)
        
        # final_layer_kernel = base prototypes
        base_prototypes = final_layer_kernel.double().detach()
        distances = torch.tensor([
            torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
            for center in cluster_centers
        ], device=self.device)
        
        distances1 = torch.tensor([
            torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
            for center in cluster_centers1
        ], device=self.device)
        
        unrealiable_labels = torch.argmin(distances).item()
        unrealiable_labels1 = torch.argmin(distances1).item()
        
        unrealiable_mask = cluster_labels == unrealiable_labels
        unrealiable_mask1 = cluster_labels1 == unrealiable_labels1
        
        unrealiable_feats = novel_points_feats[unrealiable_mask]
        unrealiable_feats1 = novel_points_feats1[unrealiable_mask1]
        
        reliable_feats = novel_points_feats[~unrealiable_mask]
        reliable_feats1 = novel_points_feats1[~unrealiable_mask1]

        # if self.novel_queue is not None:
        #     reliable_feats = torch.vstack((reliable_feats, self.novel_queue))
        #     reliable_feats1 = torch.vstack((reliable_feats1, self.novel_queue))
        # else: 
        #     self.novel_queue = None 
        
        novel_prototypes = self.model.encoder.final3.kernel
        
        novel_logits = torch.matmul(reliable_feats.detach(), novel_prototypes)
        novel_logits1 = torch.matmul(reliable_feats1.detach(), novel_prototypes)
        
        self.swav_loss()
        self.update_queue(

        )
        breakpoint()

        # prob_s_unsup = F.softmax(logits_s_unsup, dim=1)
        # pred_s_unsup = torch.argmax(prob_s_unsup, dim=1)  
        # novel_pred_labels[~threshold_mask] = -1 # ignore for training

        # if self.entropy_minimize:
        #     unsup_probs = F.softmax(logits_novel, dim=1)
        #     unsup_probs = torch.argmax(unsup_probs, dim=1)
        #     entropy_loss = -torch.mean(torch.sum(unsup_probs * torch.log(unsup_probs + 1e-8), dim=-1)) * 0.1
        #     loss += entropy_loss 
        #     self.log("train/unsup_entropy_loss", entropy_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def swav_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        idx_logits: torch.Tensor,
        idx_targets: torch.Tensor,
        pcd_mask_logits: torch.Tensor,
        pcd_mask_targets: torch.Tensor,
    ):
        """
        Evaluates the loss function of the predicted logits w.r.t. the targets

        :param logits: predicted logits for the first augmentation of the point clouds
        :param targets: targets for the second augmentation of the point clouds
        :param idx_targets: indexes of the selected points in the second augmentation of the point clouds
        :param pcd_mask_logits: mask to separate the different point clouds in the batch
        :param pcd_mask_targets: mask to separate the different point clouds in the batch
        """
        # Remove the heads loop to simplify the function without self.num_heads
        total_loss = 0

        for pcd in range(len(pcd_mask_logits)):
            pcd_logits = logits[pcd_mask_logits[pcd]]
            pcd_targets = targets[pcd_mask_targets[pcd]]
            
            # Apply filtering based on idx_logits and idx_targets
            logit_shape = pcd_logits.shape[0]
            target_shape = pcd_targets.shape[0]
            
            mask_logits = torch.isin(idx_logits[pcd_mask_logits[pcd]], idx_targets[pcd_mask_targets[pcd]])
            mask_targets = torch.isin(idx_targets[pcd_mask_targets[pcd]], idx_logits[pcd_mask_logits[pcd]])
            pcd_logits = pcd_logits[mask_logits]
            pcd_targets = pcd_targets[mask_targets]

            # Log the percentage of points in common
            perc_to_log = (
                pcd_logits.shape[0] / logit_shape
                + pcd_targets.shape[0] / target_shape
            ) / 2
            self.log("utils/points_in_common", perc_to_log)

            # Calculate loss without weighted cross-entropy
            loss = self.swav_criterion(pcd_logits, pcd_targets)
            if self.use_uncertainty_loss:
                loss = loss[loss > 0]
            
            # Sum the loss for this point cloud
            total_loss += loss.sum()

        # Return the mean loss across all point clouds
        mean_loss = total_loss / len(pcd_mask_logits)
        return mean_loss


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
        if self.sk_queue is None:
            self.sk_queue = feats[mask]
            self.sk_indices.append(n_feats_to_retain)
            return

        if len(self.sk_indices) < self.queue_batches:
            self.sk_queue = torch.vstack((feats[mask], self.sk_queue))
            self.sk_indices.insert(0, n_feats_to_retain)
        else:
            self.sk_queue = torch.vstack(
                (feats[mask], self.sk_queue[: -self.sk_indices[-1]])
            )
            self.sk_indices.insert(0, n_feats_to_retain)
            del self.sk_indices[-1]

class ExpDiscover(ExpMixDiscover):
    def __init__(self, label_mapping, label_mapping_inv, unknown_label, **kwargs):
        super().__init__(label_mapping, label_mapping_inv, unknown_label, **kwargs)
        self.mix_unsup_seg_coeff = 0.1
        self.novel_coeff = 1
        
    def train_dataloader(self):
        if self.use_first_dataloader:
            lab_dataset = get_dataset_remission(self.hparams.dataset, "finetuning")(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
                nusc=self.nusc,
            )
            unlab_dataset = get_dataset_remission(self.hparams.dataset, 'finetuning')(
                    config_file=self.hparams.dataset_config,
                    split="train",
                    voxel_size=self.hparams.voxel_size,
                    downsampling=self.hparams.downsampling,
                    data_idx_file=self.hparams.data_idx_file,
                    # data_idx_file=self.hparams.uncertain_idx_file,
                    augment=True,
                    label_mapping=self.label_mapping,
                    labeled=False,
                    store_unlabeled_label=True,
                    nusc=self.nusc,
                )
            
            if self.debug:
                lab_dataset.num_files = 50
                unlab_dataset.num_files = 50

            with open(self.hparams.uncertain_idx_file, "rb") as f:
                ordered_indices = np.load(f)
        
            unlab_ordered_subset = Subset(unlab_dataset, ordered_indices)

            # dataset = ConcatDataset([lab_dataset, unlab_ordered_subset])

            lab_dataloader = DataLoader(
                dataset=lab_dataset,
                batch_size=self.hparams.batch_size // 2,  # 예: 배치의 절반을 lab 데이터로 설정
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=True,  
            )

            unlab_dataloader = DataLoader(
                dataset=unlab_ordered_subset,
                batch_size=self.hparams.batch_size // 2,  # 나머지 절반을 unlab 데이터로 설정
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                sampler=SequentialSampler(unlab_ordered_subset),  # 순서대로 데이터를 불러옴
                shuffle=False,
            )

            self.lab_dataloader_iter = cycle(lab_dataloader)
            self.unlab_dataloader_iter = cycle(unlab_dataloader)
            
            return [lab_dataloader, unlab_dataloader]
        
        else:
            lab_dataset = get_dataset_remission(self.hparams.dataset, "finetuning")(
                config_file=self.hparams.dataset_config,
                split="train",
                data_idx_file=self.hparams.data_idx_file,
                voxel_size=self.hparams.voxel_size,
                downsampling=self.hparams.downsampling,
                augment=True,
                label_mapping=self.label_mapping,
                nusc=self.nusc,
            )
            unlab_dataset = get_dataset_remission(self.hparams.dataset, 'finetuning')(
                    config_file=self.hparams.dataset_config,
                    split="train",
                    voxel_size=self.hparams.voxel_size,
                    downsampling=self.hparams.downsampling,
                    data_idx_file=self.hparams.data_idx_file,
                    augment=True,
                    label_mapping=self.label_mapping,
                    labeled=False,
                    store_unlabeled_label=True,
                    nusc=self.nusc,
                )
            
            if self.debug:
                lab_dataset.num_files = 50
                unlab_dataset.num_files = 50

            lab_dataloader = DataLoader(
                dataset=lab_dataset,
                batch_size=self.hparams.batch_size // 2,  
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=True,  
            )

            unlab_dataloader = DataLoader(
                dataset=lab_dataset,
                batch_size=self.hparams.batch_size // 2,  
                collate_fn=collation_fn_restricted_dataset,
                num_workers=self.hparams.num_workers,
                shuffle=True,  
            )

            self.lab_dataloader_iter = cycle(lab_dataloader)
            self.unlab_dataloader_iter = cycle(unlab_dataloader)
            
            return [lab_dataloader, unlab_dataloader] 
        
    def training_step(self, data, _, dataloader_idx=0):
        coords, feats, selected_idx, _, mapped_labels, inverse_maps, pcd_indexes = data[0]
        unsup_coords, unsup_feats, _, _, unsup_mapped_labels, unsup_inverse_maps, unsup_pcd_indexes = data[1]

        if self.unknown_label in mapped_labels:
            breakpoint()

        unsup_pcd_masks = []
        
        for i in range(len(pcd_indexes)):
            unsup_pcd_masks.append(unsup_coords[:, 0] == i)

        unsup_coords[:,0] += len(pcd_indexes)
        
        coords_cat = torch.cat((coords, unsup_coords), axis=0)
        feats_cat = torch.cat((feats, unsup_feats), axis=0)

        sp_tensor = ME.SparseTensor(features=feats_cat.float(), coordinates=coords_cat.int())

        out = self.model(sp_tensor)
        
        logits = out['logits']
        feats = out['feats']

        sup_data_num = len(selected_idx)
        
        # logits_t_sup = logits_t[:sup_data_num, :]
        logits_sup = logits[:sup_data_num, :]
        feats_sup = feats[:sup_data_num, :]

        logits_unsup = logits[sup_data_num:, :]
        feats_unsup = feats[sup_data_num:, :]

        mapped_labels = mapped_labels

        if self.unknown_label in mapped_labels.unique():
            raise ValueError
    
        mask_lab = mapped_labels != self.unknown_label
        
        # mixed feature aggregation (supervised loss)
        loss = 0
        # mix_feats, mix_labels = self.mix_features(feats=feats_sup, labels=mapped_labels, mask=mask_lab, mixing_ratio=self.mixing_ratio_feat)

        final_layer_kernel = self.model.encoder.final.kernel
        final_layer2_kernel = self.model.encoder.final2.kernel

        # mix_in_normal = torch.matmul(mix_feats.detach(), final_layer_kernel)
        # mix_out_normal = torch.matmul(mix_feats.detach(), final_layer2_kernel)
        # mix_out_normal_2, _ = torch.max(mix_out_normal, dim=1, keepdim=True)
        # mix_logits = torch.cat([mix_in_normal, mix_out_normal_2], dim =1 )

        # logits_mix_sup = torch.cat([logits_sup, mix_logits], dim=0)
        # mix_mapped_labels = torch.cat([mapped_labels, mix_labels], dim=0)

        seg_loss = self.criterion(logits_sup, mapped_labels.long())
        loss += seg_loss

        labels_tmp = mapped_labels.clone()
        labels_tmp[mapped_labels == self.unknown_label] = -1
        index_tmp = torch.arange(0, logits_sup.shape[0]).unsqueeze(0).to(logits_sup.device)
        index_tmp = torch.cat([index_tmp, labels_tmp.unsqueeze(0)], dim=0)

        preds_dummy = logits_sup.clone()
        preds_dummy[index_tmp.chunk(chunks=2, dim=0)] = -1e9

        labels_dummy = torch.ones(logits_sup.shape[0]).long().to(logits_sup.device) * self.unknown_label
        labels_dummy[labels_tmp.squeeze() == -1] = -1

        calib_loss = self.calib_coeff * self.criterion(preds_dummy, labels_dummy)
        loss += calib_loss

        results = {
            "train/loss" : loss.detach(),
            "train/sup_seg_loss" : seg_loss.detach(),
            "train/sup_calib_loss" : calib_loss.detach(),
        }

        self.log_dict(results, on_step=True, on_epoch=True, sync_dist=True)

        # Unsupervised Manner
        # mix_unsup_feats, mix_unsup_labels = self.mix_unsup_features(feats=feats_unsup, mixing_ratio=self.mixing_ratio_feat)

        prob_unsup = F.softmax(logits_unsup, dim=1)
        rc_unsup = prob_unsup[:,-1]

        pseudo_gt_unsup = torch.argmax(prob_unsup, dim=1)

        threshold = 0.2
        novel_candidate_mask = rc_unsup > threshold

        novel_feats = feats_unsup[novel_candidate_mask]
        pseudo_gt_unsup = pseudo_gt_unsup[~novel_candidate_mask]

        # mix_in_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer_kernel)
        # mix_out_normal_unsup = torch.matmul(mix_unsup_feats.detach(), final_layer2_kernel)
        # mix_out_normal_2_unsup, _ = torch.max(mix_out_normal_unsup, dim=1, keepdim=True)
        # mix_unsup_logits = torch.cat([mix_in_normal_unsup, mix_out_normal_2_unsup], dim =1 )
        # logits_mix_unsup = torch.cat([logits_unsup[~novel_candidate_mask], mix_unsup_logits], dim=0)
        # mix_mapped_labels_unsup = torch.cat([pseudo_gt_unsup, mix_unsup_labels], dim=0)
        
        # mix_unsup_seg_loss = self.criterion(logits_mix_unsup, mix_mapped_labels_unsup.long()) * self.mix_unsup_seg_coeff
        # self.log("train/unsup_mix_seg_loss", mix_unsup_seg_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
        # loss += mix_unsup_seg_loss 
        
        if self.novel_queue is not None:
            novel_feats = torch.vstack((novel_feats, self.novel_queue))
        else:
            self.novel_queue = None
        
        if novel_feats.shape[0] > 3:
            # Run Kmeans clustering to the novel candidate
            kmeans = self.kmeans.fit(novel_feats.detach().cpu())

            cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.int32, device=self.device)
            cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.double, device=self.device)
            
            # final_layer_kernel = base prototypes
            base_prototypes = final_layer_kernel.double().detach()
            distances = torch.tensor([
                torch.cdist(center.unsqueeze(0), base_prototypes.T).sum()
                for center in cluster_centers
            ], device=self.device)
            
            unrealiable_labels = torch.argmin(distances).item()
            unrealiable_mask = cluster_labels == unrealiable_labels
            # unrealiable_feats = novel_feats[unrealiable_mask]
            reliable_feats = novel_feats[~unrealiable_mask]
            reliable_cluster_labels = cluster_labels[~unrealiable_mask]

            unique_labels = torch.unique(reliable_cluster_labels)

            label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}

            # label_mapping을 사용해 reliable_cluster_labels 재정렬
            reliable_cluster_labels = torch.tensor(
                [label_mapping[label.item()] for label in reliable_cluster_labels],
                device=reliable_cluster_labels.device
            )
            
            novel_prototypes = self.model.encoder.final3.kernel        
            
            realiable_novel_logits = torch.matmul(reliable_feats.detach(), novel_prototypes)
            realiable_novel_preds = torch.argmax(realiable_novel_logits, dim=1)

            unsup_cost_matrix = np.zeros(
                    (
                        self.hparams.num_unlabeled_classes,
                        self.hparams.num_unlabeled_classes,
                    ), dtype=np.int64
                )
            idxs = np.stack([realiable_novel_preds.cpu().numpy(), reliable_cluster_labels.cpu().numpy()], axis=0)
            np.add.at(unsup_cost_matrix, tuple(idxs), 1)

            row_ind, col_ind = linear_sum_assignment(unsup_cost_matrix)

            mapping = {row: col for row, col in zip(row_ind, col_ind)}
            mapped_preds = torch.tensor([mapping[pred.item()] for pred in realiable_novel_preds], device=self.device)

            
            # for pseudo_label in range(self.hparams.num_unlabeled_classes):
            #     mask_pseudo = realiable_novel_preds == pseudo_label
            #     for j, real_label in enumerate(self.unknown_real_labels):
            #         mask_real = real_labels[mask_unknown] == real_label
            #         cost_matrix[pseudo_label, j] += torch.logical_and(
            #             mask_pseudo, mask_real
            #         ).sum()
            novel_unsup_loss = self.novel_criterion(realiable_novel_logits, mapped_preds.long()) * self.novel_coeff
            self.log("train/unsup_novel_unsup_loss", novel_unsup_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            loss += novel_unsup_loss
            mean_reliable_feats = torch.mean(reliable_feats, dim=0)
            self.update_queue(mean_reliable_feats)

        return loss
    

    def test_setup(self, visualization, save_result, ood_evaluation, evaluation_method):
        
        self.features = []
        self.mapped_labels = []
        return 
    
    def test_step(self, data, _):
        coords, feats, real_labels, _, mapped_labels, inverse_maps, pcd_indexes = data

        sp_tensor = ME.SparseTensor(features=feats.float(), coordinates=coords.int())
        
        out = self.model(sp_tensor)
        mask_lab = mapped_labels != self.unknown_label
        feats = out['feats']
        self.features.append(feats)
        self.mapped_labels.append(mapped_labels)

        return 
    
    def test_epoch_end(self, _):
        
        self.tsne = TSNE(n_components=2)

        self.features = torch.cat(self.features)
        self.mapped_labels = torch.cat(self.mapped_labels)

        tsne_result = self.tsne.fit_transform(self.features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.mapped_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label="Class Label")
        plt.title("t-SNE Visualization of Model Features")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.savefig(self.logger.log_dir, format="png")

        plt.close()
        
        return