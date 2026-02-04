"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

"""
New los updated by : Minseok Kim (alstjrx1x1@kaist.ac.kr)
2024-10-17
"""
# from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


# NOTE: noramlize를 없애야할거같긴함

class HybridDistanceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(HybridDistanceCrossEntropyLoss, self).__init__()
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, features, prototypes, labels, normalized=True):
        if normalized:
            features = features.contiguous()
            prototypes = prototypes.contiguous()
            features_normed = F.normalize(features, p=2, dim=1)
            # prototypes_normed = F.normalize(prototypes, p=2, dim=1)
            # dist = torch.cdist(features, prototypes.T, p=2) ** 2 
            l2_dist = torch.cdist(features_normed, prototypes, p=2) ** 2 
            dot_product = torch.matmul(features_normed, prototypes.T) 
            hybrid_dist = l2_dist - dot_product
            loss = F.cross_entropy(-hybrid_dist, labels)
        
            return loss
        
        else:
            # NOTE: features.shape -> [N, 96], prototypes.shape -> [96, num_classes]

            features = features.contiguous()
            
            prototypes = prototypes.contiguous()
            l2_dist = torch.cdist(features, prototypes, p=2) ** 2 
           
            # prototypes_normed = F.normalize(prototypes.T, p=2, dim=1)
            # norm_features = torch.norm(features, p=2, dim=1, keepdim=True)
            # norm_prototypes = torch.norm(prototypes, p=2, dim=1, keepdim=True).T
            # 작은 값을 clamp하여 0으로 나누는 것을 방지

            # breakpoint()
            # cosine_similarity = torch.matmul(features, prototypes.T) / (torch.norm(features, p=2, dim=1, keepdim=True) * torch.norm(prototypes, p=2, dim=1, keepdim=True).T)
            dot_product = torch.matmul(features, prototypes.T)
            # cosine_similarity = cosine_similarity / 
            hybrid_dist = l2_dist - dot_product
            # log_probs = F.log_softmax(-hybrid_dist, dim=1)
            # loss = F.nll_loss(log_probs, labels)
            loss = F.cross_entropy(-hybrid_dist, labels)
        
            return loss

class AttractiveLoss(nn.Module):
    def __init__(self):
        super(AttractiveLoss, self).__init__()

    def forward(self, features, prototypes, labels):
        # Normalize features and prototypes
        features = F.normalize(features, p=2, dim=1)
        prototypes = prototypes.T
        # prototypes = F.normalize(prototypes, p=2, dim=1)

        # Compute distances for all features to their corresponding prototypes
        prototype_labels = prototypes[:, labels.long()]  # Map features to their class prototypes
        distances = torch.norm(features - prototype_labels.T, p=2, dim=1) ** 2
        # torch.cdist(features, prototype_labels.T, p=2)**2

        # Compute mean distance as loss
        loss = distances.mean()
        if torch.isnan(loss):
            breakpoint()
        return loss
    
# class AttractiveLoss(nn.Module):
#     def __init__(self):
#         super(AttractiveLoss, self).__init__()

#     def forward(self, features, prototypes, labels):
#         dist = []
#         features = features.contiguous()
#         # prototypes = prototypes.T
#         prototypes = prototypes.contiguous()
#         features_normed = F.normalize(features, p=2, dim=1)
#         # prototypes_normed = F.normalize(prototypes, p=2, dim=1)
        
#         # 여기 for 문이 너무 느리게 만드는 거 같긴함
#         for label in labels.unique():
#             label_mask = labels == label
#             masked_features = features_normed[label_mask]
#             prototype = prototypes[label]
#             dist.append(sum(torch.cdist(masked_features, prototype.T.unsqueeze(0), p=2) ** 2) / len(masked_features))
        
#         dist = torch.stack(dist)
#         loss = dist.mean()
#         if torch.isnan(loss):
#             breakpoint()
#         return loss 

class RegularizationTerm(nn.Module):
    def __init__(self):
        super(RegularizationTerm, self).__init__()
    
    def forward(self, prototypes):
        reg_loss = torch.mean(torch.abs(prototypes-1).sum()) / (prototypes.shape[0] * prototypes.shape[1])
        if torch.isnan(reg_loss):
            breakpoint()
        return reg_loss

class AdvLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AdvLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, synthetic_features, prototypes):
        """
        synthetic_features: Tensor of shape (Ns, feature_dim) - synthetic unseen-class features
        prototypes: Tensor of shape (C, feature_dim) - prototypes for each seen class
        
        Returns:
        adv_loss: Calculated adversarial loss
        """
        # Ensure the tensors are contiguous
        synthetic_features = synthetic_features.contiguous()
        prototypes = prototypes.contiguous()
        features_normed = F.normalize(synthetic_features, p=2, dim=1)
        # prototypes_normed = F.normalize(prototypes, p=2, dim=1)

        # Calculate the squared L2 distance between synthetic features and prototypes
        distances = torch.cdist(features_normed, prototypes, p=2) ** 2  # Shape: (Ns, C)
        
        # Compute the Softmax of the distances along the prototype axis (dim=1)
        softmax_distances = F.softmax(-distances, dim=1)  # Shape: (Ns, C)

        if softmax_distances.numel() == 0:
            return torch.tensor(0.0, device=synthetic_features.device)
        # Compute the log of the Softmax values (with epsilon for numerical stability)
        log_softmax_distances = torch.log(softmax_distances + self.epsilon)

        # Calculate the adversarial loss (Adv)
        adv_loss = -torch.mean(torch.sum(softmax_distances * log_softmax_distances, dim=1)) / prototypes.size(0)
        # print(f"adv loss value : {adv_loss}")
        # if torch.isnan(adv_loss).any():
        #     breakpoint()
        
        return adv_loss

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConLossWithAuxiliary(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLossWithAuxiliary, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, aux_features):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        
        # NOTE: include auxiliary features in contrast_feature (for negative)
        contrast_feature = torch.cat((contrast_feature, aux_features), dim=0)
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # NOTE include mask for auxliary features (for negative)
        mask = torch.cat((mask, torch.zeros((mask.shape[0], aux_features.shape[0]), device=mask.device)), dim=1)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MetricSupConLoss(nn.Module):
    """
    Customized Supervised Contrastive Loss with different metrics.
    When the `metric` is `dot`, it degenerates to original supervised contrastive loss.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, metric='dot'):
        super(MetricSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.metric = metric

    def forward(self, features, labels=None, mask=None, ignore=None, reduction=True):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        if ignore is not None:
            ignore = ignore.contiguous().view(-1, 1)
            if ignore.shape[0] != batch_size:
                raise ValueError('Num of ignore does not match num of features')
            ignore_mask = torch.eq(ignore, ignore.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if self.metric == 'dot':
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            
            logits = anchor_dot_contrast - logits_max.detach()
        elif self.metric == 'reverse_dot':
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            
            logits = -anchor_dot_contrast - logits_max.detach()
        elif self.metric == 'sign_l2':
            anchor_l2_contrast = -torch.div(
                (anchor_feature.tanh().unsqueeze(1) - contrast_feature.tanh().unsqueeze(0)).pow(2).mean(-1),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_l2_contrast, dim=1, keepdim=True)
            
            logits = anchor_l2_contrast - logits_max.detach() # Previously (-) Exists...
        elif self.metric == 'reverse_sign_l2':
            anchor_l2_contrast = -torch.div(
                (anchor_feature.tanh().unsqueeze(1) - contrast_feature.tanh().unsqueeze(0)).pow(2).mean(-1),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_l2_contrast, dim=1, keepdim=True)
            
            logits = -anchor_l2_contrast - logits_max.detach() # Previously (-) Exists...
        else:
            raise NotImplementedError()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        if ignore is not None:
            ignore_mask = ignore_mask.repeat(anchor_count, contrast_count)
            exp_logits = torch.exp(logits) * logits_mask * ignore_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        else:
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if not reduction:
            return loss.view(anchor_count, batch_size)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.split_idx = None

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        
        student_out = student_output / self.student_temp
        if self.split_idx is None:
            student_out = student_out.chunk(self.ncrops)
        else:
            student_out = torch.split(student_out, self.split_idx, dim=1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        pseudo_t = torch.cat([torch.argmax(q, dim=1) for q in teacher_out])
        pseudo_s = torch.cat([torch.argmax(s, dim=1) for s in student_out])
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
        
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss, pseudo_t, pseudo_s
        # return total_loss

