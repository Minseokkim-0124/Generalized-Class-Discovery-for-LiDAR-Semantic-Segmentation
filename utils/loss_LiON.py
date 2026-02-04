'''

Date: 2023-03-22 21:24:29
LastEditTime: 2023-07-01 17:47:11

Description: 
FilePath: /openset_anomaly_detection/pad_losses/gambler_loss.py
have a nice day
'''


import sys
sys.path.append('../')
import sys
import numpy as np
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")



# from scipy.ndimage import gaussian_filter

import torch.nn.functional as F

from IPython import embed


class Gambler(torch.nn.Module):
    def __init__(self, reward, device, ood_reg=.1,
        valid_class_num = 16,unknown_cls_idx=1,novel_class_num = 4,novel_class_list = None):

        super(Gambler, self).__init__()
        self.reward = torch.tensor([reward]).to(device)
        self.ood_reg = ood_reg
        self.device = device
        # self.gaussian_layer_3d = GaussianLayer_3D(device=device)
        self.valid_class_num = valid_class_num
        self.unknown_cls_idx = unknown_cls_idx
        self.novel_class_num = novel_class_num
        assert novel_class_list is not None
        self.novel_class_list = novel_class_list

    def forward(self, pred, targets, wrong_sample=False):
        # Exclude index 0 from softmax probability computation
        #? does it need to exclude the unknown_clss[1:] ?
        
        pred[:, 0] *= 0
        pred[:, 0] -= 99999.0

        targets = targets.unsqueeze(1)

        pred_prob = torch.softmax(pred, dim=1)
        pred_prob = torch.clamp(pred_prob, min=1e-7)


        assert torch.all(pred_prob > 0), print(pred_prob[pred_prob <= 0])
        assert torch.all(pred_prob <= 1), print(pred_prob[pred_prob > 1])


        # ID and OOD probabilities
        #!+==========================================================
        true_pred, reservation = torch.hstack([pred_prob[:, :self.unknown_cls_idx], pred_prob[:, self.unknown_cls_idx+1:]]), pred_prob[:, self.unknown_cls_idx]#* unknown_cls_idx should set as 1 in nuscenes 
        # true_pred: bs,cls,x,y,z; reservation: bs,x,y,z
        reservation = reservation.unsqueeze(1)
        

        # compute the reward via the energy score. Exclude class 0
        reward = torch.logsumexp(torch.hstack([pred[:, 1:self.unknown_cls_idx], pred[:, self.unknown_cls_idx+1:]]), dim=1).pow(2)#* why not minus (as paper shown)
        # reward: bs,x,y,z
        #!+==========================================================


        """"
        unstable error 
        """
        # 3D gaussian smoothing
        if reward.nelement() > 0:
            reward = gaussian_blur_3d(reward)
        else:
            reward = self.reward



        if wrong_sample:  # if here's ood pixels inside the image
            reservation = torch.div(reservation, reward)

            #!+==========================================================
            mask = targets == self.unknown_cls_idx
            #!+==========================================================

            # mask out each of the ood output channel
            # Exclude class 0 by left appending a mask of all False's
            
            """"
            #* all_class_num (include noise) -  noval_class_num   or valid_class_num - noval_class_num
            
            ignore 0 channel 
            """
            
            # mask_for_reserve_boosting_energy = torch.hstack([(mask.unsqueeze(1) & False),  mask.unsqueeze(1).repeat(1, 19-1, 1, 1,1)])
            mask_for_reserve_boosting_energy = torch.hstack([(mask & False),  mask.repeat(1, self.valid_class_num - self.novel_class_num , 1, 1,1)])

            reserve_boosting_energy = torch.add(true_pred, reservation)[mask_for_reserve_boosting_energy]

            gambler_loss_out = torch.tensor([.0], device=self.device)
            if reserve_boosting_energy.nelement() > 0:
                reserve_boosting_energy = torch.clamp(reserve_boosting_energy, min=1e-7).log()
                gambler_loss_out = self.ood_reg * reserve_boosting_energy#? minus?  

            
            """"
            gambler loss for in-lier pixels
            ignore [0, unknown_cls_idx]
            """
            void_mask = targets == 0  #??? is it nessary ?
            targets[void_mask] = 0  # make void pixel to 0
            targets[mask] = 0  # make ood pixel to 0

            """
                忽略unknown_cls_idx ,然后前移一个channel

                对prediction 数据进行切片会导致下面这行出问题, 下标越界,  
            """

            #!========================================================================================================================================
            # print(targets.unique())
            shifted_targets = targets.clone()
            # print(shifted_targets.unique())
            for n_cls_idx in  self.novel_class_list[::-1]:
                shifted_targets = shifted_targets - torch.tensor((shifted_targets > n_cls_idx), dtype=int).to(targets.device)
                # print(n_cls_idx,shifted_targets.unique())
            # exit(0)
            #!========================================================================================================================================

            gambler_loss_in = torch.gather(true_pred, index=shifted_targets, dim=1)

            #? why add as above do ? 
            gambler_loss_in = torch.add(gambler_loss_in, reservation)

            # exclude the ood pixel mask and void pixel mask
            intermediate_value = gambler_loss_in[(~mask) & (~void_mask)]
            assert not torch.any(intermediate_value <= 0), "nan check 3"

            gambler_loss_in = intermediate_value.log()
            return -(gambler_loss_in.mean() + gambler_loss_out.mean())
        else:
            """"
                not  OOD sample in such batch. 
            """
            reservation = torch.div(reservation, reward)
            mask = targets == 0
            targets[mask] = 0
            
            
            """
                对prediction 数据进行切片会导致下面这行出问题, 下标越界
            """
            
            #!========================================================================================================================================
            # print(targets.unique())
            shifted_targets = targets.clone()
            # print(shifted_targets.unique())
            for n_cls_idx in  self.novel_class_list[::-1]:
                shifted_targets = shifted_targets - torch.tensor((shifted_targets > n_cls_idx), dtype=int).to(targets.device)
            #     print(n_cls_idx,shifted_targets.unique())
            # exit(0)
            #!========================================================================================================================================
            
            
            gambler_loss = torch.gather(true_pred, index=shifted_targets, dim=1)
            
            gambler_loss = torch.add(gambler_loss, reservation)

            intermediate_value = gambler_loss[~mask]
            gambler_loss = intermediate_value.log()
            assert not torch.any(torch.isnan(gambler_loss)), "nan check"

            return -gambler_loss.mean()


'''
description:   gaussian blur a voxel based on static kernel,  
param {*} voxel: [B,in_channel, vW,vL,vH]
param {*} kernel_size: 
param {*} sigma
return {*}
'''
def gaussian_blur_3d(voxel, kernel_size=7, sigma=1):
    voxel =voxel.unsqueeze(1)
    device = voxel.device
    # 生成高斯卷积核
    kernel = torch.zeros(kernel_size, kernel_size, kernel_size).to(device)
    center = kernel_size // 2
    variance = sigma**2
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                x, y, z = i - center, j - center, k - center
                kernel[i, j, k] = 1 / (2 * np.pi * variance)**0.5 * np.exp(-(x**2 + y**2 + z**2) / (2 * variance))
    kernel /= torch.sum(kernel)

    # 扩展卷积核的通道维度
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(voxel.shape[1],1, 1, 1, 1)

    # padding，保证卷积后尺寸不变
    padding = kernel_size // 2
    voxel = F.pad(voxel, [padding] * 6, mode='reflect')

    # 执行卷积操作
    voxel = F.conv3d(voxel, kernel, groups=voxel.shape[1])
    
    return voxel

def smooth(arr, lamda1):
    new_array = arr
    copy_of_arr = 1 * arr

    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]

    # added the third direction for 3D points
    arr_3 = torch.zeros_like(copy_of_arr)
    arr_3[:, :, :, :-1] = copy_of_arr[:, :, :, 1:]
    arr_3[:, :, :, -1] = copy_of_arr[:, :, :, -1]

    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2) + torch.sum((arr_3 - copy_of_arr) ** 2)) / 3
    return lamda1 * loss

# TODO: Should it be calculated one by one for each point cloud in the batch?
def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss



""""
ensure there are gap  between common class and noval class. 
more specifically, the noval class should has large energe while the common classes has low energy
"""
def energy_loss(logits, targets,ood_ind=5):
    # ood_ind = 5
    void_ind = 0
    T = 1.
    m_in = -12
    m_out = -6
    # Exclude class 0 from energy computation
    #!+====================================
    """"
    why not process the logits[:,ood_ind]
    """
    
    in_distribution_logits = torch.hstack([logits[:, 1:ood_ind],  logits[:, ood_ind+1:]])
    #!+====================================

    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:#* not OOD sample in the label 
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


""""

"""
def dynamic_energy_loss(logits, targets,ood_ind=5,details_targets=None,m_out_max = 0,resized_point_label = 20):
    shapenet_object_point_label = resized_point_label+1

    # ood_ind = 5
    void_ind = 0
    T = 1.
    m_in = -12
    m_out = -6
    # Exclude class 0 from energy computation
    #!+====================================
    """"
    why not process the logits[:,ood_ind]
    """
    in_distribution_logits = torch.hstack([logits[:, 1:ood_ind],  logits[:, ood_ind+1:]])
    #!+====================================

    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0: #* not OOD sample in the label 
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        
        #* in distribution  energy  and the resized point energy (20)
        tmp =  torch.pow(F.relu(Ec_in - m_in), 2).mean()  
        if (details_targets == resized_point_label).sum() != 0 :
            tmp += torch.pow(F.relu(m_out - energy[details_targets == resized_point_label]), 2).mean()

        #* out  of distribution  energy 
        all_instance_label = details_targets.unique()
        spn_labels = all_instance_label[all_instance_label>= shapenet_object_point_label]
        
        
        for spn_label in spn_labels:   
            current_instance_energies =  energy[details_targets==spn_label]
            
            if current_instance_energies.size()[0] != 0 :
                #* calculate the dynamic m_out
                dmout = m_out +  (m_out_max - m_out) * ((spn_label - shapenet_object_point_label).float() / 100)
                #* calculate the energy loss for ood
                tmp += torch.pow(F.relu(dmout - current_instance_energies), 2).mean()

        loss += tmp*0.5
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy




""""

"""
def crude_dynamic_energy_loss(logits, targets,ood_ind=5,details_targets=None,m_out_max = 0,resized_point_label = 20, resize_m_out = -6 ):
    shapenet_object_point_label = resized_point_label+1

    # ood_ind = 5
    void_ind = 0
    T = 1.
    m_in = -12
    # Exclude class 0 from energy computation
    #!+====================================
    """"
    why not process the logits[:,ood_ind]
    """
    in_distribution_logits = torch.hstack([logits[:, 1:ood_ind],  logits[:, ood_ind+1:]])
    #!+====================================

    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0: #* not OOD sample in the label 
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        
        #* in distribution  energy  and the resized point energy (20)
        in_ebm_loss =  torch.pow(F.relu(Ec_in - m_in), 2).mean()  
        out_ebm_loss = torch.tensor(0.).cuda()
        cnt = 0
        if (details_targets == resized_point_label).sum() != 0 :
            out_ebm_loss += torch.pow(F.relu(resize_m_out - energy[details_targets == resized_point_label]), 2).mean()
            cnt += 1


        if (details_targets >= shapenet_object_point_label).sum() !=0:

            shapenet_point_energy = energy[details_targets >= shapenet_object_point_label]
            out_ebm_loss += torch.pow(F.relu(m_out_max - shapenet_point_energy), 2).mean()
            cnt += 1
            

        loss += (out_ebm_loss / (cnt + 1e-8) + in_ebm_loss)*0.5
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy