import numpy as np
from sklearn.cluster import MiniBatchKMeans
import copy
import random
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import check_random_state
import torch

class SemiSupervisedStreamKM:
    def __init__(self, num_clusters, coreset_size=1000, batch_size=100):
        self.num_clusters = num_clusters
        self.coreset_size = coreset_size
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size, max_iter=10)
        self.coreset = []
        self.labeled_data = []  

    def add_labeled_batch(self, labeled_feats, labeled_labels):
        for feat, label in zip(labeled_feats, labeled_labels):
            self.labeled_data.append((feat, label))

        # # FIFO 방식으로 오래된 labeled 데이터 제거 (Optional)
        # if len(self.labeled_data) > self.coreset_size:
        #     self.labeled_data = self.labeled_data[-self.coreset_size:]

    def add_to_coreset(self, center):
        self.coreset.append(center)
        # (Optional)
        # if len(self.coreset) > self.coreset_size:
        #     self.coreset.pop(0)  # 가장 오래된 클러스터 중심 제거

    def partial_fit(self, new_data, labeled=True):
        batch = []
        for x in new_data:
            if len(self.coreset) > 0:
                distances = [np.linalg.norm(x - c) for c in self.coreset]
                closest_cluster = np.argmin(distances)
                # 가장 가까운 클러스터 중심을 배치에 추가
                batch.append(self.coreset[closest_cluster])
            else:
                batch.append(x)

        # MiniBatchKMeans 업데이트
        if len(batch) > 0:
            self.kmeans.partial_fit(batch)
            for center in self.kmeans.cluster_centers_:
                self.add_to_coreset(center)

    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_




def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis


class OnlineSemiKMeans:

    def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
                 n_init=10, num_labeled_classes=None, num_unlabeled_classes = None, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode
        self.num_lab_cls = num_labeled_classes
        self.num_unlab_cls = num_unlabeled_classes
        self.center_queue = None

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):

        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * len(l_targets))
        val_idxs = np.random.choice(range(len(l_targets)), size=(num_val_instances), replace=False)
        val_idxs.sort()
        remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets


    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)

        if pre_centers is not None:

            C = pre_centers

        else:

            C = X[random_state.randint(0, len(X))]

        C = C.view(-1, X.shape[1])

        while C.shape[0] < k:

            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2/d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)

        return C


    def fit_once(self, X, random_state):

        centers = torch.zeros(self.k, X.shape[1]).type_as(X)
        labels = -torch.ones(len(X))
        #initialize the centers, the first 'k' elements in the dataset will be our initial centers

        if self.init == 'k-means++':
            centers = self.kpp(X, k=self.k, random_state=random_state)

        elif self.init == 'random':
            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]

        #begin iterations

        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):

            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            mindist, labels = torch.min(dist, dim=1)
            inertia = mindist.sum()

            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):

        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        # 이전에는 거의 모든 클래스들을 다 가지고 있지만 우리의 경우 그러지는 못함 그래서 dictionary 형태로 바꿔야 할듯?
        # support_idxs = list(map(supp_idxs, l_classes))
        support_idxs = {c.item() : supp_idxs(c) for c in l_classes}
        # 딕셔너리로 l_centers 구하고, 모든 데이터 셋을 다 돌지 않기 떄문에 있는 라벨에 해당하는 것들만 center 계산하고
        # 없는 클래스에 대해서는 0 값
        # l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])

        l_centers = torch.randn((self.num_lab_cls, l_feats.shape[1]), dtype=l_feats.dtype, device=l_feats.device) * 0.1
        for class_label, idx_list in support_idxs.items():
            if len(idx_list) > 0:  # 해당 클래스의 샘플이 존재할 경우
                l_centers[class_label] = l_feats[idx_list].mean(0)

        cat_feats = torch.cat((l_feats, u_feats))

        # centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        # centers[:len(l_classes)] = l_centers
        centers = torch.randn([self.k, cat_feats.shape[1]], dtype=cat_feats.dtype, device=cat_feats.device) * 0.1

        # 라벨이 있는 데이터의 클러스터 중심을 올바르게 배치
        for i, class_label in enumerate(l_classes):
            centers[class_label.item()] = l_centers[i]
        

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()
        
        # NOTE: Remapping이 필요할까? 필요하긴 한데, 기존 코드랑은 방향이 조금 다름..
        # cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        # for i in range(l_num):
        #     labels[i] = cid2ncid[l_targets[i]]
        
        # labels[:len(l_targets)]  = l_targets
        labels[:len(l_targets)] = torch.tensor(l_targets, dtype=torch.long, device=labels.device)

        #initialize the centers, the first 'k' elements in the dataset will be our initial centers
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for it in range(self.max_iterations):
            centers_old = centers.clone()
            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels
            
            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                if selected.numel() > 0:  
                    selected = torch.index_select(cat_feats, 0, selected)
                    centers[idx] = selected.mean(dim=0)
                # else:
                #     print(f"Warning: No samples assigned to cluster {idx}, keeping previous center")

            # for idx in range(self.k):
            #     selected = torch.nonzero(labels == idx).squeeze()
            #     selected = torch.index_select(cat_feats, 0, selected)
            #     centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))

            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit(self, X):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


    def fit_mix(self, u_feats, l_feats, l_targets, cluster_center=None, center_only=False):

        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):

                labels, inertia, centers, n_iters = fit_func(u_feats, l_feats, l_targets, random_state)

                if best_inertia is None or inertia < best_inertia:
                    if center_only:
                        self.cluster_centers_ = centers.clone()
                    else:
                        self.labels_ = labels.clone()
                        self.cluster_centers_ = centers.clone()
                        best_inertia = inertia
                        self.inertia_ = inertia
                        self.n_iter_ = n_iters

        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(fit_func)(u_feats, l_feats, l_targets, seed)
                                                              for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            
            if center_only:
                self.cluster_centers_ = centers[best]
            else:
                self.labels_ = labels[best]
                self.inertia_ = inertia[best]
                self.cluster_centers_ = centers[best]
                self.n_iter_ = n_iters[best]


# def main():

#     import matplotlib.pyplot as plt
#     from matplotlib import style
#     import pandas as pd
#     style.use('ggplot')
#     from sklearn.datasets import make_blobs
#     from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
#     X, y = make_blobs(n_samples=500,
#                       n_features=2,
#                       centers=4,
#                       cluster_std=1,
#                       center_box=(-10.0, 10.0),
#                       shuffle=True,
#                       random_state=1)  # For reproducibility

#     cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if cuda else "cpu")
#     #  X = torch.from_numpy(X).float().to(device)


#     y = np.array(y)
#     l_targets = y[y>1]
#     l_feats = X[y>1]
#     u_feats = X[y<2]
#     cat_feats = np.concatenate((l_feats, u_feats))
#     y = np.concatenate((y[y>1], y[y<2]))
#     cat_feats = torch.from_numpy(cat_feats).to(device)
#     u_feats = torch.from_numpy(u_feats).to(device)
#     l_feats = torch.from_numpy(l_feats).to(device)
#     l_targets = torch.from_numpy(l_targets).to(device)

#     km = K_Means(k=4, init='k-means++', random_state=1, n_jobs=None, pairwise_batch_size=10)

#     #  km.fit(X)

#     km.fit_mix(u_feats, l_feats, l_targets)
#     #  X = X.cpu()
#     X = cat_feats.cpu()
#     centers = km.cluster_centers_.cpu()
#     pred = km.labels_.cpu()
#     print('nmi', nmi_score(pred, y))

#     # Plotting starts here
#     colors = 10*["g", "c", "b", "k", "r", "m"]

#     for i in range(len(X)):
#         x = X[i]
#         plt.scatter(x[0], x[1], color = colors[pred[i]],s = 10)

#     for i in range(4):
#         plt.scatter(centers[i][0], centers[i][1], s = 130, marker = "*", color='r')
#     plt.show()

# if __name__ == "__main__":
#     main()