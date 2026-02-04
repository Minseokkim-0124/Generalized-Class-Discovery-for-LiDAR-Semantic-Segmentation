#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import sys
import numpy as np
import cupy as cp
import cuml.metrics as cm


class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

    self.unknown_labels = []
    self.unknown_scores = []

  def addBatch(self, x, y, z):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify
    z_row = z.reshape(-1)

    # check
    assert(x_row.shape == x_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    # np.add.at(self.conf_matrix, idxs, 1)
    self.unknown_labels.append(y_row)
    self.unknown_scores.append(z_row)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"

  def get_confusion(self):
    return self.conf_matrix.copy()

  def get_unknown_indices(self, device_id: int = None):
    """Evaluate AUPR, AUROC, FPR95(cpu).
    Visualize Curves when device_id is None(cpu).

    Args:
        device_id (int, optional): If None, use cpu. Else, use gpu of device_id. Defaults to None.

    Returns:
        tuple: AUPR, AUROC scores.
    """
    # breakpoint()
    self.unknown_labels = np.concatenate(self.unknown_labels)
    self.unknown_scores = np.concatenate(self.unknown_scores)
    # valid = self.unknown_labels != 0
    # self.unknown_labels = self.unknown_labels[valid]
    # self.unknown_scores = self.unknown_scores[valid]
    self.unknown_labels[self.unknown_labels != 5] = 0
    self.unknown_labels[self.unknown_labels == 5] = 1
    assert(len(self.unknown_scores) == len(self.unknown_labels))

    # scores_distribution_ood = self.unknown_scores[self.unknown_labels == 1]
    # scores_distribution_in = self.unknown_scores[self.unknown_labels != 1]
    # # import pdb;pdb.set_trace()
    # scores_distribution_ood.tofile('/home/user/ssd2tb/LiDAR_diffusion/UltraLiDAR/prediction/ood.score')
    # scores_distribution_in.tofile('/home/user/ssd2tb/LiDAR_diffusion/UltraLiDAR/prediction/in.score')
    # print('Save scores distribution successfully!')

    from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
    
    if device_id is None:  # Use CPU
      precision, recall, _ = precision_recall_curve(self.unknown_labels, self.unknown_scores)
      aupr_score = auc(recall, precision)
      print('AUPR is: ', aupr_score)
      
      fpr, tpr, _ = roc_curve(self.unknown_labels, self.unknown_scores)
      auroc_score_1 = auc(fpr, tpr)
      print('AUROC is: ', auroc_score_1)
      
      import matplotlib.pyplot as plt
      from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
      save_dir = './'

      display_PR = PrecisionRecallDisplay(precision=precision, recall=recall)
      display_PR.plot()
      plt.savefig(save_dir + "/PR_curve_final.png", dpi=300, bbox_inches="tight")
      plt.clf()

      display_Roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc_score_1)
      display_Roc.plot()
      plt.savefig(save_dir + "/Roc_curve_final.png", dpi=300, bbox_inches="tight")
      plt.clf()
      
      # TODO: Uncomment below and allow the computation of FPR95
      # print('FPR95 is: ', fpr[tpr > 0.95][0])
      
    else:  # Use GPU
      with cp.cuda.Device(device_id):
        precision, recall, _ = cm.precision_recall_curve(self.unknown_labels, self.unknown_scores)
        aupr_score = auc(recall, precision)
        print('AUPR is: ', aupr_score)
        # import pdb;pdb.set_trace()

        auroc_score_1 = cm.roc_auc_score(self.unknown_labels, self.unknown_scores)
        print('AUROC is: ', auroc_score_1)

        # auroc_score_2 = roc_auc_score(self.unknown_labels, self.unknown_scores)
        # import pdb;pdb.set_trace()

    return aupr_score, auroc_score_1


if __name__ == "__main__":
  # mock problem
  nclasses = 2
  ignore = []

  # test with 2 squares and a known IOU
  lbl = np.zeros((7, 7), dtype=np.int64)
  argmax = np.zeros((7, 7), dtype=np.int64)

  # put squares
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  # make evaluator
  eval = iouEval(nclasses, ignore)

  # run
  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  print("IoU: ", m_iou)
  print("IoU class: ", iou)
  m_acc = eval.getacc()
  print("Acc: ", m_acc)