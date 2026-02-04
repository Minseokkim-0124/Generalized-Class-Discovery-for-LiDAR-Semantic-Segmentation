import os
import torch
import re
from pytorch_lightning import Callback

class mIoUEvaluatorCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        interesting_metric_regex = re.compile(r'train/IoU/[\S]+_epoch')
        IoU_list = []
        callback_metrics = trainer.callback_metrics
        for key in callback_metrics.keys():
            mo = interesting_metric_regex.search(key)
            if mo is not None:
                IoU_list.append(callback_metrics[key])
        if IoU_list:
            mIoU = torch.mean(torch.stack(IoU_list))
            pl_module.log('train/mIoU', mIoU, rank_zero_only=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        interesting_metric_regex = re.compile(r'valid/IoU/[\S]+')
        interesting_metric_regex_novel = re.compile(r'valid/IoU_novel/[\S]+')
        interesting_metric_regex_base = re.compile(r'valid/IoU_base/[\S]+')
        
        IoU_list = []
        IoU_list_novel, IoU_list_base = [], []
        callback_metrics = trainer.callback_metrics
        for key in callback_metrics.keys():
            mo = interesting_metric_regex.search(key)
            mo_novel = interesting_metric_regex_novel.search(key)
            mo_base = interesting_metric_regex_base.search(key)
            if mo is not None:
                IoU_list.append(callback_metrics[key])
            elif mo_novel is not None:
                IoU_list_novel.append(callback_metrics[key])
            elif mo_base is not None:
                IoU_list_base.append(callback_metrics[key])
            
        
        if IoU_list:
            mIoU = torch.mean(torch.stack(IoU_list))
            pl_module.log('valid/mIoU', mIoU, rank_zero_only=True)
            if len(IoU_list_novel) != 0:
                mIoU_novel = torch.mean(torch.stack(IoU_list_novel))
                pl_module.log('valid/mIoU_novel', mIoU_novel, rank_zero_only=True)
            if len(IoU_list_base) != 0:
                mIoU_base = torch.mean(torch.stack(IoU_list_base))
                pl_module.log('valid/mIoU_base', mIoU_base, rank_zero_only=True)

    def on_test_epoch_end(self, trainer, pl_module):
        interesting_metric_regex = re.compile(r'test/IoU/[\S]+')
        IoU_list = []
        IoU_list_novel, IoU_list_base = [], []
        callback_metrics = trainer.callback_metrics
        for key in callback_metrics.keys():
            mo = interesting_metric_regex.search(key)
            if mo is not None:
                IoU_list.append(callback_metrics[key])
                if 'unknown_real_label' in key:
                    breakpoint()
                    IoU_list_novel.append(callback_metrics[key])
        
        if IoU_list:
            mIoU = torch.mean(torch.stack(IoU_list))
            pl_module.log('test/mIoU', mIoU, rank_zero_only=True)
            
class PretrainCheckpointCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_filename = (
            "-".join(
                [
                    "pretrain",
                    str(pl_module.hparams.split),
                    pl_module.hparams.dataset,
                    pl_module.hparams.comment,
                ]
            )
            + ".ckpt"
        )
        checkpoint_path = os.path.join(
            pl_module.hparams.checkpoint_dir, checkpoint_filename)
        torch.save(pl_module.model.state_dict(), checkpoint_path)