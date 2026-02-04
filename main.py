import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import modules

from utils import unkn_labels as unk_labels

SEED = 1234

parser = ArgumentParser()

parser.add_argument("-s", "--split", default=0, type=int, help="split", required=True)
parser.add_argument("--dataset", choices=["SemanticKITTI", "nuScenes"], default="SemanticKITTI", type=str, help="dataset")
parser.add_argument("--dataset_config", default=None, type=str, help="dataset config file")

parser.add_argument("--voxel_size", default="0.05", type=float, help="voxel_size")
parser.add_argument("--downsampling", default="80000", type=int, help="number of points per pcd")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--num_workers", default=8, type=int, help="number of workers")

# Directroy
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir for save")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--resume_checkpoint", type=str, help="checkpoint to evaluate", required=False)
parser.add_argument("--checkpoint", type=str, help="checkpoint to evaluate", required=False)

# hyperparameter
parser.add_argument("--train_lr", default=1.0e-2, type=float, help="learning rate for newly initialized parts of the pipeline")
parser.add_argument("--finetune_lr", default=1.0e-4, type=float, help="learning rate for already initialized parts of the pipeline")
parser.add_argument("--use_scheduler", default=False, action="store_true", help="use lr scheduler (linear warm-up + cosine_annealing)")
parser.add_argument("--warmup_epochs", default=4, type=int, help="warmup epochs")
parser.add_argument("--min_lr", default=1e-5, type=float, help="min learning rate")
parser.add_argument("--momentum_for_optim", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_for_optim", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--clear_cache_int", default=1, type=int, help="frequency of clear_cache")

# ORCA 
parser.add_argument("--use_uncertain_points_pairwise", default=False, action="store_true", help="use sampling for efficiency in pairwise objective")
parser.add_argument("--use_pairwise_sampling", default=False, action="store_true", help="use sampling for efficiency in pairwise objective")
parser.add_argument("--use_adaptive_uncertainty", default=False, action="store_true", help="use adpative uncertainty or not")
parser.add_argument("--uncertainty", default=0.5, type=float, help="fixed uncertainty")

parser.add_argument("--experiment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--epochs", type=int, default=50, help="training epochs")
parser.add_argument("--set_deterministic", default=True, action="store_true")
parser.add_argument("--visualize", default=False, action="store_true")
parser.add_argument("--test", default=False, action="store_true")
parser.add_argument("--full_sup", default=False, action="store_true")
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--module", type=str, help="model type to train", default='Oracle')

def main(args):
    if args.set_deterministic:
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if args.dataset_config is None:
        if args.dataset == "SemanticKITTI":
            args.dataset_config = "config/semkitti_dataset.yaml"
        elif args.dataset == "nuScenes":
            args.dataset_config = "/config/nuscenes_dataset.yaml"
        else:
            raise NameError(f"Dataset {args.dataset} not implemented")

    with open(args.dataset_config, "r") as f:
        dataset_config = yaml.safe_load(f)

    # SemanticKITTI split
    if args.dataset == "SemanticKITTI":
        if args.split == 1:
            unknown_labels = [4,6] # other-vehicle, bicyclist, parking (4,6)
            args.data_idx_file = './split_npy/kitti/selected_idx_in__ex_4_6_50%.npy'
        
        elif args.split == 2:
            unknown_labels = [2,4,5] # motorcycle, other-vehicle, person
            args.data_idx_file = './split_npy/kitti/selected_idx_in__ex_2_4_5_45%.npy'
        
        elif args.split == 3: 
            unknown_labels = [3,6,9] # parking truck bicyclist
            args.data_idx_file = './split_npy/kitti/selected_idx_in__ex_3_6_9_40%.npy'

    # nuScenes split 
    elif args.dataset == "nuScenes":
        if args.split == 1:
            unknown_labels = [1,8] # bicycle trailer
            args.data_idx_file = './split_npy/nusc/nusc_selected_idx_in__ex_1_8_45%.npy'
        
        
        elif args.split == 2:
            unknown_labels = [1,2,5] # bicycle bus motorcycle
            args.data_idx_file = './split_npy/nusc/nusc_selected_idx_in__ex_1_2_5_35%.npy'
        
        elif args.split == 3:
            unknown_labels = [0,4,7,8]
            args.data_idx_file = './split_npy/nusc/nusc_selected_idx_in__ex_0_4_7_8_25%.npy'
            
        
    
    print(f"Unknown labels in split {args.split}")
    for labels_ in unknown_labels:
        print(f"{str(labels_).ljust(2)}: {dataset_config['labels'][dataset_config['learning_map_inv'][labels_]]}")

    label_mapping, label_mapping_inv, unknown_label = unk_labels.label_mapping(
        unknown_labels, dataset_config["learning_map_inv"].keys()
    )

    # num_unlabeled = number of unknown classes  num_labeled_classes = number of known classes
    number_of_unk = len(unknown_labels)
    args.num_classes = len(label_mapping)
    args.num_unlabeled_classes = number_of_unk
    # args.num_unlabeled_classes =  
    args.num_labeled_classes = args.num_classes - args.num_unlabeled_classes

    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1, # all models are saved
        save_weights_only=False,
        dirpath=args.checkpoint_dir,
        every_n_epochs=True,
    )

    # Tensorboard Logger
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.experiment)
    
    # Test Mode
    if args.test:
            module = getattr(modules, args.module).load_from_checkpoint(args.checkpoint,
                                                                        strict=False,
                                                                        label_mapping=label_mapping,
                                                                        label_mapping_inv=label_mapping_inv,
                                                                        unknown_label=unknown_label,
                                                                        dataset_config=args.dataset_config,
                                                                        batch_size = args.batch_size,
                                                                        num_workers= args.num_workers,
                                                                        debug=args.debug,
                                                                        test=args.test,
                                                                        name=args.experiment,
                                                                        log_dir=args.log_dir
                                                                        )
            assert module.hparams.split == args.split, f"Given split {args.split} is different from the split {module.hparams.split} in checkpoint"
                
            trainer = pl.Trainer(
                max_epochs=args.epochs,
                logger=tb_logger,
                gpus=-1,
                num_sanity_val_steps=0,
                callbacks= [checkpoint_callback],
            )
            trainer.validate(module)

    # Train Mode
    else:
        module = getattr(modules, args.module)(label_mapping, label_mapping_inv, unknown_label, **args.__dict__)
        if 'Merge' in args.module:
            if args.resume_checkpoint is not None:
                trainer = pl.Trainer(
                    resume_from_checkpoint=args.resume_checkpoint,
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    # val_check_interval=500,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)
                
            else:
                trainer = pl.Trainer(
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    # val_check_interval=500,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)

        # NOTE: FineTuning 용 train code
        elif 'FineTuning' in args.module or 'Discover' in args.module:
            checkpoint_callback = ModelCheckpoint(
                save_top_k=-1, # all models are saved
                save_weights_only=False,
                dirpath=args.checkpoint_dir,
                every_n_train_steps=500,
            )
            
            # NOTE: 이어서 학습
            if args.resume_checkpoint is not None:
                trainer = pl.Trainer(
                    resume_from_checkpoint=args.resume_checkpoint,
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    val_check_interval=500,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)
                
            else:
                trainer = pl.Trainer(
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    val_check_interval=500,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)
        
        elif 'Extra' in args.module:
            checkpoint_callback = ModelCheckpoint(
                save_top_k=-1, # all models are saved
                save_weights_only=False,
                dirpath=args.checkpoint_dir,
                every_n_train_steps=1000,
            )
            

            # NOTE: 이어서 학습
            if args.resume_checkpoint is not None:
                trainer = pl.Trainer(
                    resume_from_checkpoint=args.resume_checkpoint,
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    val_check_interval=1000,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)
                
            else:
                trainer = pl.Trainer(
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    val_check_interval=1000,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)
        
        elif 'UncertaintyCheck' in args.module:
            trainer = pl.Trainer(
                    max_epochs=1,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    callbacks= []
                )
            trainer.fit(module)
        # NOTE: 일반 module train code
        else:
            # NOTE: 이어서 학습
            if args.resume_checkpoint is not None:
                trainer = pl.Trainer(
                    resume_from_checkpoint=args.resume_checkpoint,
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)
                
            else:
                trainer = pl.Trainer(
                    max_epochs=args.epochs,
                    logger=tb_logger,
                    gpus=-1,
                    num_sanity_val_steps=0,
                    callbacks= [checkpoint_callback],
                )
                trainer.fit(module)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)