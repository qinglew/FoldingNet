import argparse
import os
import time
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# torch.multiprocessing.set_start_method('spawn')

# Set working directory
# os.chdir("foldingnet_light")
print(f"Current WD: {os.getcwd()}")

# from model import args
from datasets import ShapeNetPartDataset, ModelNet40, ShapeNetCorePointsClouds_v2
from model import AutoEncoderLight, AutoEncoder
# from chamfer_distance.chamfer_distance import ChamferDistance

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="../ModelNet40_data/modelnet40_ply_hdf5_2048/")
parser.add_argument('--npoints', type=int, default=1024)
parser.add_argument('--mpoints', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=int, default=0.9)
parser.add_argument('--beta2', type=int, default=0.99)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--log_dir', type=str, default='./log')
args = parser.parse_args()

if "Shape" in args.root:
    dataset_name = "ShapeNetCore_v2"
elif "Model" in args.root:
    dataset_name = "ModelNet40"

ae = AutoEncoderLight(
    lr=args.lr,
    betas=[args.beta1, args.beta2],
    weight_decay=args.weight_decay,
    npoints=args.npoints,
    mpoints=args.mpoints,
    epochs=args.epochs,
    dataset_name=dataset_name
)

# prepare training and testing dataset
# train_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='train', classification=False, data_augmentation=True)
# test_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='test', classification=False, data_augmentation=True)

print("Loading training dataset ...")
if "Shape" in args.root:
    train_dataset = ShapeNetCorePointsClouds_v2(
        root=args.root,
        npoints=args.npoints,
        split="train",
        normalize=True,
        data_augmentation=True,
        sample_npoints=True
    )
if "Model" in args.root:
    train_dataset = ModelNet40(
        root=args.root, 
        npoints=args.npoints, 
        split='train', 
        normalize=True, 
        data_augmentation=True
    )

print("Loading validation dataset ...")
if "Shape" in args.root:
    val_dataset = ShapeNetCorePointsClouds_v2(
        root=args.root,
        npoints=args.npoints,
        split="val",
        normalize=True,
        data_augmentation=False,
        sample_npoints=False
    )
if "Model" in args.root:
    val_dataset = ModelNet40(
        root=args.root, 
        npoints=args.npoints, 
        split='test', 
        normalize=True, 
        data_augmentation=False
    )

print("Creating data loaders ...")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print("Number of training data: {}".format(len(train_dataset)))
print("Number of validation data: {}".format(len(val_dataset)))

# Set a callback to save best checkpoints
best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    filename="best_checkpoint",
)

last_checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_last=True,
    filename="last_checkpoint_at_{epoch:02d}",
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

#train model
trainer = pl.Trainer(accelerator="gpu", 
                     max_epochs=args.epochs, 
                     callbacks=[best_checkpoint_callback, 
                                last_checkpoint_callback,
                                lr_monitor])
trainer.fit(ae, train_loader, val_loader)