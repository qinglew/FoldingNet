import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import (
    ModelNet40_train,
    ModelNet40_test, 
    ShapeNetCore_train,
    ShapeNetCore_test
)
from model import AutoEncoder
from train_utils import get_args

# Read config file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
cmd_line_args = parser.parse_args()
args = get_args(cmd_line_args.config)

if "Shape" in args.root:
    dataset_name = "ShapeNetCore"
elif "Model" in args.root:
    dataset_name = "ModelNet40"


# Create the model
ae = AutoEncoder(
    dataset_name=dataset_name,
    lr=args.lr,
    betas=[args.beta1, args.beta2],
    weight_decay=args.weight_decay,
    npoints=args.npoints,
    mpoints=args.mpoints,
    epochs=args.epochs,
)


print("Loading training dataset ...")
if "Shape" in args.root:
    train_dataset = ShapeNetCore_train(
        root=args.root,
        npoints=args.npoints,
        normalize=True,
        data_augmentation=True
    )
if "Model" in args.root:
    train_dataset = ModelNet40_train(
        root=args.root, 
        npoints=args.npoints,  
        normalize=True, 
        data_augmentation=True
    )

print("Loading validation dataset ...")
if "Shape" in args.root:
    val_dataset = ShapeNetCore_test(
        root=args.root,
        npoints=args.npoints,
        split="val",
        normalize=True
    )
if "Model" in args.root:
    val_dataset = ModelNet40_test(
        root=args.root, 
        npoints=args.npoints, 
        split='test', 
        normalize=True
    )

print("Creating data loaders ...")
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers
)

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