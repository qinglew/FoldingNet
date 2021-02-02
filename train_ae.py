import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import ShapeNetPartDataset
from model import AutoEncoder
from loss import ChamferDistance


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
parser.add_argument('--npoints', type=int, default=2048)
parser.add_argument('--mpoints', type=int, default=2025)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()


# prepare training and testing dataset
train_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='train', classification=False, data_augmentation=True)
test_dataset = ShapeNetPartDataset(root=args.root, npoints=args.npoints, split='test', classification=False, data_augmentation=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
autoendocer = AutoEncoder()
autoendocer.to(device)

# loss function
cd_loss = ChamferDistance()
# optimizer
optimizer = optim.Adam(autoendocer.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)

batches = int(len(train_dataloader) / args.batch_size + 0.5)

min_cd_loss = 1e3

# begin training
for epoch in range(1, args.epochs + 1):
    # training
    autoendocer.train()
    for i, data in enumerate(train_dataloader):
        point_clouds, _ = data
        point_clouds.to(device)

        recons = autoendocer(point_clouds)
        ls = cd_loss(point_clouds, recons)
        
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch + 1, args.epochs, i + 1, batches, ls.item() / len(point_clouds)))
    
    # evaluation
    autoendocer.eval()
    total_cd_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            point_clouds, _ = data
            point_clouds.to(device)
            recons = autoendocer(point_clouds)
            ls = cd_loss(point_clouds, recons)
            total_cd_loss += ls.item()
    
    mean_cd_loss = total_cd_loss / len(test_dataset)
    if mean_cd_loss < min_cd_loss:
        min_cd_loss = mean_cd_loss

    print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {}.\033[0m'.format(epoch + 1, args.epochs, mean_cd_loss, min_cd_loss))
