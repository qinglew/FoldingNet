import torch

from datasets import ShapeNetPartDataset
from model import AutoEncoder
from chamfer_distance.chamfer_distance import ChamferDistance


test_dataset = ShapeNetPartDataset(root='/home/rico/Workspace/Dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                                   npoints=2048, split='test', classification=False, data_augmentation=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
model = AutoEncoder()
model.load_state_dict(torch.load('log/model_lowest_cd_loss.pth'))
device = torch.device('cuda')
model.to(device)

cd_loss = ChamferDistance()

# evaluation
model.eval()
total_cd_loss = 0
with torch.no_grad():
    for data in test_dataloader:
        point_clouds, _ = data
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = model(point_clouds)
        ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        total_cd_loss += ls.item()

# calculate the mean cd loss
mean_cd_loss = total_cd_loss / len(test_dataset)
print('Mean Chamfer Distance of all Point Clouds:', mean_cd_loss)
