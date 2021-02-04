import random
import torch
from datasets import ShapeNetPartDataset
from model import AutoEncoder
from chamfer_distance.chamfer_distance import ChamferDistance
from utils import show_point_cloud


ae = AutoEncoder()
ae.load_state_dict(torch.load('log/model_lowest_cd_loss.pth'))
ae.eval()

DATASET_PATH = '/home/rico/Workspace/Dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0'
test_dataset = ShapeNetPartDataset(root=DATASET_PATH, npoints=2048, split='train', classification=False, data_augmentation=False, class_choice='Table')
input_pc = test_dataset[random.randint(0, len(test_dataset))][0]
show_point_cloud(input_pc)

input_tensor = input_pc.unsqueeze(0).permute(0, 2, 1)
output_tensor = ae(input_tensor)
reconstructed_pc = output_tensor.permute(0, 2, 1).squeeze().detach().numpy()

show_point_cloud(reconstructed_pc)


cd_loss = ChamferDistance()
print(cd_loss(input_tensor, output_tensor))
