import torch
import torch.nn as nn

from utils import setup_seed


class ChamferDistance(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        """
        Args:
            x: tensor with size of (B, N, 3)
            y: tensor with size of (B, M, 3)
        """
        xy = -2 * torch.matmul(x, y.permute(0, 2, 1))  # (B, N, M)
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)  # (B, N, 1)
        yy = torch.sum(y ** 2, dim=-1, keepdim=True).permute(0, 2, 1)  # (B, 1, M)
        dist = torch.sqrt(xy + xx + yy + 1e-10)  # (B, N, M)
        distance1 = torch.min(dist, dim=2)[0]  # (B, N)
        distance2 = torch.min(dist, dim=1)[0]  # (B, M)
        mean_dist1 = distance1.mean(dim=-1, keepdim=True)  # (B, 1)
        mean_dist2 = distance2.mean(dim=-1, keepdim=True)  # (B, 1)
        return torch.max(torch.cat([mean_dist1, mean_dist2], dim=1), dim=1)[0].sum()


if __name__ == '__main__':
    setup_seed(21)

    cd_loss = ChamferDistance()
    pcs1 = torch.rand(10, 2048, 3)
    pcs2 = torch.rand(10, 2048, 3)
    print(cd_loss(pcs1, pcs2))
