import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import index_points, knn


class GraphLayer(nn.Module):
    """
    Graph layer.

    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    """
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        """
        Parameters
        ----------
            x: tensor with size of (B, C, N)
        """
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, N, C)
        
        # Feature Map
        x = F.relu(self.bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    """
    Graph based encoder.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

    def forward(self, x):
        b, c, n = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))


        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))

        x = torch.max(x, dim=-1)[0]
        return x


class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_channel):
        super(Decoder, self).__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)

        # reshape
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        
        self.m = self.grid.shape[1]

        self.fold1 = FoldingLayer(in_channel + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channel + 3, [512, 512, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)
        
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, 45 * 45)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        
        return recon2


if __name__ == '__main__':
    pcs = torch.randn(32, 3, 2048)

    encoder = Encoder()
    codewords = encoder(pcs)
    print(codewords.size())

    decoder = Decoder(codewords.size(1))
    recons = decoder(codewords)
    print(recons.size())
