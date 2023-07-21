import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import (
    Optional, List, Union, Iterable, Literal, Tuple
)
from utils import index_points, knn
from chamfer_distance.chamfer_distance import ChamferDistance
from torch import optim


"""
In this script we define all the components/layers of the FoldingNet model.

Legend -> tensors dimensions:
    - B = batch size (note: each item of a batch correspond to a pointcloud)./
    - C = number of channels (e.g., the input of the model are arrays of 3D coordinates, 
          i.e., matrices of size Nx3. Hence, in this case, the number of channels C is 3).
    - N = number of points in the input pointcloud.
    - k = it refers to the number of items in a neighborhood.
    - M = number of points in the folded grid.
"""


#--------------------------------------------------------------------------------
class GraphLayer(pl.LightningModule):
    """
    Graph layer that performs KNN maxpoling.
    """

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            k: Optional[int] = 16
        ):
        '''
        Parameters:
        -----------
            in_channels: (int)
                The number of input channels to the mapping layer after KNN maxpooling.

            out_channels: (int)
                The number of output channels to the mapping layer after KNN maxpooling.

            k: (Optional[int], deafult = 16)
                The number of neighbors considered to construct the KNN graph.
        '''

        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Parameters
        ----------
            x: (torch.Tensor)
                Tensor of size (B, C_in, N)

        Returns:
        --------
            x: (torch.Tensor)
                Tensor of size (B, C_out, N)
        """
        
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling (performed separately on each dimension)
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, N, C)
        
        # Feature Map
        x = F.relu(self.bn(self.conv(x)))

        return x
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
class Encoder(pl.LightningModule):
    """
    Graph based encoder.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channels=64, out_channels=128, k=16)
        self.graph_layer2 = GraphLayer(in_channels=128, out_channels=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        # self.conv5 = nn.Conv1d(512, 512, 1)
        # self.bn5 = nn.BatchNorm1d(512)

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

        # get codeword
        x = self.bn4(self.conv4(x))
        x = torch.max(x, dim=-1)[0]

        ''' 
        Note: this last part of the model to get the codeword differs a bit 
        from the one reported in the paper.
        Indeed, the max pooling should instead preceed the linear/conv layer,
        and it should be followed instead by 2 layers.

        # max pooling
        x = torch.max(x, dim=-1)[0]

        # two layer MLP
        x = self.bn4(self.conv4(x))
        x = self.bn5(self.conv5(x))
        '''

        return x
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
class FoldingLayer(pl.LightningModule):
    """
    The folding operation of FoldingNet

    """
    def __init__(
            self, 
            in_channels: int,
            out_channels_lst: List[int]
        ):
        '''
        Parameters:
        -----------
            in_channels: (int)
                The number of input channels 
                (e.g., 514 if input is codewords + 2D grid)
            out_channels_lst: (List[int])
                The numbers of output channels from each layer of the MLP
        '''
        super().__init__()

        layers = []
        for out_channels in out_channels_lst[:-1]:
            conv = nn.Conv1d(in_channels, out_channels, 1)
            bn = nn.BatchNorm1d(out_channels)
            activ = nn.ReLU(inplace=True)
            layers.extend([conv, bn, activ])
            in_channels = out_channels
        out_layer = nn.Conv1d(in_channels, out_channels_lst[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(
            self, 
            grids: torch.Tensor, 
            codewords: torch.Tensor
        ):
        """
        Parameters
        ----------
            grids: (torch.Tensor) 
                Reshaped 2D grids (B, 2, M) or intermediate reconstructed point clouds (B, 3, M)
            codewords: (torch.Tensor)
                Replicated codewords computed by the Encoder (B, 512, M)
        
        Returns:
            x: (torch.Tensor)
                A tensor of size (B, 3, M), that is either the intermediate or final
                reconstructed point cloud.
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x
#-------------------------------------------------------------------------------



#-------------------------------------------------------------------------------
class Decoder(pl.LightningModule):
    """
    Decoder module of FoldingNet
    
    """
    def __init__(
            self, 
            in_channels: Optional[int] = 512,
            grid_size: Optional[int] = 2025
        ):
        '''
        Parameters:
        -----------
            in_channels: (Optional[int], default = 512)
                The number of input channels in the Decoder, that is
                the length of the codeword produced by the Encoder.

            grid_size: (Optional[int], default = 2025)
                The number of points in the grid to be folded (i.e., M in the model legend)
        '''
        super().__init__()

        # Sample the grids in 2D space
        xx = np.linspace(-0.3, 0.3, int(np.sqrt(grid_size)), dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, int(np.sqrt(grid_size)), dtype=np.float32)
        grid = np.meshgrid(xx, yy)   # (2, sqrt(M), sqrt(M))

        # Reshape grid
        self.grid = torch.Tensor(np.asarray(grid)).view(2, -1)  # (2, M)
        self.m = self.grid.shape[1]

        # Folding layers
        self.fold1 = FoldingLayer(in_channels + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(in_channels + 3, [512, 512, 3])

    def forward(self, x):
        """
        Parameters:
        -----------
            x: (torch.Tensor)
                A tensor of size (B, C) (e.g., the codeword, of size (B, 512))
        """

        batch_size = x.shape[0]

        # replicate grid for batch operation
        grid = self.grid.to(x.device)                      # (2, M)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, M)
        
        # replicate codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, M)
        
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        
        return recon2
#-------------------------------------------------------------------------------



# #-------------------------------------------------------------------------------
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
# #-------------------------------------------------------------------------------



#---------------------------------------------------------------------------------
class AutoEncoder(pl.LightningModule):
    '''
    The end-to-end Autoencoder model FoldingNet
    '''
    def __init__(
            self, 
            dataset_name: Literal['ModelNet40', 'ShapeNetCore'],
            lr: Optional[float] = 1e-4, 
            betas: Optional[List[float]] = [0.9, 0.99], 
            weight_decay: Optional[float] = 1e-6, 
            npoints: Optional[int] = 2048, 
            mpoints: Optional[int] = 2025, 
            epochs: Optional[int] = 1000, 
        ):
        '''
        Parameters:
        -----------
        dataset_name:
        dataset_name: (Literal['ModelNet40', 'ShapeNetCore'])
            lr: (Optional[float] = 1e-4)
            betas: (Optional[List[float]] = [0.9, 0.99])
            weight_decay: (Optional[float] = 1e-6)
            npoints: (Optional[int] = 2048)
                The number of points in the input point cloud.
            mpoints: (Optional[int] = 2025) 
                The number of points in the output point cloud.
            epochs: (Optional[int] = 1000) 
        '''
        super().__init__()

        self.save_hyperparameters()
        self.cd_loss = ChamferDistance()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.pc_to_plot = None
        # self.pred_pc_to_plot = None
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.npoints = npoints, 
        self.mpoints = mpoints
        self.epochs = epochs,
        self.dataset = dataset_name


    def forward(
            self, 
            input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ):
        '''
        Parameters:
        -----------
            input: Union[torch.Tensor, Tuple(torch.Tensor, torch.Tensor)]
                Tensor of size (B, 3, N)

        Returns:
        --------
            x: (torch.Tensor)
                Tensor of size (B, 3, M)
        '''

        # extract input (x point cloud, y label)
        x, y = input
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

    def training_step(
            self, 
            train_batch: Union[Iterable[torch.Tensor], torch.Tensor], 
            batch_idx: int
        ):
        '''
        Parameters:
        -----------
            train_batch: (Union[Iterable[torch.Tensor], torch.Tensor])
                Tensor of size (B, N, 3), or iterable storing the tensor
                and additional items (e.g., a label associated to the point cloud)
        '''

        # extract input (x point cloud, y label)
        x, y = train_batch 
        
        # input
        x = x.permute(0, 2, 1) # (B, 3, N)
        
        # codeword
        z = self.encoder(x) # (B, 512)
        
        # reconstructed
        x_hat = self.decoder(z) # (B, 3, M)
        
        loss = self.cd_loss(x.permute(0, 2, 1), x_hat.permute(0, 2, 1))
        
        self.log(
            'train_loss', loss, on_step=True, 
            on_epoch=True, prog_bar=True, batch_size=x.size()[0]
        )
        
        return loss
    

    def validation_step(
            self, 
            val_batch: Union[Iterable[torch.Tensor], torch.Tensor], 
            batch_idx: int
        ):
        '''
        Parameters:
        -----------
            val_batch: (Union[Iterable[torch.Tensor], torch.Tensor])
                Tensor of size (B, N, 3), or iterable storing the tensor
                and additional items (e.g., a label associated to the point cloud)
        '''

        # extract input (x point cloud, y label)
        x, y = val_batch 
        
        # input 
        x = x.permute(0, 2, 1)

        # codeword
        z = self.encoder(x) # (B, 512)
        
        # reconstructed
        x_hat = self.decoder(z) # (B, 3, M)
        
        loss = self.cd_loss(x.permute(0, 2, 1), x_hat.permute(0, 2, 1))

        self.log(
            'val_loss', loss, on_step=False,
            on_epoch=True, prog_bar=True, batch_size=x.size()[0]
        )
        
        # # Save the first point cloud in the batch to plot it at the end of the epoch
        # if batch_idx == 0:
        #     self.pc_to_plot = x.permute(0, 2, 1)[0].to("cpu")
        #     self.pred_pc_to_plot = x_hat.permute(0, 2, 1)[0].to("cpu")

        return loss
    
    def test_step(
            self, 
            test_batch: Union[Iterable[torch.Tensor], torch.Tensor], 
            batch_idx: int
        ):
        '''
        Parameters:
        -----------
            test_batch: (Union[Iterable[torch.Tensor], torch.Tensor])
                Tensor of size (B, N, 3), or iterable storing the tensor
                and additional items (e.g., a label associated to the point cloud)
        '''

        # extract input (x point cloud, y label)
        x, y = test_batch 

        # input 
        x = x.permute(0, 2, 1)

        # codeword
        z = self.encoder(x) # (B, 512)
        
        # reconstructed
        x_hat = self.decoder(z) # (B, 3, M)
        
        return self.cd_loss(x.permute(0, 2, 1), x_hat.permute(0, 2, 1))
    

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), 
                         lr=self.lr, 
                         betas=self.betas, 
                         weight_decay=self.weight_decay)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, 
                                                                  mode='min',
                                                                  factor=0.5,
                                                                  patience=20,
                                                                  min_lr=1e-7),
                "monitor": "val_loss",
                "frequency": 1
            },
        }
#-------------------------------------------------------------------------------


if __name__ == '__main__':
    pcs = torch.randn(32, 3, 2048)

    encoder = Encoder()
    codewords = encoder(pcs)
    print(codewords.size())

    decoder = Decoder(codewords.size(1))
    recons = decoder(codewords)
    print(recons.size())

    ae = AutoEncoder()
    y = ae(pcs)
    print(y.size())