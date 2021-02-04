# FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation

This is an implementation for FoldingNet in PyTorch. FoldingNet is a autoencoder for point cloud. As for the details of the paper, please reference on [arXiv](https://arxiv.org/abs/1712.07262).

## Environment

* Ubuntu 18.04 LTS
* Python 3.8
* CUDA 10.1
* PyTorch 1.7.1

## Reconstruction

In order to train the model to do the reconstruction, use the command:

```bash
python train_ae.py --batch_size <batch_size> --epochs <epochs> --lr <lr> --weight_decay <weight_decay> --num_workers <num_workers>
```

## Examples

<img src="images/airplane_gt.png" width="350"/>
<img src="images/airplane_rc.png" width="350"/>

<img src="images/chair_gt.png" width="350"/>
<img src="images/chair_rc.png" width="350"/>

<img src="images/chair1_gt.png" width="350"/>
<img src="images/chair1_rc.png" width="350"/>

<img src="images/lamp_gt.png" width="350"/>
<img src="images/lamp_rc.png" width="350"/>

<img src="images/table_gt.png" width="350"/>
<img src="images/table_rc.png" width="350"/>
