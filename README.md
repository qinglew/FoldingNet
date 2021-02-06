# FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation

This is an implementation for FoldingNet in PyTorch. FoldingNet is a autoencoder for point cloud. As for the details of the paper, please reference on [arXiv](https://arxiv.org/abs/1712.07262).

## Environment

* Ubuntu 18.04 LTS
* Python 3.8.5
* CUDA 10.1.243
* PyTorch 1.7.0

## Reconstruction on ShapeNet

In order to train the model to do the reconstruction, use the command:

```bash
python train_ae.py --batch_size <batch_size> --epochs <epochs> --lr <lr> --weight_decay <weight_decay> --num_workers <num_workers>
```

In order to evaluate the model, see the `evaluation_ae.py`

## Transfer Classification on ModelNet40

I train the AutoEncoder on ShapeNet and use the encoder to extract the features of point clouds of training set of ModelNet40. I train a SVM on the features extracted from ModelNet40's training dataset and evaluate the svm on the testing dataset of ModelNet40.

Accuracy Overall | 79.82%
-- | --
Precision | 92.42%
Recall | 81.08%
F1-Score | 86.38%

category | precision | recall | f1-score | support
-- | -- | -- | -- | --
0 | 1.0000 | 1.0000 | 1.0000 | 100
1 | 0.9762 | 0.8200 | 0.8913 | 50
2 | 0.9400 | 0.9400 | 0.9400 | 100
3 | 0.6500 | 0.6500 | 0.6500 | 20
4 | 0.9300 | 0.9300 | 0.9300 | 100
5 | 0.9574 | 0.9000 | 0.9278 | 100
6 | 0.8333 | 1.0000 | 0.9091 | 20
7 | 0.9896 | 0.9500 | 0.9694 | 100
8 | 0.9896 | 0.9500 | 0.9694 | 100
9 | 0.9444 | 0.8500 | 0.8947 | 20
10 | 0.7500 | 0.4500 | 0.5625 | 20
11 | 0.7778 | 0.7000 | 0.7368 | 20
12 | 0.7586 | 0.7674 | 0.7630 | 86
13 | 0.8261 | 0.9500 | 0.8837 | 20
14 | 0.8000 | 0.7442 | 0.7711 | 86
15 | 0.0000 | 0.0000 | 0.0000 | 20
16 | 0.9125 | 0.7300 | 0.8111 | 100
17 | 1.0000 | 0.9500 | 0.9744 | 100
18 | 0.9500 | 0.9500 | 0.9500 | 20
19 | 0.8750 | 0.7000 | 0.7778 | 20
20 | 1.0000 | 1.0000 | 1.0000 | 20
21 | 0.9787 | 0.9200 | 0.9485 | 100
22 | 0.9794 | 0.9500 | 0.9645 | 100
23 | 0.7812 | 0.5814 | 0.6667 | 86
24 | 0.9286 | 0.6500 | 0.7647 | 20
25 | 1.0000 | 0.7200 | 0.8372 | 100
26 | 0.9153 | 0.5400 | 0.6792 | 100
27 | 1.0000 | 0.2000 | 0.3333 | 20
28 | 0.9663 | 0.8600 | 0.9101 | 100
29 | 0.9167 | 0.5500 | 0.6875 | 20
30 | 0.9796 | 0.9600 | 0.9697 | 100
31 | 0.8462 | 0.5500 | 0.6667 | 20
32 | 0.7692 | 0.5000 | 0.6061 | 20
33 | 0.8913 | 0.8200 | 0.8542 | 100
34 | 0.8000 | 0.8000 | 0.8000 | 20
35 | 1.0000 | 0.9400 | 0.9691 | 100
36 | 0.9367 | 0.7400 | 0.8268 | 100
37 | 0.8452 | 0.7100 | 0.7717 | 100
38 | 1.0000 | 0.5000 | 0.6667 | 20
39 | 0.7500 | 0.4500 | 0.5625 | 20
micro avg | 0.9242 | 0.8108 | 0.8638 | 2468 
macro avg | 0.8786 | 0.7468 | 0.7949 | 2468
weighted avg | 0.9172 | 0.8108 | 0.8542 | 2468
samples avg | 0.8002 | 0.8108 | 0.8037 | 2468

## Examples

![Examples](misc/examples.png)
