import json
import os
import random

import h5py
import numpy as np
import torch
import torch.utils.data as data


class ShapeNetPartDataset(data.Dataset):
    def __init__(self, root, split='train', npoints=2500, classification=False, class_choice=None, data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.cat2id = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        # parse category file.
        with open(os.path.join(self.root, 'synsetoffset2category.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]

        # parse segment num file.
        with open('misc/num_seg_classes.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])

        # if a subset of classes is specified.
        if class_choice is not None:
            self.cat2id = {k: v for k, v in self.cat2id.items()
                           if k in class_choice}
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        self.datapath = []
        splitfile = os.path.join(
            self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        filelist = json.load(open(splitfile, 'r'))
        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat2id.values():
                self.datapath.append([
                    self.id2cat[category],
                    os.path.join(self.root, category, 'points', uuid + '.pts'),
                    os.path.join(self.root, category, 'points_label', uuid + '.seg')
                ])

        # category and corresponding category number
        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))

        # cache
        self.cache = {}
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
            fn, cls, point_set, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64) - 1
            if len(self.cache) < self.cache_size:
                self.cache[index] = [fn, cls, point_set, seg]

        # randomly sample self.npoints point from the origin point cloud.
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # normalize into a sphere with origin (0, 0, 0) and radius 1.
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        # data augmentation
        if self.data_augmentation and self.split == 'train':
            # random rotation
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)
            # random jitter
            point_set += np.random.normal(0, 0.02, size=point_set.shape)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class ModelNet40(data.Dataset):
    def __init__(self, root, npoints=1024, split='train', normalize=True, data_augmentation=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.cat2id = {}

        # load classname and class id
        with open('misc/modelnet40category2id.txt', 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = int(ls[1])
        self.id2cat = {v: k for k, v in self.cat2id.items()}

        # find all .h5 files
        with open(os.path.join(self.root, '{}_files.txt'.format(split)), 'r') as f:
            data_files = [os.path.join(self.root, line.strip().split('/')[-1]) for line in f]

        # load data from .h5 files
        point_clouds, labels = [], []
        for filename in data_files:
            with h5py.File(filename, 'r') as data_file:
                point_clouds.append(np.array(data_file['data']))
                labels.append(np.array(data_file['label']))
        self.pcs = np.concatenate(point_clouds, axis=0)
        self.lbs = np.concatenate(labels, axis=0)
    
    def __getitem__(self, index):
        point_cloud = self.pcs[index]
        label = self.lbs[index]
        classname = self.id2cat[label[0]]

        # select self.npoints from the original point cloud randomly
        choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
        point_cloud = point_cloud[choice, :]

        # normalize into a sphere whose radius is 1
        if self.normalize:
            point_cloud = point_cloud - np.mean(point_cloud, axis=0)
            dist = np.max(np.sqrt(np.sum(point_cloud  ** 2, axis=1)))
            point_cloud = point_cloud / dist

        # data augmentation - random rotation and random jitter
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)  # random jitter

        return point_cloud, label, classname

    def __len__(self):
        return len(self.pcs)


if __name__ == '__main__':
    shapenet_path = "/home/rico/Workspace/Dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0"
    train_dataset = ShapeNetPartDataset(root=shapenet_path, classification=False, class_choice=None, npoints=2048, split='train')
    test_dataset = ShapeNetPartDataset( root=shapenet_path, classification=False, class_choice=None, npoints=2048, split='test')
    print("\033[32mShapeNet\033[0m: {} training data and {} testing data".format(len(train_dataset), len(test_dataset)))
    pc, seg = train_dataset[random.randint(0, len(train_dataset))]
    print(pc.size(), seg.size())

    modelnet_path = '/home/rico/Workspace/Dataset/modelnet/modelnet40_hdf5_2048'
    train_dataset = ModelNet40(root=modelnet_path, split='train', data_augmentation=False, npoints=2048)
    test_dataset = ModelNet40(root=modelnet_path, split='test', data_augmentation=False, npoints=2048)
    
    print("\033[32mModelNet40\033[0m: {} training data and {} testing data".format(len(train_dataset), len(test_dataset)))
    
    point_cloud, label, classname = train_dataset[random.randint(0, len(train_dataset))]
    print(point_cloud.shape, label.shape, classname, label)
