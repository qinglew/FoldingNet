import numpy as np
import os
import json
import torch
import h5py
from tqdm import tqdm
import torch
import torch.utils.data as data
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.structures import Meshes
from pytorch3d.ops import (
    sample_points_from_meshes, 
    convert_pointclouds_to_tensor
)
from pytorch3d.renderer import TexturesVertex


#------------------------------------------------------------------------------------------------------
class ShapeNetCorePointClouds(ShapeNetCore):
    def __init__(
            self, 
            root,
            classes=None,
            npoints_from_mesh = 2048,
            npoints=1024, 
            split='train', 
            normalize=True, 
            data_augmentation=True,
            sample_npoints=True):
        super().__init__(data_dir=root, 
                         version=2, 
                         load_textures=False,
                         synsets=classes)
        self.npoints_from_mesh = npoints_from_mesh
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.data_augmentation = data_augmentation
        self.sample_npoints = sample_npoints

        # cache
        self.cache = {}
        self.cache_size = 18000

        # if torch.cuda.is_available():
        #     self.device = torch.device("cuda:0")
        #     torch.cuda.set_device(self.device)
        # else:
        #     self.device = torch.device("cpu")
        self.device = torch.device("cpu")

        # print(f"Synset ids of the dataset: {self.synset_ids}")
        # print(f"Model ids of the dataset: {self.model_ids}")

    def __getitem__(self, idx):
        # Load the mesh at the current idx (either from file or from cache)
        if idx in self.cache:
            verts, faces, label = self.cache[idx]
        else:
            model = self._get_item_ids(idx)
            model_path = os.path.join(
                self.shapenet_dir, model["synset_id"], model["model_id"], self.model_dir
            )
            verts, faces, _ = self._load_mesh(model_path)
            label = self.synset_dict[model["synset_id"]]
            if len(self.cache) < self.cache_size:
                self.cache[idx] = [verts, faces, label]


        # Convert the mesh into a point cloud
        model_textures = TexturesVertex(
            verts_features=torch.ones_like(verts, device=self.device)[None]
        )
        model_mesh = Meshes(
            verts=[verts.to(self.device)],   
            faces=[faces.to(self.device)],
            textures=model_textures
        )

        point_cloud = sample_points_from_meshes(model_mesh, self.npoints_from_mesh)
        point_cloud = convert_pointclouds_to_tensor(point_cloud)[0].squeeze(0)


        # randomly sample self.npoints points from the origin point cloud.
        if self.sample_npoints:
            if self.split == 'train': # random choice to increase data diversity
                choice = np.random.choice(len(point_cloud), self.npoints, replace=True)
            elif (self.split == 'test') or (self.split == 'val'): # deterministic choice: validation/test set must remain equal to itself at each evaluation step
                np.random.seed(1234)
                choice = np.random.choice(len(point_cloud), self.npoints, replace=False)
            point_cloud = point_cloud[choice, :]


        # normalize into a sphere with origin (0, 0, 0) and radius 1.
        if self.normalize:
            point_cloud = point_cloud - torch.mean(point_cloud, axis=0).unsqueeze(0)
            dist = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, axis=1)), 0)[0]
            point_cloud = point_cloud / dist

        # data augmentation
        if self.data_augmentation and self.split == 'train':
            # random rotation
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = torch.tensor([[np.cos(theta), -np.sin(theta)], 
                                            [np.sin(theta), np.cos(theta)]],
                                            dtype=torch.float32,
                                            device=self.device)
            point_cloud[:, [0, 2]] = point_cloud[:, [0, 2]].matmul(rotation_matrix)
            # random jitter
            point_cloud += torch.from_numpy(np.random.normal(0, 0.02, size=point_cloud.shape)).to(self.device)

        # get synset id of the label
        synset_ID = self.synset_inv[label]

        return point_cloud.cpu().numpy(), label, synset_ID

    def __len__(self):
        return len(self.model_ids)
#------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------
def save_point_clouds_dataset(split, 
                              data,
                              max_num_samples=2048,
                              save_dir = "../ShapeNet_data/ShapeNetCore_pointclouds_v2/"
    ):
    file_prefix = "data_" + split + "_n"
    point_clouds_dataset = np.empty((max_num_samples, 2048, 3), dtype=np.float32)
    categories_dataset = np.empty((max_num_samples, 1), dtype="|S10")
    synset_dataset = np.empty((max_num_samples, 1), dtype="|S10")
    file_id = 0
    counter = 0
    for i, sample in tqdm(enumerate(data), total=len(data)):
        # get data (point cloud, category, synset Id)
        pcloud, cat, id = sample
        # store values the datasets & increment the counter
        point_clouds_dataset[counter, :, :] = pcloud
        categories_dataset[counter, :] = cat
        synset_dataset[counter, :] = id
        # when the datasets are full save them in an hdf5 file
        if (counter == (max_num_samples - 1)) or (i == (len(data) - 1)):
            # in the last round, remove the part of the dataset that it is not needed
            if (i == (len(data) - 1)):
                cut_idx = i % max_num_samples
                point_clouds_dataset = point_clouds_dataset[:(cut_idx+1), ...]
                categories_dataset = categories_dataset[:(cut_idx+1), ...]
                synset_dataset = synset_dataset[:(cut_idx+1), ...]

            counter = 0
            file_name = file_prefix + str(file_id) + ".h5"
            print(f"Saving dataset {file_name}")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with h5py.File(os.path.join(save_dir, file_name), "w") as f:
                f.create_dataset(
                    name="point_clouds",
                    shape=point_clouds_dataset.shape,
                    data=point_clouds_dataset,
                    dtype=np.float32,
                    compression=None
                )

                f.create_dataset(
                    name="categories",
                    shape=categories_dataset.shape,
                    data=categories_dataset,
                    dtype=categories_dataset.dtype,
                    compression=None
                )

                f.create_dataset(
                    name="synsetId",
                    shape=synset_dataset.shape,
                    data=synset_dataset,
                    dtype=synset_dataset.dtype,
                    compression=None
                )
            
            file_id += 1
        counter += 1
#------------------------------------------------------------------------------------------------------



if __name__ == "__main__":

    # Load the datasets
    shapenet_path = "../ShapeNet_data/ShapeNetCore_split/"
    train_dataset = ShapeNetCorePointClouds(root=os.path.join(shapenet_path, "train"),
                                            npoints_from_mesh=2048,
                                            sample_npoints=False,
                                            data_augmentation=False,
                                            normalize=False)
    val_dataset = ShapeNetCorePointClouds(os.path.join(shapenet_path, "val"),
                                          npoints_from_mesh=2048,
                                          sample_npoints=False,
                                          data_augmentation=False,
                                          normalize=False)
    test_dataset = ShapeNetCorePointClouds(os.path.join(shapenet_path, "test"),
                                          npoints_from_mesh=2048,
                                          sample_npoints=False,
                                          data_augmentation=False,
                                          normalize=False)


    save_point_clouds_dataset(
        split="train",
        data=train_dataset,
    )

    save_point_clouds_dataset(
        split="test",
        data=test_dataset,
    )

    save_point_clouds_dataset(
        split="val",
        data=val_dataset,
    )

        




