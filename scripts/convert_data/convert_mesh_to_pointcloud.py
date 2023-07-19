import numpy as np
import os
import h5py
from open3d.io import read_point_cloud, write_point_cloud
from tqdm import tqdm

def convert_mesh_to_coords(
        file_path,
        label,
        save_dir=None
):
    """
    Convert a mesh (.ply, .stl, .vtk, ...) into a point cloud.

    args:
        file_path: the path to the file to convert
        save_dir: the path to the directory to store the converted file in
        label: a string defining the class of the current mesh

    return:
        None in the case save_dir is specified and saving was successfull, 
        or a np.ndarray containing the point cloud otherwise.
    """
    # get file name and file extensione
    fname = os.path.basename(file_path)
    fext = fname[(fname.find(".")+1):]

    if fext not in ["ply", "stl", "vtk"]:
        raise ValueError("The file extension should be among these ones: "".ply"", "".stl"", "".vtk""")

    pc = read_point_cloud(file_path, format=fext)

    pc = np.asarray(pc.points)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = fname.replace(fext, "h5")
        with h5py.File(os.path.join(save_dir, save_name), "w") as f_out:
            f_out.create_dataset(
                name="coordinates",
                data=pc
            )
            f_out.create_dataset(
                name="label",
                data=label
            )
        return
    else:
        return pc, label


if __name__=="__main__":

    path_to_file_dir = "/nas/groups/iber/Users/Federico_Carrara/3d_tissues_preprocessing_and_segmentation/meshes/bladder_cells/MBC19_S5_St1_Crop_GFP_clean_top/"
    file_names = os.listdir(path_to_file_dir)
    file_paths = [os.path.join(path_to_file_dir, file_name) for file_name in file_names]

    for fpath in tqdm(file_paths, desc="Converting files: "):
        convert_mesh_to_coords(
            file_path=fpath, 
            label="cuboidal",
            save_dir="/nas/groups/iber/Users/Federico_Carrara/foldingnet/Cells_data/bladder_MBC19_top/",
        )

    print("All the files have been successfully converted!")