import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pickle

def make_3d_plot(data, figure=None):
    # if figure is None, then it means we have to create a new one
    if not figure:
        fig = plt.figure()
        subplot_pos = 121
        text2D_coords = (0.45, 0.05)
        title="Ground Truth"
    else:
        fig=figure
        subplot_pos = 122
        text2D_coords = (0.55, 0.05)
        title="Prediction"

    ax = fig.add_subplot(121, projection='3d')

    ax._axis3don = False
    ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], s=5, cmap="red")
    ax.set_title(title)
    ax.text2D(text2D_coords[0], 
              text2D_coords[1], 
              f"npoints = {data.shape[0]}", 
              transform=ax.transAxes, 
              horizontalalignment='center')

    return fig


def save_3d_plots(gt_data, pred_data, categories, save_dir, model_name):
    """
    Make and save 3d plots of the ground truth and predicted point clouds.

    args:
        - gt_data: an iterable whose elements are np.ndarray or torch.tensor 
        storing the ground truth point clouds coordinates. 
        Shape is (n_pcs, n_points, 3).
        - pred_data: an iterable whose elements are np.ndarray or torch.tensor 
        storing the predicted point clouds coordinates. 
        Shape is (n_pcs, n_points, 3).
        - categories: an iterable whose elements are strings storing the name
        of the category associated to the samples. Shape is (n_pcs, 1) or (n_pcs).
        - save_dir: the path to the directory in which the plots will be saved.
        - model_name: a string containing the name of the model that serves
        to create the folder and give the name to the plots.

    Note: the plots will be saved in "./save_dir/model_name/3d_plots". 
    Each file will be called (category + "_3d_plot.fig.pickle).
    """
    # set the directory to save the plots in
    path_to_save_dir = os.path.join(save_dir, model_name, "3d_plots")
    if not os.path.exists(path_to_save_dir):
            os.makedirs(path_to_save_dir)
    
    for gt, pred, cat in zip(gt_data, pred_data, categories):
        # # n.b. pred has shape (1, 3, mpoints) -> transform to (mpoints, 3)
        # pred = pred.squeeze(0)
        # pred = torch.einsum("ij->ji", pred)

        pred = pred.cpu()

        # Make the plots
        fig = make_3d_plot(gt, None)
        fig = make_3d_plot(pred, None)

        # save the 3d plots in fig.pickle format
        file_name =  cat + "_3d_plot" + ".fig.pickle"
        with open(os.path.join(path_to_save_dir, file_name), 'wb') as f:
            pickle.dump(fig, f)

        plt.close()


def plot_samples(
        gt_dataset,
        model,
        save_dir,
        model_name,
        categories="all",
        make_3d_plots=True,
        make_2d_plots=True,
        make_tsne_plots=True,
        device="cuda"
) -> None:
    """
    
    """
    # take a reduced dataset containing only one sample for each category
    gt_dataset.sample_categories(categories)
    reduced_gt_dataset = gt_dataset
    # reduced_gt_dataloader = DataLoader(reduced_gt_dataset, batch_size=1, shuffle=False)

    # create a list of ground truth and predicted point clouds and a list
    # of the associated categories
    gt_samples, pred_samples, labels = [], [], []
    model.eval()
    # count=1
    for gt_sample in reduced_gt_dataset:
        gt_samples.append(gt_sample[0])
        labels.append(gt_sample[2])
        # if count == 1:
        #     print(gt_sample)
        #     print(gt_sample[0].shape)
        with torch.no_grad():
            curr_gt_sample = gt_sample[0].unsqueeze(0).permute(0, 2, 1).to(device)
            # print(f"curr_gt_sample shape: {curr_gt_sample.shape}")
            pred_sample = model(curr_gt_sample)
            # convert to (m_points, 3) shape
            pred_sample = pred_sample.squeeze(0).permute(1, 0)
            pred_samples.append(pred_sample)
        
    if make_3d_plots:
        save_3d_plots(
            gt_data=gt_samples,
            pred_data=pred_samples,
            categories=labels,
            save_dir=save_dir,
            model_name=model_name
        )
    
    # if make_2d_plots:
    

    # if make_tsne_plots:


def open_3d_plot(file_path):
    """
    Open an interactive 3D plot in a window.

    args:
        file_path: path to the .fig.pickle file

    """
    with open(file_path, 'rb') as f:
        fig = pickle.load(f)
    
    plt.show()


if __name__=="__main__":
    open_3d_plot("images/ModelNet40/3d_plots/wardrobe_3d_plot.fig.pickle")
