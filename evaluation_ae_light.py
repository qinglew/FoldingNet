import torch
import numpy as np
import os
import pytorch_lightning as pl
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import ModelNet40, ShapeNetCorePointsClouds_v2
from model import AutoEncoderLight
from chamfer_distance.chamfer_distance import ChamferDistance
from make_plots import save_3d_plots, plot_samples


def evaluate_and_predict(
        model,
        test_dataloader,
        metrics,
        device
) -> list:
    """
    Evaluate the current model on a test dataset using a set of pre-defined metrics and return model predictions
    args: 
        model: the pytorch model to be evaluated
        test_dataloader: the test_dataset (torch Dataset object) which the model should be evaluated on 
        metrics: a single function, or a list of functions corresponding to the metrics which the model should be evaluated with
        device: the device on which data are processed
    return:
        values: a single values, or a list containing the value for each metric
        predictions: a np.ndarray containing the predictions for the test samples
    """
    model.eval()

    if type(metrics) != "list":
        metrics = [metrics]        

    values = []
    predictions = []
    for metric in metrics:
        total_value = 0
        with torch.no_grad():
            for data in tqdm(test_dataloader, desc="Evaluating test samples: "):
                point_clouds, labels, _ = data
                point_clouds = point_clouds.permute(0, 2, 1)
                point_clouds = point_clouds.to(device)
                preds = model(point_clouds)
                # compute the metric for the batch
                ls = metric(point_clouds.permute(0, 2, 1), preds.permute(0, 2, 1))
                total_value += ls.item()
                # store predictions
                predictions.append((preds, labels))

        # calculate the mean 
        mean_value = total_value / len(test_dataloader)
        values.append(mean_value)
    
    if len(metrics) > 1:
        return values, predictions
    else:
        return values[0], predictions
    
def main(
        data_dir,
        ckpt_dir,
        model_name
):
    test_dataset = ModelNet40(root=data_dir, 
                              npoints=2048, 
                              split='test', 
                              normalize=True, 
                              data_augmentation=False
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    ae = AutoEncoderLight().load_from_checkpoint(ckpt_dir).to("cuda")

    cd_loss = ChamferDistance()

    values, preds = evaluate_and_predict(ae, test_dataloader, cd_loss, "cuda")

    print(f"Chamfer Loss evaluated on {len(test_dataset)} samples in the test set: {values}")
    print(len(preds))

    plot_samples(
        gt_dataset=test_dataset,
        model=ae,
        save_dir="./images/",
        model_name=model_name,
        categories="all",
        make_3d_plots=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="./lightning_logs/version_95/checkpoints/best_checkpoint.ckpt")
    parser.add_argument('--model_name', type=str, default="ModelNet40")
    parser.add_argument('--data_root', type=str, default="../ModelNet40_data/modelnet40_ply_hdf5_2048/")
    args = parser.parse_args()

    main(
        data_dir=args.data_root,
        model_name=args.model_name,
        ckpt_dir=args.checkpoint_dir
    )