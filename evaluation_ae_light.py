import torch
import numpy as np
import os
import pytorch_lightning as pl
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import (
    Iterable, Optional, Callable, 
    Union, List, Tuple, 
    Dict, Literal, Callable
)
from datasets import (
    ModelNet40_test, ShapeNetCore_test
)
from model import AutoEncoder
from chamfer_distance.chamfer_distance import ChamferDistance
from make_plots import save_3d_plots, plot_samples


#-------------------------------------------------------------------------------
def evaluate_and_predict(
        model: pl.LightningModule,
        test_dataloader: torch.utils.data.DataLoader,
        metrics: Iterable[Callable],
        device: torch.device
) -> Tuple[Dict[str, float], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Evaluate the current model on a test dataset using a set of pre-defined metrics and 
    return model predictions.
    
    Parameters:
    ----------- 
        model: (pl.LightningModule)
            The pytorch model to be evaluated

        test_dataloader: (torch.utils.data.DataLoader)
            The test_dataset (torch Dataset object) which the model should be evaluated on 
        
        metrics: (Iterable[Callable])
            A list of nn.Module's corresponding to the metrics which the model should be 
            evaluated on.
        
        device: (torch.device)
            The device on which data are processed
    
    Returns:
    --------
        results: (Dict[str, float]) 
            A dictionary associating the name of each test metric
            the computed value on the test set.

        predictions: (List[Tuple[torch.Tensor, torch.Tensor]]) 
            A list of tuples whose content is a pair of tensors, one storing the
            predicted point clouds, and one the associated (ground truth) labels.
    """
    
    func2name = {
        ChamferDistance: 'chamfer_dist',
    }      

    model.eval()  

    results = {}
    predictions = []
    for metric in metrics:
        metric_name = func2name[type(metric)]
        total_value = 0
        with torch.no_grad():
            for data in tqdm(test_dataloader, desc="Evaluating test samples: "):
                point_clouds, labels = data
                point_clouds = point_clouds.permute(0, 2, 1)
                point_clouds = point_clouds.to(device)
                preds = model(point_clouds)
                # compute the metric for the batch
                ls = metric(
                    point_clouds.permute(0, 2, 1), preds.permute(0, 2, 1)
                )
                total_value += ls.item()
                # store predictions
                predictions.append((preds, labels))

        # calculate the mean 
        mean_value = total_value / len(test_dataloader)
        results[metric_name] = mean_value
    
    return results, predictions
    


def main(
        data_dir: str,
        ckpt_dir: str,
        model_name: Literal['ModelNet', 'ShapeNet']
):
    
    assert model_name in ('ModelNet', 'ShapeNet'), "Unknown model name."
    
    if model_name == 'ModelNet':
        test_dataset = ModelNet40_test(
            root=data_dir, 
            npoints=2048, 
            split='test', 
            normalize=True,
        )
    elif model_name == 'ShapeNet':
        test_dataset = ShapeNetCore_test(
            root=data_dir, 
            npoints=2048, 
            split='test', 
            normalize=True,
        )

    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    ae = AutoEncoder().load_from_checkpoint(ckpt_dir).to("cuda")

    cd_loss = ChamferDistance()

    values, _ = evaluate_and_predict(ae, test_dataloader, cd_loss, "cuda")

    print(f"Chamfer Loss evaluated on {len(test_dataset)} samples in the test set: {values}")

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