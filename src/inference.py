import os
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def inference(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run inference with the trained model.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    # Set seed for random number generators
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Load checkpoint
    if cfg.ckpt_path:
        log.info(f"Loading checkpoint from {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        log.warning("No checkpoint path provided!")

    # Set device
    device = cfg.inference.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Prepare data
    datamodule.prepare_data()
    datamodule.setup("predict")

    # Run inference
    log.info("Starting inference!")

    # Get predict dataloader for inference (includes all data)
    predict_dataloader = datamodule.predict_dataloader()

    output_list = []
    model.eval()
    # ------------------------ INFERENCE LOOP ------------------------ #
    with torch.no_grad():
        for batch in tqdm(predict_dataloader, desc="Predicting"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Collect outputs
            output_list.append(outputs)

    # ------------------------ INFERENCE LOOP ------------------------ #

    # Return metric_dict and object_dict as expected by task_wrapper
    metric_dict = {
        "inference_completed": True,
        "num_predictions": len(all_outputs),
        "output_path": plot_save_path,
    }
    object_dict = {
        "predictions": all_outputs,
        "config": cfg,
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for inference.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # Apply extra utilities
    extras(cfg)

    # Run inference
    inference(cfg)


if __name__ == "__main__":
    main()
