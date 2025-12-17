"""Generate and visualise the results of the mini-experiment"""

import math
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from experiment.cfm_toy_model import CFMToyModel
from experiment.utils import get_device, set_seed, load_checkpoint
from experiment.cfm_train import BASE_CHECKPOINT_PATH, sample_circle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = get_device()
DEVICE = "cpu"


@dataclass
class VisualisationConfig:
    n_targets: int = 100
    seed: int = 42
    time_interval: float = 0.1
    device: str = DEVICE
    model_checkpoint_path: str | Path = BASE_CHECKPOINT_PATH / "cfm_toy_model_1000.pt"


def generate_coordinates(model, config):
    num_eval_samples_per_step = config.n_targets
    time_interval = config.time_interval
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        # Generate noised samples
        samples = torch.randn(
            (num_eval_samples_per_step, 2),
            device=config.device,
        )
        for timestep in tqdm(torch.arange(0, 1, time_interval)):
            timesteps = torch.full(
                (num_eval_samples_per_step, 1),
                timestep.item(),
                device=config.device,
            )
            input_tensor = torch.cat(
                (samples, timesteps),
                dim=-1,
            )
            # Predict the velocity at the given timestep
            velocity_prediction = model(input_tensor)
            # Evolve the flow foward in time with Euler step
            samples.add_(velocity_prediction, alpha=time_interval)

    samples_numpy = samples.cpu().float().numpy()
    logger.info(f"Samples shape: {samples_numpy.shape}")
    x_coordinates = samples_numpy[..., 0]
    y_coordinates = samples_numpy[..., 1]
    model.train(model_was_training)
    return x_coordinates, y_coordinates


def visualise_results(x_coordinates, y_coordinates, n_targets):
    plt.scatter(x_coordinates, y_coordinates, alpha=0.5)
    circle = sample_circle(n_targets)
    plt.scatter(circle[:, 0], circle[:, 1], alpha=0.5, c="red")
    plt.axis("equal")
    plt.title("Flow: Noise → Constrained Solution")
    plt.show()


if __name__ == "__main__":
    visualisation_config = VisualisationConfig()
    set_seed(visualisation_config.seed)
    model = CFMToyModel().to(device=visualisation_config.device)
    model, _ = load_checkpoint(
        model,
        visualisation_config.model_checkpoint_path,
        visualisation_config.device,
    )
    x_coordinates, y_coordinates = generate_coordinates(model, visualisation_config)
    visualise_results(
        x_coordinates,
        y_coordinates,
        visualisation_config.n_targets,
    )
