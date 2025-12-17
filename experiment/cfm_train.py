"""Train the CFM toy model"""

import math
import time
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm.auto import tqdm

from experiment.cfm_toy_model import CFMToyModel
from experiment.utils import (
    set_seed,
    get_device,
    save_checkpoint,
    count_model_params,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
BASE_CHECKPOINT_PATH = _PROJECT_ROOT / "training_checkpoints"
BASE_CHECKPOINT_PATH.mkdir(exist_ok=True, parents=True)

DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")


@dataclass
class TrainingConfig:
    device: str = DEVICE
    seed: int = 42
    learning_rate: float = 1e-3
    total_batch_size: int = 64
    eval_step_period: int = 200
    n_training_steps: int = 1000
    checkpoint_dir: str | Path = BASE_CHECKPOINT_PATH
    save_checkpoint_period: int = 250


def sample_circle(n_samples: int) -> Tensor:
    """Target: points on unit circle"""
    theta = torch.rand(n_samples) * 2 * math.pi
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


def train(model, optimiser, config):
    training_start_time = time.perf_counter()
    model.train()
    for step in tqdm(range(config.n_training_steps + 1)):
        final_step = step == config.n_training_steps
        # Sample a random timestep from a uniform distribution
        timestep = torch.rand(1, device=config.device)
        timesteps = torch.full(
            (config.total_batch_size, 1),
            timestep.item(),
            device=config.device,
        )
        # Sample from gaussian noise
        x0 = torch.randn(
            (config.total_batch_size, 2),
            device=config.device,
        )
        # Sample from the true distribution
        x1 = sample_circle(config.total_batch_size).to(device=config.device)

        # Linearly interpolate between the distributions (optimal transport)
        xt = (1 - timesteps) * x0 + timesteps * x1

        # Target velocity
        target_velocity = x1 - x0

        # Predicted velocity
        input_tensor = torch.cat((xt, timesteps), dim=-1)
        predicted_velocity = model(input_tensor)

        # Flow matching loss
        loss = F.mse_loss(predicted_velocity, target_velocity)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if final_step or step % config.eval_step_period == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.4f}")

        if final_step or step % config.save_checkpoint_period == 0:
            save_checkpoint(
                config.checkpoint_dir,
                model.state_dict(),
                optimiser.state_dict(),
                step,
            )
    training_end_time = time.perf_counter()
    logger.info(
        f"Training finished in: {training_end_time - training_start_time:.2f} seconds!"
    )
    return model


def main():
    cfm_training_config = TrainingConfig()
    set_seed(cfm_training_config.seed)
    model = CFMToyModel().to(device=cfm_training_config.device)
    # Note: all the model parameters are trainable anyway in our case
    num_trainable_parameters = count_model_params(model, trainable_only=True)
    logger.info(f"Number of trainable parameters: {num_trainable_parameters:,}")
    adamw_optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=cfm_training_config.learning_rate,
    )
    train(model, adamw_optimiser, cfm_training_config)


if __name__ == "__main__":
    main()
