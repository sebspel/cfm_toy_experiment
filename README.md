# Emergent Coordination in CFM

## Author
Sebastian Sjostrom

## Overview
This repository is a toy experiment for demonstrating collective emergent behaviour in a conditional flow matching (CFM) model with learned particle interactions.

### Framework
- **Source Distribution**: Gaussian noise (standard normal for training).
- **Target Distribution**: Uniformly spaced points on a unit circle.
- **OT Pairing**: Source and target distribution samples are optimal transport (OT) matched via a linear sum assignment.
- **Model Architecture**: Standard MLP for regressing the velocity field combined with a separate MLP taking pairwise features as inputs.

## Collective Dynamics
The particles collectively exhibit coordinated swirling dynamics to fill the circumference of the circle with approximately uniform spacing. Notably, the particles attempt to avoid clustering due to the learned pairwise interactions without any explicit programming.

## Installation
```bash
pip install uv
uv sync
```

## Usage
1. Train the CFM model for 10,000 steps:
```bash
uv run -m experiment.cfm_train
```
2. Generate the animation and visualise the emergent behaviour:
```bash
uv run -m experiment.visualise_results
```

