# DQ-NAS: Differentiable Quality-of-Service-Aware Neural Architecture Search

This repository contains the source code for the paper "DQ-NAS: Differentiable Quality-of-Service-Aware Neural Architecture Search for Communication Networks". 

*(Please update the paper title and add a brief introduction to your work here.)*

## Requirements

To install the necessary dependencies, please run the following command:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── nas_main.py             # Main script to run the Neural Architecture Search (ANAS)
├── sim_main.py             # Main script to run network simulations for model validation
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── search/                 # Core logic for the NAS controller, model, and trainer
├── simulation/             # Network simulation logic (Fat-Tree) and validation scripts
├── utils/                  # Utility scripts for configuration, data loading, and models
├── tf_impl/                # Helper scripts for data processing (feature extraction, scaling)
└── saved/                  # Default directory for saving models and simulation results
```

## Usage

This project has been refactored to use command-line arguments for running experiments, ensuring reproducibility and ease of use.

### 1. Running the Neural Architecture Search

Use `nas_main.py` to start the architecture search. You can configure the search space and hyperparameters via arguments.

**Example:**
```bash
python nas_main.py --visible_gpus 0 --encoder_nodes 5 --decoder_nodes 3 --encoder_hidden 128 --controller_lr 0.0004
```

To see all available options, run:
```bash
python nas_main.py --help
```

### 2. Running Network Simulations

Use `sim_main.py` to run a network simulation using a trained model to evaluate its performance (e.g., delay, jitter).

**You must provide a model identifier for this script.**

**Example:**
```bash
python sim_main.py --model_identifier "ANAS" --k_val 4 --traffic_patterns "poisson" --visible_gpus 0 1
```

To see all available options, run:
```bash
python sim_main.py --help
```

### 3. Validating Simulation Results

Use `simulation/validate.py` to merge simulation trace files, calculate final metrics, and generate plots.

**Example:**
```bash
python simulation/validate.py --identifier "mse-5.14e-05" --traffic_patterns "poisson" "bursty"
```

To see all available options, run:
```bash
python simulation/validate.py --help
```
