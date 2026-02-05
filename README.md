# Coevolution for Multi-Model Neural Networks: Training Beyond End-to-End Gradients

This repository contains the code and resources for the tutorial "Coevolution for Multi-Model Neural Networks: Training Beyond End-to-End Gradients". It demonstrates how to combine Evolutionary Algorithms (EA) with Gradient-Based Learning to train populations of Generative Adversarial Networks (GANs).

## Overview

The tutorial explores:
1.  **Introductory Genetic Algorithms**: Simple examples (OneMax) to understand the core concepts of selection, crossover, and mutation.
2.  **Standard GAN Training**: Understanding the baseline dynamics of single-pair GAN training.
3.  **Population-Based Training (Co-evolution)**: Training populations of Generators and Discriminators to improve stability and diversity.
4.  **Evolutionary Strategies**: optimization of ensemble weights using non-gradient based methods.


## File Structure & Explanations

### Notebooks

- **`onemax_ea.ipynb`**
  - A standalone educational notebook introducing Genetic Algorithms.
  - Solves the "OneMax" problem (maximizing the number of 1s in a bitstring).
  - Used to teach the primitives of Evolution: Selection, Crossover, and Mutation.

- **`coevolutionary_GAN.ipynb`**
  - **The Main Tutorial Notebook**.
  - Implements a Competitive Co-evolutionary algorithm where a population of Generators evolves against a population of Discriminators.
  - Features:
    - Population initialization using `GenerativeModel` and `DiscriminativeModel`.
    - Co-evolutionary training loop.
    - **Ensemble Weight Evolution**: Optimizes the weights of the generator ensemble to minimize Sliced Wasserstein Distance (SWD) using Evolutionary Strategies (ES).

- **`simple_GAN.ipynb`**
  - An introductory notebook implementing a standard, single-pair GAN.
  - Serves as a baseline to understand the mode collapse problem on the Ring Dataset before moving to more complex population methods.
  - Uses the same `Visualizer` for real-time monitoring.

### Examples

- **`onemax_ea.py`**
  - A pure Python script implementation of the OneMax Genetic Algorithm.
  - Useful for running the GA without a notebook interface.

- **`binary_bilinear_ccea.py`**
  - Implements a Competitive Co-Evolutionary Algorithm for the Binary Bilinear Maximin Problem.
  - Demonstrates how two populations (X and Y) evolve against each other with opposing objectives.
  - Includes visualization of fitness dynamics over generations.

### Core Modules

- **`models.py`**
  - Defines the Neural Network architectures:
    - `Generator`: Mapping latent space to data space.
    - `Discriminator`: Classifying real vs. fake data.
  - Defines Population Wrappers:
    - `GenerativeModel`: Manages a population of Generators and implements `evolve_ensemble_weights`.
    - `DiscriminativeModel`: Manages a population of Discriminators.

- **`train.py`**
  - Contains `train_one_epoch`: The standard gradient-based training step for a single Generator-Discriminator pair.
  - Abstracts away the PyTorch boilerplate for computing GAN losses (BCE, etc.) and updating weights.

- **`dataset.py`**
  - `RingMixtureDataset`: A synthetic dataset consisting of 2D Gaussians arranged in a ring.
  - Ideal for visualizing "Mode Collapse" (when a GAN captures only a few modes of the real distribution).

- **`evaluation.py`**
  - Metrics for quantitative evaluation:
    - `sliced_wasserstein_distance`: A robust metric for measuring the distance between distributions.
    - `mode_stats`: Utilities to check how many modes of the Ring dataset are covered.

- **`visualization.py`**
  - `Visualizer`: A utility class for real-time potting within Jupyter Notebooks.
  - Displays:
    - Generated samples vs. Real samples (Scatter plot).
    - Loss curves (Generator vs. Discriminator).
    - Metrics (SWD scores over time).
    - Mode coverage ratios.

## Usage

1.  **Explore **`onemax_ea.ipynb`** if you are new to Evolutionary Algorithms.
2.  **Start with `simple_GAN.ipynb`** to understand the baseline problem.
3.  **Move to `coevolutionary_GAN.ipynb`** to see how populations and co-evolution solve mode collapse.

## Installation

You can set up the environment using either `pip` or `conda`.

### Using Conda

```bash
conda env create -f requirements.yml
conda activate coevolution-tutorial
```

### Using Pip

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch
- Matplotlib
- NumPy
- Pandas
- Jupyter / IPython
