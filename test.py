"""
test.py

This script implements the Co-Evolutionary Training Loop.

Experiment Structure:
1. Initialize Populations: Create populations of Generators and Discriminators.
2. Initialize Evaluator: Set up metrics evaluation (SWD, Modes).
3. Evolutionary Loop (Generations):
    a. Selection: Choose parents based on fitness.
    b. Variation (Training): Train offspring using standard GAN algorithms.
    c. Evaluation: Compute fitness for offspring via interactions.
    d. Replacement: Replace worst performing individuals in the population.
4. Visualization: Plot loss curves, generated samples, and metrics in real-time.
"""

## Create a simple population of 5 generators and 5 discriminators

import models
import train
import evaluation
import dataset
import importlib

# Reload to capture any external file changes
importlib.reload(dataset)
importlib.reload(models)
importlib.reload(train)
importlib.reload(evaluation)

from models import GenerativeModel, DiscriminativeModel
from dataset import RingMixtureDataset
from train import train_one_epoch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from visualization import Visualizer

# Enable interactive mode for live updates in script
# plt.ion() # Handled by Visualizer

num_visualization_samples = 2000
batch_size = 32
dataset = RingMixtureDataset(n_samples=num_visualization_samples, collapse_to=[], k=8)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Plot setup: 2x2 dashboard
visualizer = Visualizer(dataset, samples_to_visualize=num_visualization_samples)

print("Setup complete. Starting training...")

proba = []

population_size = 5
dataloaders = []

results_list = []
for j in range(10):
    for i in range(population_size):
        collapse_to = [] 
        dataset = RingMixtureDataset(n_samples=num_visualization_samples, collapse_to=collapse_to, k=8)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloaders.append(dataloader)

    gen_population = GenerativeModel(population_size)
    disc_population = DiscriminativeModel(population_size, dataloaders=dataloaders)

    loss_criterion = nn.BCEWithLogitsLoss()

    gen_population.info()
    disc_population.info()

    gen_population.evaluate_population(disc_population, dataloader, num_batches=5)
    disc_population.evaluate_population(gen_population, dataloader, num_batches=5)

    gen_population.info()
    disc_population.info()

    # --- Evolutionary Training Loop ---
    #
    # Unlike standard GAN training which updates a single pair of models,
    # we evolve populations.
    #
    # Key Steps per Generation:
    # 1. Selection: Pick best performing models (lowest loss) as parents.
    # 2. Variation: 'Mutate' parents by training them for one epoch.
    #    - This creates offspring with updated parameters.
    # 3. Evaluation: Assess offspring fitness against the current opponent population.
    # 4. Replacement: Insert offspring into population and remove worst individuals.
    #    - This maintains population size while improving overall quality.

    for i in range(50):
        #print(" \n ======================================")
        #print(f" - Generation {i}")
        offspring_generator = gen_population.selection(2)
        offspring_discriminator = disc_population.selection(2)
        
        # Create optimizers for the specific offspring parameters (since they are deepcopies)
        disc_loss, gen_loss = train_one_epoch(
            offspring_generator, offspring_discriminator, 
            offspring_generator.optimizer, offspring_discriminator.optimizer, 
            offspring_discriminator.dataloader, 
            loss_criterion, d_steps=1, z_dim=2, device="cpu"
        )
        offspring_generator.generation += 1
        offspring_discriminator.generation += 1

        gen_population.insert_individual(offspring_generator)
        disc_population.insert_individual(offspring_discriminator)

        gen_population.evaluate_population(disc_population, dataloader, num_batches=5)
        disc_population.evaluate_population(gen_population, dataloader, num_batches=5)

        gen_population.remove_n_worst_individuals(1)
        disc_population.remove_n_worst_individuals(1)

        gen_population.info()
        disc_population.info()

        generated_samples, swd_metric, stats = gen_population.evaluate_metrics(dataset)


        #if i % 5 == 0:
        #    visualizer.update(i, disc_loss, gen_loss, generated_samples, swd_metric, stats)
        #    print(f"Final stats: SWD {swd_metric:.4f} | JS divergence {stats['js_to_uniform']:.4f} | Coverage: {stats['coverage']}")

    results_list.append(stats)
    print(f"RUN({j}) - Final stats: SWD {swd_metric:.4f} | JS divergence {stats['js_to_uniform']:.4f} | Coverage: {stats['coverage']}")

import pandas as pd
results_df = pd.DataFrame(results_list)
pd.DataFrame(results_df).to_csv("results-1.csv", index=False)
print(results_df.describe())

visualizer.show()
