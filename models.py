"""
models.py

This module defines the core neural network architectures (Generator, Discriminator)
and the population management logic for the Co-evolutionary GAN.

Key Concepts:
- Generators: Attempt to create realistic data to fool discriminators.
- Discriminators: Attempt to distinguish between real data and fake data produced by generators.
- Populations: Managing groups of models to enable evolutionary selection and variation.
"""

import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim


from evaluation import sliced_wasserstein_distance, mode_stats, compute_mode_centers

class Generator(nn.Module):
    """
    Feed-forward neural network representing a Generator.
    Maps a latent vector z to data space (x).
    """
    def __init__(self, z_dim=2, h=16, id=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 2),
        )

        self.fitness = 0.0
        self.name = "Generator"
        self.id = id
        self.generation = 0

    def forward(self, z):
        return self.net(z)

    def info(self):
        return f"Generator {self.id} | Fitness: {self.fitness} | Generation: {self.generation}"


class Discriminator(nn.Module):
    """
    Feed-forward neural network representing a Discriminator.
    Maps data points (x) to a scalar score (logits).
    """
    def __init__(self, h=16, id=0, dataloader=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, h), nn.LeakyReLU(0.2),
            nn.Linear(h, h), nn.LeakyReLU(0.2),
            nn.Linear(h, 1)  # logits
        )

        self.fitness = 0.0
        self.name = "Discriminator"
        self.id = id
        self.generation = 0
    
        self.dataloader = dataloader
        
    def forward(self, x):
        return self.net(x).squeeze(1)

    def info(self):
        return f"Discriminator {self.id} | Fitness: {self.fitness} | Generation: {self.generation}"

class Populations:
    """
    Base class for managing populations of models.
    Implements common evolutionary operations like selection and fitness evaluation.
    """
    def __init__(self):
        self.population = []

    def selection(self, n):
        """
        Performs Tournament Selection.
        
        Randomly samples 'n' individuals and selects the best one (lowest fitness/loss).
        Returns a deep copy to ensure variations (training) don't affect the original population
        until explicit insertion.
        """
        if n > len(self.population):
            n = len(self.population)
        tournament = random.sample(self.population, n)
        # Assumes fitness is loss (lower is better)
        best = min(tournament, key=lambda x: x.fitness)
        offspring = copy.deepcopy(best)
        offspring.optimizer = optim.Adam(offspring.parameters(), lr=2e-3, betas=(0.5, 0.999))
        return offspring

    def insert_individual(self, individual):
        self.population.append(individual)

    def get_n_worst_individuals(self, n):
        # Sort by fitness descending (higher is worse assuming loss)
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]

    def remove_n_worst_individuals(self, n):
        worst = self.get_n_worst_individuals(n)
        for ind in worst:
            if ind in self.population:
                self.population.remove(ind)

    def evaluate_population(self, opponent_population, dataloader, num_batches=100):
        """
        All-vs-All Evaluation Strategy.
        
        Each individual in the current population competes against every individual
        in the opponent population. The average loss across all matches determines fitness.
        
        Fitness = Average Loss (Lower is better for selection stability in this implementation)
        """
        # Determine device from the first model in population
        device = next(self.population[0].parameters()).device
        bce = nn.BCEWithLogitsLoss()
        
        is_generative = isinstance(self, GenerativeModel)
        
        for individual in self.population:
            total_loss = 0.0
            count = 0
            
            individual.eval() # Set to eval mode for evaluation
            
            for opponent in opponent_population.population:
                opponent.eval()
                
                # Accumulate loss over the dataloader
                match_loss = 0.0
                batch_count = 0
                
                with torch.no_grad():
                    for i, real_batch in enumerate(dataloader):
                        if i >= num_batches:
                            break
                        real_batch = real_batch.to(device)
                        batch_size = real_batch.size(0)
                        
                        # Generate z
                        z_dim = individual.net[0].in_features if is_generative else opponent.net[0].in_features
                        z = torch.randn(batch_size, z_dim, device=device)
                        
                        if is_generative:
                            # Self is Generator, Opponent is Discriminator
                            # G loss: D(G(z)) -> 1
                            fake = individual(z)
                            d_fake = opponent(fake)
                            loss = bce(d_fake, torch.ones(batch_size, device=device))
                        else:
                            # Self is Discriminator, Opponent is Generator
                            # D loss: D(x) -> 1, D(G(z)) -> 0
                            fake = opponent(z)
                            d_real = individual(real_batch)
                            d_fake = individual(fake)
                            loss = bce(d_real, torch.ones(batch_size, device=device)) + \
                                   bce(d_fake, torch.zeros(batch_size, device=device))
                        
                        match_loss += loss.item()
                        batch_count += 1
                
                # Average loss for this match
                if batch_count > 0:
                    total_loss += match_loss / batch_count
                    count += 1
            
            # Fitness is the mean loss of all competitions
            if count > 0:
                individual.fitness = total_loss / count
            
            individual.train() # Reset to train mode


class GenerativeModel(Populations):
    """
    Class that represents a population of generators, defining the generative model 
    trained using a Competitive Coevolutionary Algorithm.
    """
    def __init__(self, population_size, z_dim=2, h=16):
        super().__init__()
        self.population = [Generator(z_dim, h, id=i) for i in range(population_size)]
        self.ensemble_weights = [1.0 / population_size for _ in range(population_size)]

    def info(self):
        for i, individual in enumerate(self.population):
            print(f"Individual {i}: {individual.info()}")

    def evaluate_metrics(self, dataset, device="cpu", num_samples=2000, weights=None):
        """
        Evaluates the entire generative population as a mixture of experts.
        
        1. Compiles generated samples from ALL generators in the population.
        2. Computes Sliced Wasserstein Distance (SWD) against real data.
        3. Computes Mode Statistics (Coverage, JS Divergence) to measure diversity.
        """
        centers = compute_mode_centers(dataset.k, dataset.radius).to(device)
        
        if weights is None:
            weights = self.ensemble_weights
            
        # Ensure weights sum to 1 and are valid
        weights_tensor = torch.tensor(weights, device='cpu', dtype=torch.float32)
        weights_tensor = torch.relu(weights_tensor)
        if weights_tensor.sum() == 0:
            weights_tensor = torch.ones_like(weights_tensor)
        weights_tensor /= weights_tensor.sum()
        
        # Determine number of samples per generator based on weights
        counts = (weights_tensor * num_samples).long()
        
        # Adjust for rounding errors to ensure total samples == num_samples
        diff = num_samples - counts.sum().item()
        if diff > 0:
            # Add remainder to the ones with highest weights or just first ones
            counts[:diff] += 1
            
        fake_vis_list = []
        with torch.no_grad():
            for i, G in enumerate(self.population):
                n = counts[i].item()
                if n > 0:
                    z_dim = G.net[0].in_features
                    fake_vis_list.append(G(torch.randn(n, z_dim, device=device)).cpu())
        
        if len(fake_vis_list) > 0:
            fake_vis = torch.cat(fake_vis_list, dim=0)
        else:
            # Fallback if no samples (shouldn't happen with proper weights)
            fake_vis = torch.empty(0, 2)
        
        # Use simple slice of dataset for real reference, assuming shuffled or representative
        real_vis = dataset.x[:num_samples].to(device)
        
        swd = sliced_wasserstein_distance(real_vis, fake_vis.to(device), device=device)
        stats = mode_stats(fake_vis.to(device), centers, sigma=dataset.sigma)
        
        return fake_vis, swd, stats

    def evolve_ensemble_weights(self, dataset, metric="swd", num_samples=2000, n_perturbations=10, sigma=0.05, learning_rate=0.1, device="cpu"):
        """
        Applies Evolutionary Strategies (ES) to optimize self.ensemble_weights.
        
        Args:
            dataset: Dataset object
            metric: 'swd' or 'js_to_uniform'
            num_samples: Samples for evaluation
            n_perturbations: Number of noise samples for ES
            sigma: Noise std dev
            learning_rate: LR for weight update
        """
        
        current_weights = torch.tensor(self.ensemble_weights, dtype=torch.float32)
        
        # noise samples
        noise = torch.randn(n_perturbations, len(self.population)) * sigma
        
        rewards = torch.zeros(n_perturbations)
        
        for i in range(n_perturbations):
            # Perturb weights (mirrored sampling for better stability)
            # We try w + noise and w - noise? Or just w + noise.
            # OpenAI ES uses w + noise[i], but mirrored sampling uses (F(w+e) - F(w-e))
            # Let's use simple non-mirrored for brevity or mirrored as it's better? 
            # The prompt just says "applies Evolutionary Strategies". 
            # Mirrored sampling is standard for ES gradients.
            
            # Use w + noise
            w_pert = current_weights + noise[i]
            # Normalize/project
            w_pert = torch.relu(w_pert)
            if w_pert.sum() > 0:
                w_pert /= w_pert.sum()
            else:
                w_pert = torch.ones_like(w_pert) / len(w_pert)
            
            # Evaluate using list conversion for compatibility
            _, swd, stats = self.evaluate_metrics(dataset, device=device, num_samples=num_samples, weights=w_pert.tolist())
            
            value = swd if metric == "swd" else stats["js_to_uniform"]
            
            # We want to MINIMIZE the metric, so reward is negative value
            rewards[i] = -value
            
        # Standard ES gradient estimate:
        # grad ~ (1 / (n * sigma)) * sum(noise[i] * reward[i])
        # Centering rewards is crucial
        rewards_centered = (rewards - rewards.mean())
        if rewards_centered.std() > 1e-6:
            rewards_centered /= rewards_centered.std()
            
        grad = torch.matmul(noise.T, rewards_centered) / (n_perturbations * sigma)
        
        # Update weights (Gradient Ascent on Reward -> Gradient Descent on Metric)
        # new_weights = current + lr * grad
        new_weights = current_weights + learning_rate * grad
        
        # Project back to simplex
        new_weights = torch.relu(new_weights)
        if new_weights.sum() > 0:
            new_weights /= new_weights.sum()
        else:
            new_weights = torch.ones_like(new_weights) / len(new_weights)
            
        self.ensemble_weights = new_weights.tolist()
        #print(f"Updated ensemble weights: {[f'{w:.4f}' for w in self.ensemble_weights]}")

        



class DiscriminativeModel(Populations):
    """
    Class that represents a population of discriminators, defining the discriminative model 
    trained using a Competitive Coevolutionary Algorithm.
    """
    def __init__(self, population_size, h=16, dataloader=None):
        super().__init__()
        self.dataloader = dataloader
        self.population = []
        
        print("Discriminative Model initialized.")
        self.population_size = population_size
        for i in range(population_size):
            self.population.append(Discriminator(h=h, id=i, dataloader=dataloader))  

    def info(self):
        for i, individual in enumerate(self.population):
            print(f"Individual {i}: {individual.info()}")
          

