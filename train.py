import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from evaluation import plot_results, sliced_wasserstein_distance, mode_stats, compute_mode_centers

def train_one_epoch(generator, discriminator, optG, optD, loader, bce, d_steps=1, z_dim=2, device="cpu"):
    """
    Standard GAN Training Step.
    
    This function performs one epoch of training for a single Generator-Discriminator pair.
    
    Training Dynamics:
    1. Discriminator Step:
       - Objective: Maximize log(D(x)) + log(1 - D(G(z)))
       - Loss: Binary Cross Entropy (BCE) with targets 1 for real, 0 for fake.
       
    2. Generator Step:
       - Objective: Maximize log(D(G(z)))
       - Loss: Non-saturating loss (BCE with target 1). Minimizing log(1 - D(G(z))) suffers from vanishing gradients when D is strong.
    """
    lossD_item = 0.0
    lossG_item = 0.0
    
    for real in loader:
        real = real.to(device)
        number_of_samples = real.size(0)

        # ---- Train D ----
        for _ in range(d_steps):
            z = torch.randn(number_of_samples, z_dim, device=device)
            fake = generator(z).detach()

            lossD = bce(discriminator(real), torch.ones(number_of_samples, device=device)) + \
                    bce(discriminator(fake), torch.zeros(number_of_samples, device=device))

            optD.zero_grad()
            lossD.backward()
            optD.step()
            lossD_item = lossD.item()

        # ---- Train G (non-saturating) ----
        z = torch.randn(number_of_samples, z_dim, device=device)
        fake = generator(z)
        lossG = bce(discriminator(fake), torch.ones(number_of_samples, device=device))

        optG.zero_grad()
        lossG.backward()
        optG.step()
        lossG_item = lossG.item()
        
    return lossD_item, lossG_item
