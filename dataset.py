import math
import torch
from torch.utils.data import Dataset

class RingMixtureDataset(Dataset):
    def __init__(self, n_samples=20000, k=8, radius=2.0, sigma=0.08, seed=0, collapse_to=None):
        self.k = k
        self.radius = radius
        self.sigma = sigma

        if collapse_to is None:
            collapse_to = []

        g = torch.Generator().manual_seed(seed)

        if len(collapse_to) > 0:
            weights = torch.ones(k)
            for i in collapse_to:
                weights[i] = 10.0
            weights = weights / weights.sum()
            idx = torch.multinomial(weights, n_samples, replacement=True, generator=g)
            print(f"Dataset biased toward modes: {collapse_to}")
        else:
            idx = torch.randint(0, k, (n_samples,), generator=g)

        angles = 2 * math.pi * idx.float() / k
        centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
        noise = sigma * torch.randn(n_samples, 2, generator=g)
        self.x = (centers + noise).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]