import torch
import math
import matplotlib.pyplot as plt

def compute_quality_metrics(real_data, gen_data, dataset_obj=None, device="cpu"):
    # Wrapper for SWD and Mode Stats
    k = 8
    radius = 2.0
    sigma = 0.08
    if dataset_obj:
        k = getattr(dataset_obj, 'k', 8)
        radius = getattr(dataset_obj, 'radius', 2.0)
        sigma = getattr(dataset_obj, 'sigma', 0.08)
    
    centers = compute_mode_centers(k, radius).to(device)
    swd = sliced_wasserstein_distance(real_data, gen_data, device=device)
    stats = mode_stats(gen_data, centers, sigma=sigma)
    return swd, stats

def compute_metrics(G, real_vis, device, centers, dataset, vis_n, z_dim):
    """
    Computes metrics for the current generator.
    
    Args:
        G: Generator model
        real_vis: Batch of real data for comparison
        device: Device to run on
        centers: Mode centers tensor
        dataset: Dataset object (containing sigma)
        vis_n: Number of samples to generate for visualization/metrics
        z_dim: Latent dimension size
        
    Returns:
        fake_vis: Generated samples (CPU)
        swd: Sliced Wasserstein Distance
        stats: Dictionary of mode statistics
    """
    with torch.no_grad():
        fake_vis = G(torch.randn(vis_n, z_dim, device=device)).cpu()
        swd = sliced_wasserstein_distance(real_vis, fake_vis, device=device)
        stats = mode_stats(fake_vis.to(device), centers, sigma=dataset.sigma)
    return fake_vis, swd, stats

def sliced_wasserstein_distance(real, fake, num_projections=128, p=2, device="cpu", seed=0):
    # Sliced Wasserstein Distance
    # Project data onto random directions and compare distributions
    batch_size = real.size(0)
    dim = real.size(1)
    
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        
    projections = torch.randn(num_projections, dim, device=device, generator=rng)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)
    
    # Project the data
    real_projections = real @ projections.T
    fake_projections = fake @ projections.T
    
    # Sort the projections
    real_projections, _ = torch.sort(real_projections, dim=0)
    fake_projections, _ = torch.sort(fake_projections, dim=0)
    
    # Compute Distance
    dist = torch.abs(real_projections - fake_projections).pow(p).mean()
    return dist.pow(1/p).item()

def compute_mode_centers(k, radius):
    angles = torch.linspace(0, 2 * math.pi, k + 1)[:-1]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    return centers

def mode_stats(x_fake, centers, sigma=0.08, min_count_for_covered=20):
    # Compute counts and ratios of samples near each mode center
    # x_fake: (N, 2)
    # centers: (k, 2)
    
    k = centers.size(0)
    
    # Distance from each sample to each center
    # x_fake[:, None, :] shape (N, 1, 2)
    # centers[None, :, :] shape (1, k, 2)
    dists = torch.norm(x_fake[:, None, :] - centers[None, :, :], dim=2) # (N, k)
    
    nearest_dist, nearest_idx = dists.min(dim=1)
    
    counts = torch.zeros(k, dtype=torch.long, device=x_fake.device)
    for i in range(k):
        counts[i] = (nearest_idx == i).sum()
        
    total = x_fake.size(0)
    ratios = counts.float() / total
    
    coverage = int((counts >= min_count_for_covered).sum().item())
    
    # JS divergence to uniform
    u = torch.full((k,), 1.0 / k, device=x_fake.device)
    eps = 1e-12
    p = ratios.clamp(min=eps)
    m = 0.5 * (p + u)
    js = 0.5 * (p * (p / m).log()).sum() + 0.5 * (u * (u / m).log()).sum()
    
    overall_mean_dist = float(nearest_dist.mean().item())
    within_2sigma = float((nearest_dist <= (2.0 * sigma)).float().mean().item())
    
    return {
        "counts": counts,
        "ratios": ratios,
        "coverage": coverage,
        "js_to_uniform": js.item(),
        "overall_mean_dist": overall_mean_dist,
        "pct_within_2sigma": within_2sigma,
    }

def plot_results(real_data, gen_data, step=None, title=None):
    if title is None:
        title = f"Real vs Generated (step {step})" if step is not None else "Real vs Generated"
        
    plt.figure(figsize=(8, 8))
    # real_data and gen_data should be numpy arrays
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.cpu().numpy()
    if isinstance(gen_data, torch.Tensor):
        gen_data = gen_data.cpu().numpy()
        
    plt.scatter(real_data[:, 0], real_data[:, 1], s=5, alpha=0.5, label="Real Data", c="blue")
    plt.scatter(gen_data[:, 0], gen_data[:, 1], s=5, alpha=0.5, label="Generated Data", c="orange")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.show()
