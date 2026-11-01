import torch


def estimate_patch_curvature(points, knn_idx, k=32):
    """PCA surface variation for the first k neighbors of each group."""
    if points.ndim != 2 or points.shape[1] != 3 or knn_idx.ndim != 2:
        raise ValueError('Expected points [N, 3] and knn_idx [G, K].')
    if k < 1:
        raise ValueError('k must be positive.')
    neighbors = points[knn_idx[:, :min(k, knn_idx.shape[1])].long()]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / max(neighbors.shape[1], 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    curvature = eigenvalues[:, :1] / eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_patch_positions(points, centers):
    if points.ndim != 2 or points.shape[1] != 3 or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError('Expected points [N, 3] and centers [G, 3].')
    origin = points.mean(dim=0, keepdim=True)
    radius = torch.linalg.vector_norm(points - origin, dim=1).max().clamp_min(1e-12)
    return torch.nan_to_num((centers - origin) / radius)


def build_group_labels(point_mask, knn_idx):
    mask = torch.as_tensor(point_mask, device=knn_idx.device).reshape(-1)
    if knn_idx.ndim != 2 or torch.any(knn_idx < 0) or torch.any(knn_idx >= mask.numel()):
        raise ValueError('knn_idx must have shape [G, K] and valid point indices.')
    ratio = (mask[knn_idx.long()] > 0).float().mean(dim=1)
    return (ratio > 0).float(), ratio
