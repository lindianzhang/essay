import numpy as np
from scipy.spatial import cKDTree


def generate_pseudo_anomaly(points, anomaly_type, rng, config):
    """Deform a local surface on an already downsampled normal point cloud."""
    if anomaly_type not in ('bump', 'dent', 'scratch'):
        raise ValueError(f'Unknown pseudo anomaly type: {anomaly_type}')
    original = np.asarray(points, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or len(original) < 4:
        raise ValueError('points must have shape [N, 3] with at least four points.')
    config = {} if config is None else config
    tree = cKDTree(original)
    distances, _ = tree.query(original, k=min(2, len(original)))
    spacing = float(np.median(distances[:, 1]))
    diameter = float(np.linalg.norm(np.ptp(original, axis=0)))
    scale = max(spacing, diameter * 1e-6, 1e-8)
    center_index = int(rng.integers(len(original))) if hasattr(rng, 'integers') else int(rng.randint(len(original)))
    location = original[center_index]
    local_idx = tree.query(location, k=min(32, len(original)))[1]
    local = original[np.atleast_1d(local_idx)]
    _, eigenvectors = np.linalg.eigh(np.cov((local - local.mean(axis=0)).T))
    normal = eigenvectors[:, 0]
    if np.dot(normal, location - original.mean(axis=0)) < 0:
        normal = -normal
    tangent = eigenvectors[:, 2]
    tangent_other = np.cross(normal, tangent)
    angle = float(rng.uniform(0, 2 * np.pi))
    direction = np.cos(angle) * tangent + np.sin(angle) * tangent_other
    across = np.cross(normal, direction)
    scale_ranges = {
        'micro': {'area': (0.004, 0.018), 'length': (0.015, 0.05),
                  'strength': (0.05, 0.18), 'spacing': 1.5},
        'small': {'area': (0.02, 0.05), 'length': (0.05, 0.12),
                  'strength': (0.12, 0.35), 'spacing': 2.0},
        'obvious': {'area': (0.05, 0.12), 'length': (0.12, 0.25),
                    'strength': (0.20, 0.55), 'spacing': 3.0},
    }
    probabilities = np.asarray(config.get('scale_probabilities', (0.7, 0.2, 0.1)), dtype=float)
    if probabilities.shape != (3,) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError('scale_probabilities must contain three nonnegative values summing to one.')
    scale_level = config.get('scale_level')
    if scale_level is None:
        scale_level = rng.choice(tuple(scale_ranges), p=probabilities)
    if scale_level not in scale_ranges:
        raise ValueError(f'Unknown pseudo anomaly scale: {scale_level}')
    ranges = scale_ranges[scale_level]
    radius = max(scale * ranges['spacing'],
                 diameter * float(rng.uniform(*config.get('area_range', ranges['area']))))
    length = max(radius * 2,
                 diameter * float(rng.uniform(*config.get('length_range', ranges['length']))))
    width = max(scale * (0.6 if scale_level == 'micro' else 1.0),
                radius * float(rng.uniform(0.3, 0.7)))
    strength = max(scale * (0.3 if scale_level == 'micro' else 0.6),
                   radius * float(rng.uniform(*config.get('strength_range', ranges['strength']))))
    offset = original - location
    along = offset @ direction
    sideways = offset @ across
    if anomaly_type == 'scratch':
        support = (np.abs(along) < length / 2) & (np.abs(sideways) < width)
        profile = np.exp(-0.5 * (sideways / width) ** 2) * np.maximum(0, 1 - (2 * along / length) ** 2)
        displacement = -strength * profile * support
    else:
        radial = np.linalg.norm(offset - np.outer(offset @ normal, normal), axis=1)
        support = radial < radius
        profile = np.maximum(0, 1 - (radial / radius) ** 2) ** 2
        displacement = (1 if anomaly_type == 'bump' else -1) * strength * profile * support
    mask = np.abs(displacement) > 0
    changed = original + displacement[:, None] * normal
    metadata = {'type': anomaly_type, 'scale_level': scale_level, 'location': location.tolist(),
                'scale': {'radius': float(radius), 'length': float(length), 'width': float(width),
                          'spacing': spacing}, 'strength': float(strength),
                'direction': direction.tolist()}
    return changed.astype(np.asarray(points).dtype, copy=False), mask.astype(np.uint8), metadata
