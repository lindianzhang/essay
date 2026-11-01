import numpy as np
import open3d as o3d
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


def estimate_nature3d_normals(points):
    """Estimate unit normals and orient them away from the point cloud center."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError('points must have shape [N, 3] with at least one point.')
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=min(30, len(points))))
    normals = np.asarray(cloud.normals).copy()
    radial = points.astype(np.float64) - points.mean(axis=0)
    lengths = np.linalg.norm(normals, axis=1)
    missing = lengths <= 1e-12
    normals[missing] = radial[missing]
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths <= 1e-12] = (0.0, 0.0, 1.0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[np.sum(normals * radial, axis=1) < 0] *= -1
    return normals


def generate_nature3d_pseudo_anomaly(points, rng, config=None, normals=None):
    """Make one spherical, normal directed deformation using Nature3D-AD proportions."""
    source = np.asarray(points)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) == 0:
        raise ValueError('points must have shape [N, 3] with at least one point.')
    config = {} if config is None else config
    part_num = int(config.get('part_num', 96))
    if part_num <= 0:
        raise ValueError('part_num must be positive.')
    distance_range = config.get('distance_range', (0.06, 0.08))
    coord = source.astype(np.float64)
    if normals is None:
        normals = estimate_nature3d_normals(coord)
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != coord.shape:
        raise ValueError('normals must have shape [N, 3].')

    center_index = int(rng.integers(len(coord))) if hasattr(rng, 'integers') else int(rng.randint(len(coord)))
    center = coord[center_index]
    tree = cKDTree(coord)
    neighbor_count = min(len(coord), max(1, len(coord) // part_num))
    distances, _ = tree.query(center, k=neighbor_count)
    radius = float(np.atleast_1d(distances)[-1])
    region_indices = np.asarray(tree.query_ball_point(center, radius), dtype=np.int64)
    region_distances = np.linalg.norm(coord[region_indices] - center, axis=1)
    if region_distances.max() > 0:
        movement_ratio = 1.0 - region_distances / region_distances.max()
    else:
        movement_ratio = np.ones_like(region_distances)

    # Nature3D-AD normalizes the whole cloud to a unit sphere before displacement.
    unit_sphere_radius = float(np.linalg.norm(coord - coord.mean(axis=0), axis=1).max())
    maximum_displacement = float(rng.uniform(*distance_range)) * unit_sphere_radius
    direction = int(rng.choice((-1, 1)))
    movements = direction * maximum_displacement * movement_ratio
    changed = coord.copy()
    changed[region_indices] += normals[region_indices] * movements[:, None]
    mask = np.zeros(len(coord), dtype=np.uint8)
    mask[region_indices] = 1
    changed_points = changed.astype(source.dtype if np.issubdtype(source.dtype, np.floating) else np.float64,
                                    copy=False)
    metadata = {'type': 'nature3d', 'center': center.tolist(),
                'radius': radius,
                'max_displacement': float(np.linalg.norm(changed_points.astype(np.float64) - coord, axis=1).max())}
    return changed_points, mask, metadata
