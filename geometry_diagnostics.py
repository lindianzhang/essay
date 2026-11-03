"""Read-only diagnostics for a trained geometry MLP and its normal memory banks."""

import copy
import csv
import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from data.real3d import real3d_classes
from feature_extractors.FPFH import FPFHFeatures
from feature_extractors.geometry_features import build_group_labels
from feature_extractors.geometry_mlp import ConcatFusionMLP, GeometryFusionMLP
from geometry_mlp_runner import _extract, _loader, _normalized
from patchcore_runner import PatchCore


_BRANCH_NAMES = ('Geometry', 'Position', 'Curvature')
_BRANCH_COLORS = ('#3478a4', '#e39a34', '#72a66a')


def nearest_bank_with_indices(embedding, bank, chunk_size=256):
    """Use the evaluation distance formula and also retain the nearest bank row."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    distances, indices = [], []
    for start in range(0, embedding.shape[0], chunk_size):
        nearest = torch.cdist(embedding[start:start + chunk_size], bank).min(dim=1)
        distances.append(nearest.values)
        indices.append(nearest.indices)
    return torch.cat(distances), torch.cat(indices)


def _sample_path(path):
    return str(path[0] if isinstance(path, (tuple, list)) else path)


def _select_samples(args, category):
    dataset = _loader(args, 'test', category).dataset
    ordered = sorted(range(len(dataset)), key=lambda index: dataset.pcd_paths[index])
    normal = [index for index in ordered if dataset.labels[index] == 0][:args.diag_normal_count]
    anomaly = [index for index in ordered if dataset.labels[index] != 0][:args.diag_anomaly_count]
    selected = normal + anomaly
    loader = DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False,
                        num_workers=0, drop_last=False)
    return [(pc, mask, label, _sample_path(path)) for pc, mask, label, path in loader]


def _cache_path(cache_dir, category, sample_path):
    digest = hashlib.sha256(sample_path.encode('utf-8')).hexdigest()[:16]
    return cache_dir / f'{category}_{Path(sample_path).stem}_{digest}.npz'


def _baseline_args(args):
    baseline_args = copy.copy(args)
    baseline_args.dataset = 'real'
    baseline_args.feature = 'FPFH'
    baseline_args.agg_method = 'mean'
    baseline_args.num_group = 4096
    baseline_args.group_size = 128
    baseline_args.max_nn = 40
    baseline_args.use_MSND = True
    baseline_args.num_MSND = 2
    baseline_args.msnd_impl = 'official'
    baseline_args.use_LFSA = True
    baseline_args.vis_save = False
    return baseline_args


def _baseline_scores(args, category, samples, cache_dir):
    """Read point scores first; fit the original four-sample baseline only if needed."""
    cached, pending = {}, []
    for pc, mask, label, sample_path in samples:
        cache_path = _cache_path(cache_dir, category, sample_path)
        points = pc.squeeze(0).numpy()
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                if (all(key in data for key in ('sample_path', 'points', 'point_scores'))
                        and str(data['sample_path']) == sample_path
                        and np.array_equal(data['points'], points)
                        and data['point_scores'].size == len(points)):
                    cached[sample_path] = data['point_scores'].reshape(-1).copy()
                    continue
        pending.append((pc, mask, label, sample_path, cache_path))
    if pending:
        baseline_args = _baseline_args(args)
        if len(_loader(baseline_args, 'train', category)) != 4:
            raise ValueError(f'Simple3D baseline requires four normal training samples for {category}.')
        baseline = PatchCore(args=baseline_args)
        baseline.fit(category)
        baseline.method.init_para()
        for pc, mask, label, sample_path, cache_path in pending:
            with torch.no_grad():
                baseline.method.predict(pc, mask, label, [sample_path])
            points = pc.squeeze(0).numpy()
            scores = np.asarray(baseline.method.predictions[-1]).reshape(-1)
            if scores.size != len(points):
                raise ValueError(f'Baseline point score count differs from points: {sample_path}')
            cached[sample_path] = scores.copy()
            np.savez_compressed(cache_path, sample_path=sample_path,
                                points=points, point_scores=scores)
    return cached


def _display_values(values):
    values = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(values, (5, 99))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def save_anomaly_comparison(record, output_dir, seed=42):
    """Save full point scores as UTF-8 text and an aligned four-panel image."""
    points = record['points']
    gt = np.asarray(record['gt']).reshape(-1)
    geometry_scores = np.asarray(record['geometry_scores']).reshape(-1)
    baseline_data = record.get('baseline_scores')
    baseline_scores = (np.asarray(baseline_data, dtype=np.float64).reshape(-1)
                       if baseline_data is not None else np.full(len(points), np.nan))
    if gt.size != len(points) or geometry_scores.size != len(points):
        raise ValueError('Points, GT, and Geometry MLP scores must have identical lengths.')
    if baseline_scores.size != len(points):
        baseline_scores = np.full(len(points), np.nan)
    sample_path = record['sample']
    stem = _cache_path(output_dir, record['category'], sample_path).stem
    rng = np.random.default_rng(seed)
    display_indices = (np.sort(rng.choice(len(points), size=10000, replace=False))
                       if len(points) > 10000 else np.arange(len(points)))

    sample_name = Path(sample_path).name
    label = record.get('label', int(np.any(gt > 0)))
    defect_type = ('good' if label == 0 else
                   Path(sample_path).parent.name
                   if Path(sample_path).parent.name != 'test' else
                   Path(sample_path).stem.split('_', 1)[-1])
    method_name = ('Concat-MLP' if record.get('architecture') == 'concat'
                   else 'Geometry MLP')
    gt_valid = bool(np.isfinite(gt).all() and np.isin(gt, (0, 1)).all())
    anomaly_region = gt > 0

    def number(value):
        return format(float(value), '.17g') if np.isfinite(value) else 'N/A'

    def summary(scores, selected):
        values = np.asarray(scores, dtype=np.float64)[selected]
        finite_count = int(np.isfinite(values).sum())
        if values.size == 0 or finite_count != values.size:
            return [values.size, finite_count] + ['N/A'] * 6
        top1 = max(1, int(np.ceil(values.size * 0.01)))
        top5 = max(1, int(np.ceil(values.size * 0.05)))
        return [
            values.size, finite_count, number(values.mean()), number(values.max()),
            number(np.median(values)), number(np.percentile(values, 99)),
            number(np.partition(values, values.size - top1)[-top1:].mean()),
            number(np.partition(values, values.size - top5)[-top5:].mean()),
        ]

    with (output_dir / f'{stem}.txt').open('w', encoding='utf-8', newline='') as output:
        output.write(f'Sample Name: {sample_name}\n')
        output.write(f'Sample Path: {sample_path}\n')
        output.write(f'Category: {record["category"]}\n')
        output.write(f'Defect Type: {defect_type}\n')
        output.write(f'Label: {"anomaly (1)" if label else "normal (0)"}\n')
        output.write(f'Total Points: {len(points)}\n')
        output.write(f'GT Anomaly Points: {int(anomaly_region.sum()) if gt_valid else "N/A"}\n')
        output.write(f'GT Anomaly Ratio: {number(anomaly_region.mean()) if gt_valid else "N/A"}\n')
        output.write('\nScore Statistics (raw scores; incomplete regions are N/A)\n')
        writer = csv.writer(output, delimiter='\t', lineterminator='\n')
        writer.writerow(('Method', 'Region', 'Point Count', 'Finite Score Count',
                         'Mean', 'Max', 'Median', 'P99', 'Top-1% Mean', 'Top-5% Mean'))
        methods = ((method_name, geometry_scores), ('Simple3D', baseline_scores))
        for name, scores in methods:
            writer.writerow((name, 'All Points', *summary(scores, np.ones(len(points), dtype=bool))))
            for region_name, selected in (('GT Normal', ~anomaly_region),
                                          ('GT Anomaly', anomaly_region)):
                if gt_valid:
                    writer.writerow((name, region_name, *summary(scores, selected)))
                else:
                    writer.writerow((name, region_name, 'N/A', 'N/A', *(['N/A'] * 6)))
        output.write('\nSingle-Sample Point Metrics\n')
        writer.writerow(('Method', 'P-AUROC', 'P-AUPR'))
        for name, scores in methods:
            if (gt_valid and anomaly_region.any() and (~anomaly_region).any()
                    and np.isfinite(scores).all()):
                writer.writerow((name, number(roc_auc_score(anomaly_region, scores)),
                                 number(average_precision_score(anomaly_region, scores))))
            else:
                writer.writerow((name, 'N/A', 'N/A'))
        output.write('\nPer-Point Data (original point order)\n')
        writer.writerow(('Point Index', 'X', 'Y', 'Z', 'GT Label',
                         f'{method_name} Score', 'Simple3D Score'))
        for index, (point, gt_label, geometry_score, baseline_score) in enumerate(
                zip(points, gt, geometry_scores, baseline_scores)):
            writer.writerow((index, *(number(coordinate) for coordinate in point),
                             number(gt_label), number(geometry_score),
                             number(baseline_score)))

    shown = points[display_indices]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    half_range = max(float((upper - lower).max()) / 2, 1e-6)
    fig = plt.figure(figsize=(20, 5.8))
    panels = (
        ('Original Point Cloud', np.full(len(display_indices), 0.45), 'Greys', False),
        ('Ground Truth', gt[display_indices], 'coolwarm', False),
        ('Simple3D', _display_values(baseline_scores[display_indices])
         if np.isfinite(baseline_scores).all() else None, 'inferno', True),
        ('Geometry MLP', _display_values(geometry_scores[display_indices]), 'inferno', True),
    )
    for index, (title, colors, cmap, is_score) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, index, projection='3d')
        if colors is None:
            ax.text2D(0.2, 0.5, 'Baseline comparison disabled', transform=ax.transAxes)
        else:
            ax.scatter(shown[:, 0], shown[:, 1], shown[:, 2], c=colors,
                       cmap=cmap, norm=Normalize(0, 1), s=0.5, linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set(xlim=(center[0] - half_range, center[0] + half_range),
               ylim=(center[1] - half_range, center[1] + half_range),
               zlim=(center[2] - half_range, center[2] + half_range))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if is_score and colors is not None:
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                         ax=ax, shrink=0.55, pad=0.02)
    fig.suptitle(f"{record['category'].title()} | {Path(sample_path).name}")
    fig.text(0.5, 0.02,
             'Score colors use separate 5th-99th percentile scaling for display only; raw scores are in the TXT.',
             ha='center', fontsize=9)
    fig.savefig(output_dir / f'{stem}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_gate_weight_analysis(records, output_dir):
    """Save actual gate coefficients and their category and GT-region distributions."""
    rows = []
    for record in records:
        for group_index, (ratio, weights) in enumerate(zip(record['ratios'], record['weights'])):
            rows.append((record['category'], record['sample'], group_index, ratio,
                         *weights.tolist()))
    with (output_dir / 'gate_weights.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('category', 'sample', 'group_index', 'gt_anomaly_ratio',
                         'geometry_weight', 'position_weight', 'curvature_weight'))
        writer.writerows(rows)

    categories = list(dict.fromkeys(record['category'] for record in records))
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(categories)), 5))
    for branch, (name, color) in enumerate(zip(_BRANCH_NAMES, _BRANCH_COLORS)):
        series = [np.concatenate([record['weights'][:, branch] for record in records
                                  if record['category'] == category]) for category in categories]
        positions = np.arange(len(categories)) * 4 + branch
        boxes = ax.boxplot(series, positions=positions, widths=0.7, patch_artist=True,
                           showfliers=False)
        for patch in boxes['boxes']:
            patch.set_facecolor(color)
        ax.plot([], [], color=color, linewidth=8, label=name)
    ax.set_xticks(np.arange(len(categories)) * 4 + 1, [category.title() for category in categories])
    ax.set_ylabel('Gate weight')
    ax.set_title('Gate weight distributions by category')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'category_gate_weights.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for branch, (name, color) in enumerate(zip(_BRANCH_NAMES, _BRANCH_COLORS)):
        for region, offset in ((False, 0), (True, 1)):
            values = np.concatenate([record['weights'][record['ratios'] > 0 if region
                                                         else record['ratios'] == 0, branch]
                                     for record in records])
            if values.size:
                boxes = ax.boxplot([values], positions=[branch * 3 + offset], widths=0.7,
                                   patch_artist=True, showfliers=False)
                boxes['boxes'][0].set_facecolor(color)
                boxes['boxes'][0].set_alpha(0.55 if region else 1.0)
        ax.plot([], [], color=color, linewidth=8, label=name)
    ax.set_xticks([0.5, 3.5, 6.5], _BRANCH_NAMES)
    ax.set_ylabel('Gate weight')
    ax.set_title('Gate weights in GT-normal and GT-anomaly candidate groups')
    ax.text(0.02, 0.98, 'Opaque: no GT anomaly points\nTranslucent: contains GT anomaly points',
            transform=ax.transAxes, va='top')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_dir / 'region_gate_weights.png', dpi=180)
    plt.close(fig)


def save_distance_analysis(records, output_dir):
    """Save exact nearest-neighbor distances and compare GT-defined regions."""
    with (output_dir / 'nearest_distances.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('category', 'sample', 'group_index', 'gt_anomaly_ratio',
                         'nearest_distance', 'nearest_bank_index'))
        for record in records:
            for group_index, (ratio, distance, bank_index) in enumerate(zip(
                    record['ratios'], record['distances'], record['nearest_indices'])):
                writer.writerow((record['category'], record['sample'], group_index,
                                 float(ratio), float(distance), int(bank_index)))

    all_ratios = np.concatenate([record['ratios'] for record in records])
    all_distances = np.concatenate([record['distances'] for record in records])
    bins = np.histogram_bin_edges(all_distances, bins=50)
    fig, ax = plt.subplots(figsize=(9, 5))
    for selected, name, color in ((all_ratios == 0, 'GT-normal groups', '#3478a4'),
                                  (all_ratios > 0, 'GT-anomaly candidate groups', '#dc633f')):
        if selected.any():
            ax.hist(all_distances[selected], bins=bins, density=True, alpha=0.5,
                    label=name, color=color)
    ax.set_xlabel('Nearest normal memory bank distance')
    ax.set_ylabel('Density')
    ax.set_title('Nearest distance distributions by GT region')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'region_distance_distribution.png', dpi=180)
    plt.close(fig)

    categories = list(dict.fromkeys(record['category'] for record in records))
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(categories)), 5))
    for category_index, category in enumerate(categories):
        category_distances = np.concatenate([record['distances'] for record in records
                                             if record['category'] == category])
        category_ratios = np.concatenate([record['ratios'] for record in records
                                          if record['category'] == category])
        for offset, selected, color in ((0, category_ratios == 0, '#3478a4'),
                                         (1, category_ratios > 0, '#dc633f')):
            if selected.any():
                boxes = ax.boxplot([category_distances[selected]],
                                   positions=[category_index * 3 + offset], widths=0.7,
                                   patch_artist=True, showfliers=False, whis=(0, 100))
                boxes['boxes'][0].set_facecolor(color)
    ax.set_xticks(np.arange(len(categories)) * 3 + 0.5,
                  [category.title() for category in categories])
    ax.set_ylabel('Nearest normal memory bank distance')
    ax.set_title('Nearest distance by category and GT region')
    ax.plot([], [], color='#3478a4', linewidth=8, label='GT-normal')
    ax.plot([], [], color='#dc633f', linewidth=8, label='GT-anomaly candidate')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'category_distance_comparison.png', dpi=180)
    plt.close(fig)


def run_geometry_diagnostics(args):
    """Create three diagnostic views without changing training or evaluation."""
    if args.diag_normal_count < 0 or args.diag_anomaly_count < 0:
        raise ValueError('Diagnostic sample counts must be nonnegative.')
    categories = list(dict.fromkeys(args.diag_categories))
    unknown = set(categories) - set(real3d_classes())
    if unknown:
        raise ValueError(f'Unknown Real3D-AD diagnostic categories: {sorted(unknown)}')
    if not categories:
        raise ValueError('At least one diagnostic category is required.')
    root = Path(args.diag_dir) if args.diag_dir else (
        Path('/root/autodl-fs/geometry_diagnostics') if os.name == 'posix'
        else Path(__file__).resolve().parent / 'diagnostics')
    directories = {name: root / name for name in
                   ('anomaly_heatmaps', 'gate_weights', 'nearest_distances', 'baseline_cache')}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.geom_checkpoint, map_location='cpu')
    extraction_args = copy.copy(args)
    extraction_args.vis_save = False
    for key, value in checkpoint['feature_config'].items():
        setattr(extraction_args, key, value)
    extractor = FPFHFeatures(args=extraction_args)
    architecture = checkpoint.get('architecture', 'gate')
    model_class = {'gate': GeometryFusionMLP, 'concat': ConcatFusionMLP}.get(architecture)
    if model_class is None:
        raise ValueError(f'Unknown checkpoint architecture: {architecture}')
    model = model_class(checkpoint['feature_dim'], checkpoint['hidden_dim'],
                        **checkpoint['model_config']).to(args.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    statistics = checkpoint['normalization']
    records = []
    for category in categories:
        samples = _select_samples(args, category)
        if not samples:
            continue
        baseline_scores = (_baseline_scores(args, category, samples, directories['baseline_cache'])
                           if args.diag_compare_baseline else {})
        bank = checkpoint['memory_bank'][category].to(args.device)
        extractor.init_para()
        for pc, mask, label, sample_path in samples:
            group = _extract(extractor, pc)
            inputs = (part.to(args.device) for part in _normalized(group, statistics))
            with torch.no_grad():
                embedding, weights = model(*inputs)
                distances, nearest_indices = nearest_bank_with_indices(embedding, bank)
                extractor.compute_anomay_scores(
                    group['features'], mask, label, [sample_path], pc.squeeze(0),
                    group['points'].unsqueeze(0).to(args.device), group['center'].to(args.device),
                    external_patch_scores=distances)
            points = pc.squeeze(0).numpy()
            gt = mask.reshape(-1).cpu().numpy()
            scores = np.asarray(extractor.predictions[-1]).reshape(-1)
            if gt.size != len(points) or scores.size != len(points):
                raise ValueError(f'Point score or GT count differs from points: {sample_path}')
            _, ratios = build_group_labels(mask.reshape(-1), group['knn_idx'])
            record = {
                'category': category, 'sample': sample_path, 'points': points,
                'label': int(torch.as_tensor(label).reshape(-1)[0].item()),
                'architecture': architecture,
                'gt': gt, 'geometry_scores': scores,
                'baseline_scores': baseline_scores.get(sample_path, np.full(len(points), np.nan)),
                'center_idx': group['center_idx'].numpy(),
                'knn_idx': group['knn_idx'].numpy(),
                'group_features': group['features'].numpy(),
                'embedding': embedding.detach().cpu().numpy(),
                'positions': group['positions'].numpy(),
                'curvature': group['curvature'].numpy(),
                'weights': (weights.detach().cpu().numpy()
                            if weights is not None else None),
                'ratios': ratios.cpu().numpy(),
                'distances': distances.detach().cpu().numpy(),
                'nearest_indices': nearest_indices.detach().cpu().numpy(),
            }
            save_anomaly_comparison(record, directories['anomaly_heatmaps'], seed=args.geom_seed)
            records.append(record)
    if not records:
        raise ValueError('No test samples were selected for diagnostics.')
    if architecture == 'gate':
        save_gate_weight_analysis(records, directories['gate_weights'])
    save_distance_analysis(records, directories['nearest_distances'])
    print(f'Geometry diagnostics saved to {root}')
