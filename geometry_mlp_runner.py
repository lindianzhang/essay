import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.pseudo_anomaly import generate_pseudo_anomaly
from data.real3d import get_real_loader, real3d_classes
from feature_extractors.FPFH import FPFHFeatures
from feature_extractors.geometry_features import build_group_labels
from feature_extractors.geometry_mlp import GeometryFusionMLP


_FEATURE_KEYS = ('geometry', 'position', 'curvature')
_EXTRACTION_KEYS = ('num_group', 'group_size', 'max_nn', 'use_MSND',
                    'num_MSND', 'msnd_impl', 'curvature_k')


def _loader(args, split, category):
    return get_real_loader(split, class_name=category,
                           data_format=args.real3d_format, data_root=args.real3d_root)


def _extract(extractor, pc, center_idx=None):
    with torch.no_grad():
        group = extractor.get_fpfh_features(
            pc, return_group_info=True,
            fixed_center_idx=(center_idx.to(extractor.args.device)
                              if center_idx is not None else None))
    return {key: value.detach().cpu() for key, value in group.items()}


def _branches(group):
    return {'geometry': group['features'].float(),
            'position': group['positions'].float(),
            'curvature': group['curvature'].float()}


def _normalization(groups):
    result = {}
    for key in _FEATURE_KEYS:
        values = torch.cat([_branches(group)[key].detach().cpu() for group in groups])
        result[key] = {'mean': values.mean(dim=0),
                       'std': values.std(dim=0, unbiased=False).clamp_min(1e-6)}
    return result


def _normalized(group, statistics):
    branches = _branches(group)
    return [(branches[key].cpu() - statistics[key]['mean']) / statistics[key]['std']
            for key in _FEATURE_KEYS]


def _collect_examples(args, extractor, samples, normals, statistics, rng, count):
    features = [[] for _ in _FEATURE_KEYS]
    labels, weights = [], []
    for pc, normal in zip(samples, normals):
        normal_features = _normalized(normal, statistics)
        for index in range(count):
            anomaly_type = ('bump', 'dent', 'scratch')[index % 3]
            changed, point_mask, _ = generate_pseudo_anomaly(
                pc.squeeze(0).numpy(), anomaly_type, rng, config={})
            pseudo = _extract(extractor, torch.from_numpy(changed).unsqueeze(0),
                              center_idx=normal['center_idx'])
            group_labels, ratios = build_group_labels(point_mask, pseudo['knn_idx'])
            positive = torch.nonzero(group_labels > 0, as_tuple=True)[0].cpu()
            if positive.numel() == 0:
                continue
            negative_count = min(positive.numel(), normal_features[0].shape[0])
            negative = torch.from_numpy(rng.choice(normal_features[0].shape[0],
                                                    size=negative_count, replace=False))
            pseudo_features = _normalized(pseudo, statistics)
            for branch, normal_branch, pseudo_branch in zip(features, normal_features, pseudo_features):
                branch.extend((normal_branch[negative], pseudo_branch[positive]))
            labels.extend((torch.zeros(negative_count), torch.ones(positive.numel())))
            weights.extend((torch.ones(negative_count),
                            0.25 + 0.75 * ratios[positive.to(ratios.device)].detach().cpu()))
    if not labels:
        raise ValueError('Pseudo anomalies produced no positive groups.')
    return TensorDataset(*(torch.cat(branch) for branch in features),
                         torch.cat(labels), torch.cat(weights))


def _epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_weight = 0.0, 0
    for geometry, position, curvature, labels, weights in loader:
        geometry, position, curvature = (part.to(device) for part in
                                          (geometry, position, curvature))
        labels, weights = labels.to(device), weights.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits, _, _ = model(geometry, position, curvature)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction='none') * weights).mean()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_weight += labels.numel()
    return total_loss / total_weight


def train_geometry_mlp(args):
    if args.pseudo_per_sample < 3:
        raise ValueError('--pseudo_per_sample must be at least 3 to include every anomaly type.')
    torch.manual_seed(args.geom_seed)
    np.random.seed(args.geom_seed)
    rng = np.random.default_rng(args.geom_seed)
    extractor = FPFHFeatures(args=args)
    torch.manual_seed(args.geom_seed)
    train_samples, validation_samples = [], []
    for category in real3d_classes():
        samples = [pc for pc, _, label, _ in _loader(args, 'train', category)
                   if int(torch.as_tensor(label).reshape(-1)[0]) == 0]
        if not samples:
            raise ValueError(f'No normal training samples for {category}.')
        if len(samples) > 1:
            held_out = int(rng.integers(len(samples)))
            validation_samples.append(samples.pop(held_out))
        else:
            validation_samples.append(samples[0])
        train_samples.extend(samples)
    train_normals = [_extract(extractor, pc) for pc in train_samples]
    statistics = _normalization(train_normals)
    validation_normals = [_extract(extractor, pc) for pc in validation_samples]
    train_data = _collect_examples(args, extractor, train_samples, train_normals,
                                   statistics, rng, args.pseudo_per_sample)
    validation_rng = np.random.default_rng(args.geom_seed + 1)
    validation_data = _collect_examples(args, extractor, validation_samples,
                                        validation_normals, statistics, validation_rng, 3)
    train_loader = DataLoader(train_data, batch_size=args.geom_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.geom_batch_size)
    feature_dim = train_normals[0]['features'].shape[1]
    model = GeometryFusionMLP(feature_dim, use_position=args.geom_use_position,
                              use_curvature=args.geom_use_curvature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.geom_lr)
    best_loss = float('inf')
    checkpoint_path = Path(args.geom_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.geom_epochs):
        training_loss = _epoch(model, train_loader, args.device, optimizer)
        validation_loss = _epoch(model, validation_loader, args.device)
        print(f'Epoch {epoch + 1}: train loss {training_loss:.6f}, validation loss {validation_loss:.6f}')
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({
                'model_state': model.state_dict(),
                'normalization': statistics,
                'feature_dim': feature_dim,
                'hidden_dim': 64,
                'feature_config': {key: getattr(args, key) for key in _EXTRACTION_KEYS},
                'model_config': {'use_position': args.geom_use_position,
                                 'use_curvature': args.geom_use_curvature},
                'training_config': {'epochs': args.geom_epochs, 'batch_size': args.geom_batch_size,
                                    'lr': args.geom_lr, 'seed': args.geom_seed,
                                    'pseudo_per_sample': args.pseudo_per_sample},
                'validation_loss': best_loss,
            }, checkpoint_path)
    print(f'Saved shared geometry MLP checkpoint: {checkpoint_path}')


def evaluate_geometry_mlp(args):
    checkpoint = torch.load(args.geom_checkpoint, map_location='cpu')
    extraction_args = copy.copy(args)
    for key, value in checkpoint['feature_config'].items():
        setattr(extraction_args, key, value)
    extractor = FPFHFeatures(args=extraction_args)
    model = GeometryFusionMLP(checkpoint['feature_dim'], checkpoint['hidden_dim'],
                              **checkpoint['model_config']).to(args.device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    statistics = checkpoint['normalization']
    results = []
    for category in real3d_classes():
        extractor.init_para()
        for pc, mask, label, path in _loader(args, 'test', category):
            group = _extract(extractor, pc)
            geometry, position, curvature = (part.to(args.device) for part in
                                              _normalized(group, statistics))
            with torch.no_grad():
                logits, _, _ = model(geometry, position, curvature)
                scores = torch.sigmoid(logits)
                extractor.compute_anomay_scores(
                    group['features'], mask, label, path, pc.squeeze(0),
                    group['points'].unsqueeze(0).to(args.device), group['center'].to(args.device),
                    external_patch_scores=scores)
        pixel_labels = np.asarray(extractor.pixel_labels).reshape(-1)
        pixel_scores = np.asarray(extractor.pixel_preds).reshape(-1)
        p_aupr = (average_precision_score(pixel_labels, pixel_scores)
                  if np.unique(pixel_labels).size > 1 else float('nan'))
        extractor.calculate_metrics()
        result = (extractor.image_rocauc, extractor.pixel_rocauc, p_aupr)
        results.append(result)
        print(f'{category}: O-AUROC={result[0]:.3f}, P-AUROC={result[1]:.3f}, P-AUPR={result[2]:.3f}')
    mean = np.nanmean(np.asarray(results), axis=0)
    print(f'Mean: O-AUROC={mean[0]:.3f}, P-AUROC={mean[1]:.3f}, P-AUPR={mean[2]:.3f}')
    return results
