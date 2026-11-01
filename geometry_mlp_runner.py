import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
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
    normal_features_by_branch = [[] for _ in _FEATURE_KEYS]
    pseudo_features_by_branch = [[] for _ in _FEATURE_KEYS]
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
            negative = torch.nonzero(group_labels == 0, as_tuple=True)[0].cpu()
            pair_count = min(positive.numel(), negative.numel())
            if pair_count == 0:
                continue
            positive = torch.from_numpy(rng.choice(positive.numpy(), size=pair_count, replace=False))
            negative = torch.from_numpy(rng.choice(negative.numpy(), size=pair_count, replace=False))
            selected = torch.cat((negative, positive))
            pseudo_features = _normalized(pseudo, statistics)
            for normal_list, pseudo_list, normal_branch, pseudo_branch in zip(
                    normal_features_by_branch, pseudo_features_by_branch,
                    normal_features, pseudo_features):
                normal_list.append(normal_branch[selected])
                pseudo_list.append(pseudo_branch[selected])
            labels.append(torch.cat((torch.zeros(pair_count), torch.ones(pair_count))))
            weights.append(torch.cat((torch.ones(pair_count),
                                      0.25 + 0.75 * ratios[positive].detach().cpu())))
    if not labels:
        raise ValueError('Pseudo anomalies produced no positive groups.')
    return TensorDataset(*(torch.cat(branch) for branch in normal_features_by_branch),
                         *(torch.cat(branch) for branch in pseudo_features_by_branch),
                         torch.cat(labels), torch.cat(weights))


def _epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_weight = 0.0, 0
    for normal_geometry, normal_position, normal_curvature, \
            pseudo_geometry, pseudo_position, pseudo_curvature, labels, weights in loader:
        normal_inputs = (part.to(device) for part in
                         (normal_geometry, normal_position, normal_curvature))
        pseudo_inputs = (part.to(device) for part in
                         (pseudo_geometry, pseudo_position, pseudo_curvature))
        labels, weights = labels.to(device), weights.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            normal_embedding, _ = model(*normal_inputs)
            pseudo_embedding, _ = model(*pseudo_inputs)
            distance = torch.linalg.vector_norm(normal_embedding - pseudo_embedding,
                                                dim=1) / (normal_embedding.shape[1] ** 0.5)
            pair_loss = (1 - labels) * distance.square() + labels * torch.relu(0.5 - distance).square()
            loss = (pair_loss * weights).mean()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_weight += labels.numel()
    return total_loss / total_weight


def _embedding(model, group, statistics, device):
    inputs = (part.to(device) for part in _normalized(group, statistics))
    with torch.no_grad():
        embedding, _ = model(*inputs)
    return embedding


def _nearest_bank_scores(embedding, bank, chunk_size=256):
    return torch.cat([
        torch.cdist(embedding[start:start + chunk_size], bank).min(dim=1).values
        for start in range(0, embedding.shape[0], chunk_size)
    ])


def train_geometry_mlp(args):
    if args.pseudo_per_sample < 3:
        raise ValueError('--pseudo_per_sample must be at least 3 to include every anomaly type.')
    torch.manual_seed(args.geom_seed)
    np.random.seed(args.geom_seed)
    rng = np.random.default_rng(args.geom_seed)
    extractor = FPFHFeatures(args=args)
    torch.manual_seed(args.geom_seed)
    train_samples, validation_samples = [], []
    train_normals, validation_normals = [], []
    normal_groups_by_category = {}
    for category in real3d_classes():
        samples = [pc for pc, _, label, _ in _loader(args, 'train', category)
                   if int(torch.as_tensor(label).reshape(-1)[0]) == 0]
        if len(samples) != 4:
            raise ValueError(f'{category} requires four normal training samples; found {len(samples)}.')
        groups = [_extract(extractor, pc) for pc in samples]
        normal_groups_by_category[category] = groups
        held_out = int(rng.integers(4))
        validation_samples.append(samples[held_out])
        validation_normals.append(groups[held_out])
        train_samples.extend(pc for index, pc in enumerate(samples) if index != held_out)
        train_normals.extend(group for index, group in enumerate(groups) if index != held_out)
    statistics = _normalization(train_normals)
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
    best_state = None
    for epoch in range(args.geom_epochs):
        training_loss = _epoch(model, train_loader, args.device, optimizer)
        validation_loss = _epoch(model, validation_loader, args.device)
        print(f'Epoch {epoch + 1}: train loss {training_loss:.6f}, validation loss {validation_loss:.6f}')
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError('--geom_epochs must be positive.')
    model.load_state_dict(best_state)
    model.eval()
    memory_bank = {
        category: torch.cat([_embedding(model, group, statistics, args.device).cpu()
                             for group in groups], dim=0)
        for category, groups in normal_groups_by_category.items()
    }
    checkpoint_path = Path(args.geom_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': best_state,
        'memory_bank': memory_bank,
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
        bank = checkpoint['memory_bank'][category].to(args.device)
        for pc, mask, label, path in _loader(args, 'test', category):
            group = _extract(extractor, pc)
            embedding = _embedding(model, group, statistics, args.device)
            scores = _nearest_bank_scores(embedding, bank)
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
