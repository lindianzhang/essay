import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from data.pseudo_anomaly import (
    estimate_nature3d_normals, generate_nature3d_pseudo_anomaly,
    generate_pseudo_anomaly,
)
from data.real3d import get_real_loader, real3d_classes
from feature_extractors.FPFH import FPFHFeatures
from feature_extractors.geometry_features import build_group_labels
from feature_extractors.geometry_mlp import ConcatFusionMLP, GeometryFusionMLP


_FEATURE_KEYS = ('geometry', 'position', 'curvature')
_EXTRACTION_KEYS = ('num_group', 'group_size', 'max_nn', 'use_MSND',
                    'num_MSND', 'msnd_impl', 'curvature_k')
_CACHE_VERSION = 1
_MULTISCALE_GENERATOR_VERSION = 1
_MULTISCALE_PART_NUMS = (384, 192, 96, 48)
_SOURCE_FINGERPRINT_CACHE = {}


def _source_fingerprint(kind):
    sources = {
        'pseudo': ('data/pseudo_anomaly.py',),
        'feature': ('feature_extractors/FPFH.py',
                    'feature_extractors/geometry_features.py', 'data/real3d.py'),
    }
    if kind not in sources:
        raise ValueError(f'Unknown source fingerprint kind: {kind}')
    if kind not in _SOURCE_FINGERPRINT_CACHE:
        root = Path(__file__).resolve().parent
        _SOURCE_FINGERPRINT_CACHE[kind] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources[kind]
        }
    return _SOURCE_FINGERPRINT_CACHE[kind]


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


def _make_cache_key(points, sample_id, category, split, anomaly_index, generator,
                    generator_config, seed, feature_config=None, feature_points=None,
                    center_idx=None):
    def array_digest(values):
        array = np.ascontiguousarray(values)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        return {'sha256': digest, 'shape': list(array.shape), 'dtype': str(array.dtype)}

    payload = {
        'version': _CACHE_VERSION,
        'source_path': str(sample_id),
        'source_points': array_digest(points),
        'category': category,
        'split': split,
        'anomaly_index': anomaly_index,
        'generator': generator,
        'generator_config': generator_config,
        'seed': int(seed),
        'source_fingerprint': _source_fingerprint(
            'feature' if feature_config is not None else 'pseudo'
        ),
    }
    if feature_config is not None:
        payload['feature_config'] = feature_config
        payload['feature_points'] = array_digest(
            points if feature_points is None else feature_points
        )
        payload['fixed_center_idx'] = (
            None if center_idx is None else array_digest(
                torch.as_tensor(center_idx).detach().cpu().numpy()
            )
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _load_or_generate_pseudo(points, sample_id, anomaly_index, seed, normals,
                             cache_path, generator='nature3d', generator_config=None,
                             cache_status=None):
    cache_path = Path(cache_path)
    generator_config = {} if generator_config is None else generator_config
    serialized_config = json.dumps(generator_config, sort_keys=True,
                                   separators=(',', ':'))
    if cache_status is not None:
        cache_status['hit'] = False
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if (str(cached['cache_key'].item()) != cache_path.stem
                        or int(cached['cache_version'].item()) != _CACHE_VERSION
                        or str(cached['sample_id'].item()) != str(sample_id)
                        or int(cached['anomaly_index'].item()) != anomaly_index
                        or int(cached['seed'].item()) != seed
                        or str(cached['generator'].item()) != generator
                        or str(cached['generator_config'].item()) != serialized_config):
                    raise ValueError('Pseudo cache identity differs from the request.')
                changed = cached['changed_points'].copy()
                point_mask = cached['point_mask'].copy()
                metadata = json.loads(str(cached['metadata'].item()))
                if (changed.shape != points.shape or point_mask.shape != (len(points),)
                        or changed.dtype != np.asarray(points).dtype
                        or not np.isfinite(changed).all()
                        or not np.isin(point_mask, (0, 1)).all()
                        or not isinstance(metadata, dict)):
                    raise ValueError('Pseudo cache content has invalid dimensions or values.')
            if cache_status is not None:
                cache_status['hit'] = True
            return changed, point_mask, metadata
        except Exception:
            pass  # Regenerate only this missing or damaged cache entry.

    item_rng = np.random.default_rng(seed)
    if generator in ('nature3d', 'nature3d_multiscale'):
        if normals is None:
            normals = {}
        if 'values' not in normals:
            normals['values'] = estimate_nature3d_normals(points)
        changed, point_mask, metadata = generate_nature3d_pseudo_anomaly(
            points, item_rng, config=generator_config, normals=normals['values']
        )
        if generator == 'nature3d_multiscale':
            metadata = {**metadata, 'part_num': int(generator_config['part_num']),
                        'generator_version': _MULTISCALE_GENERATOR_VERSION}
    elif generator == 'custom':
        anomaly_type = generator_config['anomaly_type']
        changed, point_mask, metadata = generate_pseudo_anomaly(
            points, anomaly_type, item_rng, config=generator_config.get('parameters', {})
        )
    else:
        raise ValueError(f'Unknown pseudo generator: {generator}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz',
                                         dir=cache_path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary, changed_points=changed, point_mask=point_mask,
                metadata=json.dumps(metadata, ensure_ascii=False, default=str),
                cache_key=cache_path.stem, cache_version=_CACHE_VERSION,
                sample_id=str(sample_id), anomaly_index=anomaly_index,
                seed=seed, generator=generator, generator_config=serialized_config
            )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return changed, point_mask, metadata


def _load_or_extract_group(pc, extractor, cache_key, center_idx=None, cache_status=None):
    args = extractor.args
    cache_path = Path(args.pseudo_cache_dir) / 'features' / f'{cache_key}.pt'
    feature_config = {key: getattr(args, key) for key in _EXTRACTION_KEYS}
    feature_config['use_LFSA'] = args.use_LFSA
    expected_points = pc.squeeze(0).detach().cpu().to(torch.float32)
    group_count = args.num_group
    point_count = expected_points.shape[0]
    required = ('features', 'positions', 'curvature', 'center_idx',
                'knn_idx', 'points', 'center')
    if cache_status is not None:
        cache_status['hit'] = False
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location='cpu')
            group = cached['group']
            if (cached['cache_key'] != cache_key
                    or cached['cache_version'] != _CACHE_VERSION
                    or cached['feature_config'] != feature_config
                    or set(group) != set(required)
                    or not all(isinstance(group[key], torch.Tensor)
                               and group[key].device.type == 'cpu' for key in required)
                    or group['features'].ndim != 2
                    or group['features'].shape[0] != group_count
                    or group['features'].shape[1] == 0
                    or group['positions'].shape != (group_count, 3)
                    or group['curvature'].shape != (group_count, 1)
                    or group['center_idx'].shape != (group_count,)
                    or group['knn_idx'].shape != (group_count, args.group_size)
                    or group['points'].shape != (point_count, 3)
                    or group['center'].shape != (1, group_count, 3)
                    or not torch.equal(group['points'], expected_points)
                    or torch.any(group['center_idx'] < 0)
                    or torch.any(group['center_idx'] >= point_count)
                    or torch.any(group['knn_idx'] < 0)
                    or torch.any(group['knn_idx'] >= point_count)
                    or not torch.equal(
                        group['center'].squeeze(0),
                        group['points'][group['center_idx'].long()]
                    )
                    or (center_idx is not None and not torch.equal(
                        group['center_idx'].long(), center_idx.detach().cpu().long()
                    ))):
                raise ValueError('Feature cache content or configuration differs from the request.')
            if cache_status is not None:
                cache_status['hit'] = True
            return group
        except Exception:
            pass  # Regenerate only this missing or damaged cache entry.

    group = _extract(extractor, pc, center_idx=center_idx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt',
                                         dir=cache_path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save({'cache_key': cache_key, 'cache_version': _CACHE_VERSION,
                        'feature_config': feature_config, 'group': group}, temporary)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return group


def _collect_examples(args, extractor, samples, normals, statistics, rng, count,
                      sample_ids, split, progress_desc):
    if len(samples) != len(normals) or len(samples) != len(sample_ids):
        raise ValueError('samples, normals, and sample_ids must correspond one-to-one.')
    normal_features_by_branch = [[] for _ in _FEATURE_KEYS]
    pseudo_features_by_branch = [[] for _ in _FEATURE_KEYS]
    labels, weights = [], []
    cache_root = Path(args.pseudo_cache_dir)
    feature_config = {key: getattr(args, key) for key in _EXTRACTION_KEYS}
    feature_config['use_LFSA'] = args.use_LFSA
    cache_hit = generated = 0
    with tqdm(total=len(samples) * count, desc=progress_desc,
              dynamic_ncols=True, unit='anomaly') as progress:
        for sample_number, (pc, normal, sample_id) in enumerate(
                zip(samples, normals, sample_ids), start=1):
            normal_features = _normalized(normal, statistics)
            points = pc.squeeze(0).numpy()
            category = Path(sample_id).parent.parent.name
            normal_cache = {}
            for index in range(count):
                seed_input = json.dumps(
                    [str(sample_id), index, int(args.geom_seed)],
                    ensure_ascii=False, separators=(',', ':')
                ).encode('utf-8')
                item_seed = int.from_bytes(
                    hashlib.sha256(seed_input).digest()[:8], 'big'
                )
                sampling_seed = int.from_bytes(
                    hashlib.sha256(seed_input + b':sampling').digest()[:8], 'big'
                )
                sampling_rng = np.random.default_rng(sampling_seed)
                if args.pseudo_generator == 'custom':
                    generator_config = {
                        'anomaly_type': ('bump', 'dent', 'scratch')[index % 3],
                        'parameters': {},
                    }
                elif args.pseudo_generator == 'nature3d':
                    generator_config = {
                        'part_num': 96, 'distance_range': [0.06, 0.08]
                    }
                elif args.pseudo_generator == 'nature3d_multiscale':
                    scale_seed = int.from_bytes(
                        hashlib.sha256(seed_input + b':scale').digest()[:8], 'big'
                    )
                    part_num = int(np.random.default_rng(scale_seed).choice(
                        _MULTISCALE_PART_NUMS, p=(0.25, 0.25, 0.25, 0.25)
                    ))
                    generator_config = {
                        'part_num': part_num, 'distance_range': [0.06, 0.08],
                        'multiscale_version': _MULTISCALE_GENERATOR_VERSION,
                    }
                else:
                    raise ValueError(f'Unknown pseudo generator: {args.pseudo_generator}')
                pseudo_key = _make_cache_key(
                    points, sample_id, category, split, index,
                    args.pseudo_generator, generator_config, item_seed
                )
                pseudo_path = cache_root / 'pseudo' / f'{pseudo_key}.npz'
                progress.set_postfix(
                    category=category, sample=f'{sample_number}/{len(samples)}',
                    cache_hit=cache_hit, generated=generated, refresh=True
                )
                progress.refresh()
                pseudo_status = {}
                changed, point_mask, _ = _load_or_generate_pseudo(
                    points, sample_id, index, item_seed, normal_cache, pseudo_path,
                    generator=args.pseudo_generator,
                    generator_config=generator_config, cache_status=pseudo_status
                )
                group_key = _make_cache_key(
                    points, sample_id, category, split, index,
                    args.pseudo_generator, generator_config, item_seed,
                    feature_config=feature_config, feature_points=changed,
                    center_idx=normal['center_idx']
                )
                progress.refresh()  # Show the current item before feature extraction.
                feature_status = {}
                pseudo = _load_or_extract_group(
                    torch.from_numpy(changed).unsqueeze(0), extractor, group_key,
                    center_idx=normal['center_idx'], cache_status=feature_status
                )
                if pseudo_status['hit'] and feature_status['hit']:
                    cache_hit += 1
                else:
                    generated += 1
                progress.set_postfix(
                    category=category, sample=f'{sample_number}/{len(samples)}',
                    cache_hit=cache_hit, generated=generated, refresh=True
                )
                progress.update(1)
                group_labels, ratios = build_group_labels(point_mask, pseudo['knn_idx'])
                positive = torch.nonzero(group_labels > 0, as_tuple=True)[0].cpu()
                negative = torch.nonzero(group_labels == 0, as_tuple=True)[0].cpu()
                pair_count = min(positive.numel(), negative.numel())
                if pair_count == 0:
                    continue
                positive = torch.from_numpy(sampling_rng.choice(
                    positive.numpy(), size=pair_count, replace=False
                ))
                negative = torch.from_numpy(sampling_rng.choice(
                    negative.numpy(), size=pair_count, replace=False
                ))
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
    if args.pseudo_generator == 'custom' and args.pseudo_per_sample < 3:
        raise ValueError('--pseudo_per_sample must be at least 3 to include every anomaly type.')
    if args.pseudo_cache_dir is None:
        args.pseudo_cache_dir = str(
            Path('/root/autodl-fs/real3d_pseudo_cache') if os.name == 'posix'
            else Path(__file__).resolve().parent / 'pseudo_cache'
        )
    torch.manual_seed(args.geom_seed)
    np.random.seed(args.geom_seed)
    rng = np.random.default_rng(args.geom_seed)
    extractor = FPFHFeatures(args=args)
    torch.manual_seed(args.geom_seed)
    train_samples, validation_samples = [], []
    train_ids, validation_ids = [], []
    train_cache_keys, validation_cache_keys = [], []
    train_normals, validation_normals = [], []
    normal_groups_by_category = {}
    normal_samples_by_category = {}
    categories = (
        [args.geom_category]
        if args.geom_category is not None
        else real3d_classes()
    )
    if not set(categories).issubset(set(real3d_classes())):
        raise ValueError(f'Invalid category: {args.geom_category}')

    for category in categories:
        samples = [
            (pc, str(path[0] if isinstance(path, (tuple, list)) else path))
            for pc, _, label, path in _loader(args, 'train', category)
            if int(torch.as_tensor(label).reshape(-1)[0]) == 0
        ]
        if len(samples) != 4:
            raise ValueError(f'{category} requires four normal training samples; found {len(samples)}.')
        normal_samples_by_category[category] = samples
    feature_config = {key: getattr(args, key) for key in _EXTRACTION_KEYS}
    feature_config['use_LFSA'] = args.use_LFSA
    normal_count = sum(len(samples) for samples in normal_samples_by_category.values())
    with tqdm(total=normal_count, desc='Normal feature extraction',
              dynamic_ncols=True, unit='sample') as normal_progress:
        for category, samples in normal_samples_by_category.items():
            held_out = int(rng.integers(4))
            groups = []
            for index, (pc, sample_id) in enumerate(samples):
                split = 'validation' if index == held_out else 'train'
                points = pc.squeeze(0).numpy()
                if args.pseudo_generator == 'nature3d':
                    generator_config = {'part_num': 96, 'distance_range': [0.06, 0.08]}
                elif args.pseudo_generator == 'nature3d_multiscale':
                    generator_config = {
                        'part_num_choices': list(_MULTISCALE_PART_NUMS),
                        'part_num_probabilities': [0.25, 0.25, 0.25, 0.25],
                        'distance_range': [0.06, 0.08],
                        'multiscale_version': _MULTISCALE_GENERATOR_VERSION,
                    }
                else:
                    generator_config = {
                        'anomaly_types': ['bump', 'dent', 'scratch'],
                        'scale_probabilities': [0.7, 0.2, 0.1],
                    }
                group_key = _make_cache_key(
                    points, sample_id, category, split, None,
                    args.pseudo_generator, generator_config, args.geom_seed,
                    feature_config=feature_config
                )
                normal_progress.set_postfix(
                    category=category, sample=f'{index + 1}/{len(samples)}',
                    refresh=True
                )
                normal_progress.refresh()
                group = _load_or_extract_group(pc, extractor, group_key)
                groups.append(group)
                normal_progress.update(1)
                if index == held_out:
                    validation_samples.append(pc)
                    validation_normals.append(group)
                    validation_ids.append(sample_id)
                    validation_cache_keys.append(group_key)
                else:
                    train_samples.append(pc)
                    train_normals.append(group)
                    train_ids.append(sample_id)
                    train_cache_keys.append(group_key)
            normal_groups_by_category[category] = groups
    statistics = _normalization(train_normals)
    train_data = _collect_examples(args, extractor, train_samples, train_normals,
                                   statistics, rng, args.pseudo_per_sample,
                                   train_ids, 'train', 'Training pseudo anomalies')
    validation_rng = np.random.default_rng(args.geom_seed + 1)
    validation_data = _collect_examples(args, extractor, validation_samples,
                                        validation_normals, statistics, validation_rng, 3,
                                        validation_ids, 'validation',
                                        'Validation pseudo anomalies')
    train_loader = DataLoader(train_data, batch_size=args.geom_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.geom_batch_size)
    feature_dim = train_normals[0]['features'].shape[1]
    model_class = {'gate': GeometryFusionMLP, 'concat': ConcatFusionMLP}.get(args.geom_arch)
    if model_class is None:
        raise ValueError(f'Unknown geometry MLP architecture: {args.geom_arch}')
    model = model_class(feature_dim, use_position=args.geom_use_position,
                        use_curvature=args.geom_use_curvature).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.geom_lr)
    checkpoint_path = Path(args.geom_checkpoint)
    checkpoint_prefix = (checkpoint_path.with_suffix('')
                         if checkpoint_path.suffix.lower() == '.pt' else checkpoint_path)
    best_checkpoint_path = checkpoint_prefix.with_name(f'{checkpoint_prefix.name}_best.pt')
    last_checkpoint_path = checkpoint_prefix.with_name(f'{checkpoint_prefix.name}_last.pt')
    resume_path = checkpoint_prefix.with_name(f'{checkpoint_prefix.name}.resume.pt')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resume_model_config = {
        'architecture': args.geom_arch,
        'feature_dim': feature_dim,
        'hidden_dim': 64,
        'use_position': args.geom_use_position,
        'use_curvature': args.geom_use_curvature,
        'model_source_sha256': hashlib.sha256(
            (Path(__file__).resolve().parent / 'feature_extractors' /
             'geometry_mlp.py').read_bytes()
        ).hexdigest(),
        'state_shapes': {name: tuple(value.shape)
                         for name, value in model.state_dict().items()},
    }
    resume_training_config = {
        'architecture': args.geom_arch,
        'optimizer': 'Adam',
        'batch_size': args.geom_batch_size,
        'lr': args.geom_lr,
        'seed': args.geom_seed,
        'pseudo_per_sample': args.pseudo_per_sample,
        'pseudo_generator': args.pseudo_generator,
        'device': str(args.device),
    }
    resume_data_config = {
        'dataset': args.dataset,
        'trained_categories': categories,
        'feature': args.feature,
        'agg_method': args.agg_method,
        'real3d_format': args.real3d_format,
        'real3d_root': args.real3d_root,
        'train_ids': train_ids,
        'validation_ids': validation_ids,
        'train_cache_keys': train_cache_keys,
        'validation_cache_keys': validation_cache_keys,
        'feature_config': feature_config,
        'pseudo_generator_config': (
            {'part_num': 96, 'distance_range': [0.06, 0.08]}
            if args.pseudo_generator == 'nature3d' else
            {'part_num_choices': list(_MULTISCALE_PART_NUMS),
             'part_num_probabilities': [0.25, 0.25, 0.25, 0.25],
             'distance_range': [0.06, 0.08],
             'multiscale_version': _MULTISCALE_GENERATOR_VERSION}
            if args.pseudo_generator == 'nature3d_multiscale' else
            {'anomaly_types': ['bump', 'dent', 'scratch'],
             'scale_probabilities': [0.7, 0.2, 0.1]}
        ),
        'pseudo_source_fingerprint': _source_fingerprint('pseudo'),
        'feature_source_fingerprint': _source_fingerprint('feature'),
    }
    best_loss = float('inf')
    best_state = None
    last_validation_loss = float('nan')
    start_epoch = 0
    if args.geom_resume and resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu')
        if (not isinstance(resume, dict)
                or resume.get('resume_format') != 'geometry_mlp_training_v1'
                or resume.get('architecture') != args.geom_arch
                or resume.get('model_config') != resume_model_config
                or resume.get('training_config') != resume_training_config
                or resume.get('data_config') != resume_data_config):
            raise ValueError(
                f'Training resume configuration does not match: {resume_path}'
            )
        required = ('epoch', 'model_state', 'optimizer_state', 'best_loss',
                    'best_model_state', 'torch_rng_state', 'cuda_rng_state',
                    'numpy_rng_state', 'split_rng_state', 'validation_rng_state')
        if any(key not in resume for key in required):
            raise ValueError(f'Incomplete training resume checkpoint: {resume_path}')
        start_epoch = int(resume['epoch'])
        if start_epoch < 0 or start_epoch > args.geom_epochs:
            raise ValueError(
                f'Completed epoch {start_epoch} exceeds target --geom_epochs={args.geom_epochs}.'
            )
        model.load_state_dict(resume['model_state'])
        optimizer.load_state_dict(resume['optimizer_state'])
        best_loss = float(resume['best_loss'])
        best_state = copy.deepcopy(resume['best_model_state'])
        last_validation_loss = float(resume.get('last_validation_loss', float('nan')))
        if best_state is None and start_epoch > 0:
            raise ValueError(f'Resume checkpoint has no best model state: {resume_path}')
        saved_cuda = resume['cuda_rng_state']
        if saved_cuda is not None:
            if not torch.cuda.is_available() or len(saved_cuda) != torch.cuda.device_count():
                raise ValueError('CUDA device configuration differs from the training resume checkpoint.')
            torch.cuda.set_rng_state_all(saved_cuda)
        torch.set_rng_state(resume['torch_rng_state'])
        numpy_state = resume['numpy_rng_state']
        np.random.set_state((
            numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2], numpy_state[3], numpy_state[4]
        ))
        rng.bit_generator.state = copy.deepcopy(resume['split_rng_state'])
        validation_rng.bit_generator.state = copy.deepcopy(resume['validation_rng_state'])
        print(f'Resuming geometry MLP after epoch {start_epoch}: {resume_path}')
    for epoch in range(start_epoch, args.geom_epochs):
        training_loss = _epoch(model, train_loader, args.device, optimizer)
        validation_loss = _epoch(model, validation_loader, args.device)
        last_validation_loss = validation_loss
        print(f'Epoch {epoch + 1}: train loss {training_loss:.6f}, validation loss {validation_loss:.6f}')
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        numpy_state = np.random.get_state()
        resume_state = {
            'resume_format': 'geometry_mlp_training_v1',
            'architecture': args.geom_arch,
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': best_loss,
            'best_model_state': best_state,
            'last_validation_loss': last_validation_loss,
            'model_config': resume_model_config,
            'training_config': resume_training_config,
            'data_config': resume_data_config,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else None),
            'numpy_rng_state': (
                numpy_state[0], numpy_state[1].tolist(),
                numpy_state[2], numpy_state[3], numpy_state[4]
            ),
            'split_rng_state': copy.deepcopy(rng.bit_generator.state),
            'validation_rng_state': copy.deepcopy(validation_rng.bit_generator.state),
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pt',
                                             dir=resume_path.parent,
                                             delete=False) as temporary:
                temporary_path = Path(temporary.name)
                torch.save(resume_state, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, resume_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    if best_state is None:
        raise ValueError('--geom_epochs must be positive.')
    last_state = copy.deepcopy(model.state_dict())
    for output_path, state, recorded_loss in (
            (last_checkpoint_path, last_state, last_validation_loss),
            (best_checkpoint_path, best_state, best_loss)):
        model.load_state_dict(state)
        model.eval()
        memory_bank = {
            category: torch.cat([_embedding(model, group, statistics, args.device).cpu()
                                 for group in groups], dim=0)
            for category, groups in normal_groups_by_category.items()
        }
        torch.save({
            'model_state': state,
            'memory_bank': memory_bank,
            'trained_categories': categories,
            'architecture': args.geom_arch,
            'normalization': statistics,
            'feature_dim': feature_dim,
            'hidden_dim': 64,
            'feature_config': {key: getattr(args, key) for key in _EXTRACTION_KEYS},
            'model_config': {'use_position': args.geom_use_position,
                             'use_curvature': args.geom_use_curvature},
            'training_config': {'epochs': args.geom_epochs, 'batch_size': args.geom_batch_size,
                                'lr': args.geom_lr, 'seed': args.geom_seed,
                                'pseudo_per_sample': args.pseudo_per_sample,
                                'pseudo_generator': args.pseudo_generator},
            'validation_loss': recorded_loss,
        }, output_path)
        print(f'Saved geometry MLP checkpoint: {output_path}')


def evaluate_geometry_mlp(args):
    import csv
    from data.real3d import DATASETS_PATH, OFFICIAL_DATASETS_PATH

    checkpoint = torch.load(args.geom_checkpoint, map_location='cpu')
    trained_categories = checkpoint.get('trained_categories')
    if trained_categories is None:
        trained_categories = list(checkpoint['memory_bank'])
    if args.geom_category is not None:
        if (args.geom_category not in trained_categories
                or args.geom_category not in checkpoint['memory_bank']):
            raise ValueError(f'Category {args.geom_category} is not present in the checkpoint.')
        categories = [args.geom_category]
    else:
        categories = list(trained_categories)
    if not categories or any(category not in checkpoint['memory_bank'] for category in categories):
        raise ValueError('Checkpoint has no valid trained categories for evaluation.')
    extraction_args = copy.copy(args)
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
    results = []
    diagnostic_rows = []
    data_root = (args.real3d_root or OFFICIAL_DATASETS_PATH
                 if args.real3d_format == 'official' else DATASETS_PATH)
    for category in categories:
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
            sample_path = Path(path[0] if isinstance(path, (list, tuple)) else path)
            relative_path = sample_path.resolve().relative_to(Path(data_root).resolve())
            is_anomaly = int(torch.as_tensor(label).reshape(-1)[0].item())
            if not is_anomaly:
                anomaly_type = 'good'
            elif args.real3d_format == 'official':
                anomaly_type = sample_path.parent.name
            else:
                anomaly_type = sample_path.stem.split('_', 1)[-1]
            point_scores = np.asarray(extractor.predictions[-1]).reshape(-1)
            top1_count = max(1, int(np.ceil(point_scores.size * 0.01)))
            top5_count = max(1, int(np.ceil(point_scores.size * 0.05)))
            diagnostic_rows.append({
                'sample_relative_path': relative_path.as_posix(),
                'anomaly_type': anomaly_type,
                'label': is_anomaly,
                'image_score': float(np.asarray(extractor.image_preds[-1]).reshape(-1)[0]),
                'point_mean': float(point_scores.mean()),
                'point_max': float(point_scores.max()),
                'point_top1_percent_mean': float(
                    np.partition(point_scores, -top1_count)[-top1_count:].mean()
                ),
                'point_top5_percent_mean': float(
                    np.partition(point_scores, -top5_count)[-top5_count:].mean()
                ),
                'point_p99': float(np.percentile(point_scores, 99)),
                'gt_anomaly_ratio': float(
                    (torch.as_tensor(mask).reshape(-1) > 0).float().mean().item()
                ),
            })
        pixel_labels = np.asarray(extractor.pixel_labels).reshape(-1)
        pixel_scores = np.asarray(extractor.pixel_preds).reshape(-1)
        p_aupr = (average_precision_score(pixel_labels, pixel_scores)
                  if np.unique(pixel_labels).size > 1 else float('nan'))
        extractor.calculate_metrics()
        result = (extractor.image_rocauc, extractor.pixel_rocauc, p_aupr)
        results.append(result)
        print(f'{category}: O-AUROC={result[0]:.3f}, P-AUROC={result[1]:.3f}, P-AUPR={result[2]:.3f}')
    if len(categories) > 1:
        mean = np.nanmean(np.asarray(results), axis=0)
        print(f'Mean: O-AUROC={mean[0]:.3f}, P-AUROC={mean[1]:.3f}, P-AUPR={mean[2]:.3f}')
    csv_path = (Path(__file__).resolve().parent / 'diagnostics' /
                'airplane_concat136_e50' / 'full_test_scores.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=(
            'sample_relative_path', 'anomaly_type', 'label', 'image_score',
            'point_mean', 'point_max', 'point_top1_percent_mean',
            'point_top5_percent_mean', 'point_p99', 'gt_anomaly_ratio'
        ))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    return results
