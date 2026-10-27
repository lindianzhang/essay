"""
Autodl server experiment runner with path configurations
针对autodl服务器的实验运行脚本
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
import torch

# Autodl paths
DATASET_BASE = "/root/autodl-fs/Real3D-AD-PCD/Real3D-AD-PCD"
CHECKPOINT_PATH = "/root/autodl-fs/output_real3dad/output_real3dad/snapshots/snapshot.pth.tar"
RESULTS_BASE = "/root/autodl-fs/experiment_results"

# Import experiment configurations
from experiment_configs import ALL_EXPERIMENTS, get_experiment_by_name, print_all_experiments

def update_args_with_experiment(args, exp_config):
    """Update args with experiment configuration"""
    for key, value in exp_config.items():
        if key not in ['name', 'description']:
            setattr(args, key, value)
    return args

def create_patchcore_improved(args):
    """Create PatchCore with improved features"""
    from patchcore_runner import PatchCore

    # Check if any improvement is enabled
    use_improved = (
        getattr(args, 'use_topk_mean', False) or
        getattr(args, 'use_lcn', False) or
        getattr(args, 'use_smooth_aggregation', False)
    )

    if use_improved:
        # Monkey patch the compute_anomay_scores method
        import feature_extractors.features as features_module
        from feature_extractors.features_improved import Features as FeaturesImproved

        # Replace the method
        features_module.Features.compute_anomay_scores = FeaturesImproved.compute_anomay_scores

        print("[Using Improved Features]")

    patchcore = PatchCore(args=args)

    if not use_improved:
        print("[Using Baseline Features]")

    return patchcore

def run_single_experiment(args, exp_config, class_name):
    """Run a single experiment for a given class"""
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print(f"Class: {class_name}")
    print(f"{'='*80}\n")

    # Update args with experiment config
    args_exp = argparse.Namespace(**vars(args))
    args_exp = update_args_with_experiment(args_exp, exp_config)

    # Create PatchCore with improved features
    patchcore = create_patchcore_improved(args_exp)

    # Fit on training data
    print(f"\n[Training] Fitting on class: {class_name}")
    start_time = time.time()
    patchcore.fit(class_name)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.2f}s")

    # Evaluate on test data
    print(f"\n[Testing] Evaluating on class: {class_name}")
    start_time = time.time()
    image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(class_name)
    test_time = time.time() - start_time
    print(f"Testing time: {test_time:.2f}s")

    # Extract results
    method_name = list(image_rocaucs.keys())[0]
    results = {
        'experiment': exp_config['name'],
        'description': exp_config['description'],
        'class_name': class_name,
        'image_rocauc': image_rocaucs[method_name],
        'pixel_rocauc': pixel_rocaucs[method_name],
        'au_pro': au_pros[method_name],
        'train_time': round(train_time, 2),
        'test_time': round(test_time, 2),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add experiment parameters
    for key, value in exp_config.items():
        if key not in ['name', 'description']:
            results[f'param_{key}'] = value

    return results

def get_dataset_classes(dataset_name):
    """Get classes for a given dataset"""
    if dataset_name == 'real3d':
        return [
            "airplane", "candybar", "car", "chicken",
            "diamond", "duck", "fish", "gemstone",
            "seahorse", "shell", "starfish", "toffees"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def run_experiments(args):
    """Main experiment runner"""
    print(f"\n{'#'*80}")
    print(f"# AUTODL EXPERIMENT RUNNER")
    print(f"# Dataset: {DATASET_BASE}")
    print(f"# Results: {RESULTS_BASE}")
    print(f"{'#'*80}\n")

    # Determine which experiments to run
    if args.exp == 'all':
        experiments = ALL_EXPERIMENTS
        print(f"Running ALL {len(experiments)} experiments")
    elif args.exp == 'list':
        print_all_experiments()
        return
    else:
        exp_config = get_experiment_by_name(args.exp)
        if exp_config is None:
            print(f"Error: Experiment '{args.exp}' not found!")
            print("\nAvailable experiments:")
            print_all_experiments()
            return
        experiments = [exp_config]

    # Determine which classes to test
    if args.class_name == 'all':
        classes = get_dataset_classes(args.dataset)
    else:
        classes = [args.class_name]

    print(f"\nDataset: {args.dataset}")
    print(f"Classes: {classes}")
    print(f"Experiments: {len(experiments)}")
    print(f"Total runs: {len(experiments) * len(classes)}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")

    # Storage for all results
    all_results = []

    # Run experiments
    overall_start = time.time()
    for exp_idx, exp_config in enumerate(experiments):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {exp_idx+1}/{len(experiments)}: {exp_config['name']}")
        print(f"# {exp_config['description']}")
        print(f"{'#'*80}")

        for class_idx, class_name in enumerate(classes):
            print(f"\n[Progress] Experiment {exp_idx+1}/{len(experiments)}, Class {class_idx+1}/{len(classes)}")

            try:
                results = run_single_experiment(args, exp_config, class_name)
                all_results.append(results)

                # Save intermediate results after each run
                save_results(all_results, args.results_dir, args.dataset)

                # Print current result
                print(f"\n[Result] {class_name}: I-ROCAUC={results['image_rocauc']:.3f}, "
                      f"P-ROCAUC={results['pixel_rocauc']:.3f}, AU-PRO={results['au_pro']:.3f}")

            except Exception as e:
                print(f"\n[ERROR] Failed on {exp_config['name']} - {class_name}")
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # Final results summary
    elapsed_time = time.time() - overall_start
    print(f"\n\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total runs: {len(all_results)}/{len(experiments) * len(classes)}")
    print(f"Total time: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    print(f"Results saved to: {args.results_dir}")

    # Print summary
    if all_results:
        print_results_summary(all_results)

def save_results(results, results_dir, dataset):
    """Save results to CSV and JSON"""
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, f'results_{dataset}_latest.csv')
    df.to_csv(csv_path, index=False)

    # Also save timestamped version
    csv_path_ts = os.path.join(results_dir, f'results_{dataset}_{timestamp}.csv')
    df.to_csv(csv_path_ts, index=False)

    # Save as JSON
    json_path = os.path.join(results_dir, f'results_{dataset}_latest.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[Saved] {csv_path}")

def print_results_summary(results):
    """Print summary of results"""
    df = pd.DataFrame(results)

    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Group by experiment
    summary = df.groupby('experiment').agg({
        'image_rocauc': ['mean', 'std', 'min', 'max'],
        'pixel_rocauc': ['mean', 'std'],
        'au_pro': ['mean', 'std']
    }).round(3)

    print("Average metrics across all classes:\n")
    print(summary)

    print(f"\n\nDetailed Image ROC-AUC by class:\n")
    detailed = df.pivot_table(
        index='class_name',
        columns='experiment',
        values='image_rocauc'
    ).round(3)
    print(detailed)

    # Find best experiment
    best_exp = df.groupby('experiment')['image_rocauc'].mean().idxmax()
    best_score = df.groupby('experiment')['image_rocauc'].mean().max()
    baseline_score = df[df['experiment']=='baseline']['image_rocauc'].mean() if 'baseline' in df['experiment'].values else None

    print(f"\n\n{'='*80}")
    print(f"BEST EXPERIMENT: {best_exp}")
    print(f"Average Image ROC-AUC: {best_score:.3f}")
    if baseline_score is not None:
        improvement = (best_score - baseline_score) * 100
        print(f"Improvement over baseline: {improvement:+.2f}%")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Autodl Experiment Runner')

    # Experiment selection
    parser.add_argument('--exp', type=str, default='baseline',
                        help='Experiment name, "all" for all, "list" to show available')
    parser.add_argument('--dataset', type=str, default='real',
                        help='Dataset name')
    parser.add_argument('--class_name', type=str, default='all',
                        help='Class name or "all"')

    # Model arguments
    parser.add_argument('--feature', type=str, default='FPFH')
    parser.add_argument('--use_LFSA', type=bool, default=True)
    parser.add_argument('--use_MSND', type=bool, default=False)
    parser.add_argument('--num_MSND', type=int, default=1)
    parser.add_argument('--max_nn', type=int, default=30)
    parser.add_argument('--num_group', type=int, default=256)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vis_save', type=bool, default=False)
    parser.add_argument('--level', type=str, default='ALL')

    # Paths
    parser.add_argument('--results_dir', type=str, default=RESULTS_BASE,
                        help='Directory to save results')

    args = parser.parse_args()

    # Check GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'

    # Run experiments
    run_experiments(args)

if __name__ == '__main__':
    main()
