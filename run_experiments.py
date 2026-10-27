"""
Experiment runner for testing three improvements systematically.

Usage:
    python run_experiments.py --dataset minishift --class_name all --exp baseline
    python run_experiments.py --dataset minishift --class_name all --exp topk_k3
    python run_experiments.py --dataset minishift --class_name all --exp all  # Run all experiments
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
from experiment_configs import ALL_EXPERIMENTS, get_experiment_by_name, print_all_experiments

def update_args_with_experiment(args, exp_config):
    """Update args with experiment configuration"""
    for key, value in exp_config.items():
        if key not in ['name', 'description']:
            setattr(args, key, value)
    return args

def run_single_experiment(args, exp_config, class_name):
    """Run a single experiment for a given class"""
    print(f"\n{'='*80}")
    print(f"Running: {exp_config['name']} - {exp_config['description']}")
    print(f"Class: {class_name}")
    print(f"{'='*80}\n")

    # Update args with experiment config
    args_exp = update_args_with_experiment(args, exp_config)

    # Import and run the patchcore runner
    from patchcore_runner import PatchCore

    # Create PatchCore instance with improved features
    patchcore = PatchCore(args=args_exp)

    # Fit on training data
    print(f"\n[Fitting] Training on class: {class_name}")
    patchcore.fit(class_name)

    # Evaluate on test data
    print(f"\n[Evaluating] Testing on class: {class_name}")
    image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(class_name)

    # Extract results
    method_name = list(image_rocaucs.keys())[0]
    results = {
        'experiment': exp_config['name'],
        'description': exp_config['description'],
        'class_name': class_name,
        'image_rocauc': image_rocaucs[method_name],
        'pixel_rocauc': pixel_rocaucs[method_name],
        'au_pro': au_pros[method_name],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add experiment parameters
    for key, value in exp_config.items():
        if key not in ['name', 'description']:
            results[f'param_{key}'] = value

    return results

def run_experiments(args):
    """Main experiment runner"""
    # Determine which experiments to run
    if args.exp == 'all':
        experiments = ALL_EXPERIMENTS
        print(f"\nRunning ALL {len(experiments)} experiments")
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
        if args.dataset == 'minishift':
            from data.MiniShiftAD import minishiftAD_classes
            classes = minishiftAD_classes()
        elif args.dataset == 'mulsen':
            from data.MulSen import mulsen_classes
            classes = mulsen_classes()
        elif args.dataset == 'shapenet':
            from data.anomalyshape import shapenet3d_classes
            classes = shapenet3d_classes()
        elif args.dataset == 'mvtec':
            from data.mvtec3d import mvtec3d_classes
            classes = mvtec3d_classes()
        elif args.dataset == 'real':
            from data.real3d import real3d_classes
            classes = real3d_classes()
        else:
            print(f"Error: Unknown dataset '{args.dataset}'")
            return
    else:
        classes = [args.class_name]

    print(f"\nDataset: {args.dataset}")
    print(f"Classes: {classes}")
    print(f"Experiments: {len(experiments)}")
    print(f"Total runs: {len(experiments) * len(classes)}\n")

    # Storage for all results
    all_results = []

    # Run experiments
    start_time = time.time()
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

            except Exception as e:
                print(f"\n[ERROR] Failed on {exp_config['name']} - {class_name}")
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # Final results summary
    elapsed_time = time.time() - start_time
    print(f"\n\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total runs: {len(all_results)}")
    print(f"Total time: {elapsed_time/60:.2f} minutes")
    print(f"Results saved to: {args.results_dir}")

    # Print summary
    print_results_summary(all_results)

def save_results(results, results_dir, dataset):
    """Save results to CSV and JSON"""
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, f'results_{dataset}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)

    # Save as JSON
    json_path = os.path.join(results_dir, f'results_{dataset}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Saved] Results updated: {csv_path}")

def print_results_summary(results):
    """Print summary of results"""
    if not results:
        return

    df = pd.DataFrame(results)

    print(f"\n\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Group by experiment
    summary = df.groupby('experiment').agg({
        'image_rocauc': ['mean', 'std'],
        'pixel_rocauc': ['mean', 'std'],
        'au_pro': ['mean', 'std']
    }).round(3)

    print("Average metrics across all classes:\n")
    print(summary)

    print(f"\n\nDetailed results by class:\n")
    detailed = df.pivot_table(
        index='class_name',
        columns='experiment',
        values='image_rocauc'
    ).round(3)
    print(detailed)

    # Find best experiment
    best_exp = df.groupby('experiment')['image_rocauc'].mean().idxmax()
    best_score = df.groupby('experiment')['image_rocauc'].mean().max()

    print(f"\n\n{'='*80}")
    print(f"BEST EXPERIMENT: {best_exp}")
    print(f"Average Image ROC-AUC: {best_score:.3f}")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Run anomaly detection experiments')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='minishift',
                        choices=['minishift', 'mulsen', 'shapenet', 'mvtec', 'real'],
                        help='Dataset name')
    parser.add_argument('--class_name', type=str, default='all',
                        help='Class name or "all" for all classes')
    parser.add_argument('--level', type=str, default='ALL',
                        help='Difficulty level for MiniShift dataset')

    # Experiment selection
    parser.add_argument('--exp', type=str, default='baseline',
                        help='Experiment name or "all" for all experiments, "list" to show all')

    # Model arguments (from original code)
    parser.add_argument('--feature', type=str, default='FPFH')
    parser.add_argument('--use_LFSA', type=bool, default=True)
    parser.add_argument('--use_MSND', type=bool, default=False)
    parser.add_argument('--num_MSND', type=int, default=1)
    parser.add_argument('--max_nn', type=int, default=30)
    parser.add_argument('--num_group', type=int, default=256)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vis_save', type=bool, default=False)

    # Results directory
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Run experiments
    run_experiments(args)

if __name__ == '__main__':
    main()
