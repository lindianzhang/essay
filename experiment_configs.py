"""
Experiment configurations for testing three improvements:
1. Top-k mean distance
2. Local Contrast Normalization (LCN)
3. Smooth object-level aggregation

Each configuration is a dictionary with experiment settings.
"""

# Baseline (original method)
BASELINE = {
    'name': 'baseline',
    'use_topk_mean': False,
    'use_lcn': False,
    'use_smooth_aggregation': False,
    'description': 'Original method: min distance + max aggregation'
}

# Improvement 1: Top-k mean distance
TOPK_EXPERIMENTS = [
    {
        'name': 'topk_k3',
        'use_topk_mean': True,
        'topk_k': 3,
        'use_lcn': False,
        'use_smooth_aggregation': False,
        'description': 'Top-3 mean distance'
    },
    {
        'name': 'topk_k5',
        'use_topk_mean': True,
        'topk_k': 5,
        'use_lcn': False,
        'use_smooth_aggregation': False,
        'description': 'Top-5 mean distance'
    },
    {
        'name': 'topk_k7',
        'use_topk_mean': True,
        'topk_k': 7,
        'use_lcn': False,
        'use_smooth_aggregation': False,
        'description': 'Top-7 mean distance'
    }
]

# Improvement 2: Local Contrast Normalization
LCN_EXPERIMENTS = [
    {
        'name': 'lcn_k3',
        'use_topk_mean': False,
        'use_lcn': True,
        'lcn_neighbors': 3,
        'use_smooth_aggregation': False,
        'description': 'LCN with 3 neighbors'
    },
    {
        'name': 'lcn_k5',
        'use_topk_mean': False,
        'use_lcn': True,
        'lcn_neighbors': 5,
        'use_lcn': False,
        'use_smooth_aggregation': False,
        'description': 'LCN with 5 neighbors'
    },
    {
        'name': 'lcn_k10',
        'use_topk_mean': False,
        'use_lcn': True,
        'lcn_neighbors': 10,
        'use_smooth_aggregation': False,
        'description': 'LCN with 10 neighbors'
    }
]

# Improvement 3: Smooth aggregation
SMOOTH_EXPERIMENTS = [
    {
        'name': 'smooth_k10_p1',
        'use_topk_mean': False,
        'use_lcn': False,
        'use_smooth_aggregation': True,
        'smooth_k': 10,
        'top_percentile': 0.01,
        'description': 'Smooth (k=10) + Top 1%'
    },
    {
        'name': 'smooth_k10_p2',
        'use_topk_mean': False,
        'use_lcn': False,
        'use_smooth_aggregation': True,
        'smooth_k': 10,
        'top_percentile': 0.02,
        'description': 'Smooth (k=10) + Top 2%'
    },
    {
        'name': 'smooth_k15_p1',
        'use_topk_mean': False,
        'use_lcn': False,
        'use_smooth_aggregation': True,
        'smooth_k': 15,
        'top_percentile': 0.01,
        'description': 'Smooth (k=15) + Top 1%'
    }
]

# Combined improvements
COMBINED_EXPERIMENTS = [
    {
        'name': 'all_improvements',
        'use_topk_mean': True,
        'topk_k': 3,
        'use_lcn': True,
        'lcn_neighbors': 5,
        'use_smooth_aggregation': True,
        'smooth_k': 10,
        'top_percentile': 0.01,
        'description': 'All three improvements combined'
    },
    {
        'name': 'topk_lcn',
        'use_topk_mean': True,
        'topk_k': 3,
        'use_lcn': True,
        'lcn_neighbors': 5,
        'use_smooth_aggregation': False,
        'description': 'Top-k + LCN'
    },
    {
        'name': 'topk_smooth',
        'use_topk_mean': True,
        'topk_k': 3,
        'use_lcn': False,
        'use_smooth_aggregation': True,
        'smooth_k': 10,
        'top_percentile': 0.01,
        'description': 'Top-k + Smooth aggregation'
    },
    {
        'name': 'lcn_smooth',
        'use_topk_mean': False,
        'use_lcn': True,
        'lcn_neighbors': 5,
        'use_smooth_aggregation': True,
        'smooth_k': 10,
        'top_percentile': 0.01,
        'description': 'LCN + Smooth aggregation'
    }
]

# All experiments
ALL_EXPERIMENTS = (
    [BASELINE] +
    TOPK_EXPERIMENTS +
    LCN_EXPERIMENTS +
    SMOOTH_EXPERIMENTS +
    COMBINED_EXPERIMENTS
)

def get_experiment_by_name(name):
    """Get experiment config by name"""
    for exp in ALL_EXPERIMENTS:
        if exp['name'] == name:
            return exp
    return None

def print_all_experiments():
    """Print all available experiments"""
    print("\n" + "="*80)
    print("AVAILABLE EXPERIMENTS")
    print("="*80)

    print("\n[BASELINE]")
    print(f"  {BASELINE['name']}: {BASELINE['description']}")

    print("\n[IMPROVEMENT 1: Top-k Mean Distance]")
    for exp in TOPK_EXPERIMENTS:
        print(f"  {exp['name']}: {exp['description']}")

    print("\n[IMPROVEMENT 2: Local Contrast Normalization]")
    for exp in LCN_EXPERIMENTS:
        print(f"  {exp['name']}: {exp['description']}")

    print("\n[IMPROVEMENT 3: Smooth Aggregation]")
    for exp in SMOOTH_EXPERIMENTS:
        print(f"  {exp['name']}: {exp['description']}")

    print("\n[COMBINED IMPROVEMENTS]")
    for exp in COMBINED_EXPERIMENTS:
        print(f"  {exp['name']}: {exp['description']}")

    print("\n" + "="*80)
    print(f"Total: {len(ALL_EXPERIMENTS)} experiments")
    print("="*80 + "\n")

if __name__ == '__main__':
    print_all_experiments()
