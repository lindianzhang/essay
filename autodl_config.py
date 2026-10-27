"""
Configuration for autodl server paths and experiment settings
"""

# Autodl server paths
DATASET_PATH = "/root/autodl-fs/Real3D-AD-PCD/Real3D-AD-PCD"
CHECKPOINT_PATH = "/root/autodl-fs/output_real3dad/output_real3dad/snapshots/snapshot.pth.tar"
OUTPUT_PATH = "/root/autodl-fs/output_real3dad"

# Experiment results path
EXPERIMENT_RESULTS_PATH = "/root/autodl-fs/experiment_results"

# Dataset configurations
DATASET_CONFIGS = {
    'real3d': {
        'path': DATASET_PATH,
        'classes': [
            "airplane",
            "candybar",
            "car",
            "chicken",
            "diamond",
            "duck",
            "fish",
            "gemstone",
            "seahorse",
            "shell",
            "starfish",
            "toffees",
        ]
    }
}

# Model configurations
MODEL_CONFIG = {
    'feature': 'FPFH',
    'use_LFSA': True,
    'use_MSND': False,
    'num_MSND': 1,
    'max_nn': 30,
    'num_group': 256,
    'group_size': 64,
    'device': 'cuda',
    'vis_save': False
}
