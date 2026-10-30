import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import torch
DATASETS_PATH = '/root/autodl-fs/Real3D-AD-PCD/Real3D-AD-PCD'
OFFICIAL_DATASETS_PATH = '../Semicore/data/real3d_train_cut'
def real3d_classes():
    return [
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

voxel_size_setting = 0.15
class Real3D(Dataset):

    def __init__(self, split, class_name):
        self.cls = class_name
        self.data_path = os.path.join(DATASETS_PATH, self.cls, split)


class Real3DTrain(Real3D):
    def __init__(self, class_name):
        super().__init__(split="train", class_name=class_name)
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        pcd_tot_paths = []
        tot_labels = []

        name = self.cls
        self.pcd_path = os.path.join(DATASETS_PATH, name, 'train')
        pcd_paths = glob.glob(self.pcd_path+ "/*.pcd")
        pcd_paths.sort()
        pcd_tot_paths.extend(pcd_paths)
        tot_labels.extend([0] * len(pcd_paths))
        return pcd_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_setting)

        resized_organized_pc = np.asarray(pcd.points)
        unorganized_pc = resized_organized_pc

        return unorganized_pc, label, label, pcd_path


class Real3DTest(Real3D):
    def __init__(self, class_name):
        super().__init__(split="test", class_name=class_name)
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        pcd_tot_paths = []
        gt_tot_paths = []
        tot_labels = []

        # Get all pcd files in test directory (exclude *_cut.pcd)
        test_dir = os.path.join(DATASETS_PATH, self.cls, 'test')
        gt_dir = os.path.join(DATASETS_PATH, self.cls, 'gt')

        pcd_files = glob.glob(os.path.join(test_dir, "*.pcd"))
        pcd_files = [f for f in pcd_files if '_cut.pcd' not in f]  # Exclude cut files
        pcd_files.sort()

        for pcd_path in pcd_files:
            filename = os.path.basename(pcd_path)
            base_name = filename.replace('.pcd', '')

            # Check if this is a good sample or defect
            if '_good' in filename:
                # Good sample
                pcd_tot_paths.append(pcd_path)
                gt_tot_paths.append(0)
                tot_labels.append(0)
            else:
                # Defect sample - look for corresponding gt file
                gt_filename = base_name + '.txt'
                gt_path = os.path.join(gt_dir, gt_filename)

                if os.path.exists(gt_path):
                    pcd_tot_paths.append(pcd_path)
                    gt_tot_paths.append(gt_path)
                    tot_labels.append(1)

        return pcd_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, gt, label = self.pcd_paths[idx], self.gt_paths[idx], self.labels[idx]

        # Read point cloud
        pcd = o3d.io.read_point_cloud(pcd_path)

        # Downsample
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_setting)

        unorganized_pc = np.asarray(pcd.points)

        # Handle ground truth
        if gt == 0:
            # Good sample - all zeros
            gt_mask = torch.zeros([1, unorganized_pc.shape[0]])
        else:
            # Defect sample - GT file contains all points with labels in column 4
            gt_data = np.loadtxt(gt, dtype=np.float32)
            gt_points = gt_data[:, :3]
            gt_labels = gt_data[:, 3]

            # Match downsampled test points to GT points using KNN
            from scipy.spatial import cKDTree
            tree = cKDTree(gt_points)
            _, indices = tree.query(unorganized_pc, k=1)

            # Get labels for downsampled points
            point_labels = gt_labels[indices]
            gt_mask = torch.from_numpy(point_labels).float()
            gt_mask = gt_mask.unsqueeze(0)

        return unorganized_pc, gt_mask[:1], label, pcd_path


class OfficialReal3D(Dataset):
    """The author's preprocessed train_cut ASC and labeled test TXT layout."""

    def __init__(self, split, class_name, data_root):
        self.cls = class_name
        self.data_root = data_root
        self.data_path = os.path.join(data_root, class_name, split)


class OfficialReal3DTrain(OfficialReal3D):
    def __init__(self, class_name, data_root):
        super().__init__('train', class_name, data_root)
        train_cut_dir = os.path.join(data_root, class_name, 'train_cut')
        self.pcd_paths = sorted(glob.glob(os.path.join(train_cut_dir, '*.asc')))
        if not self.pcd_paths:
            raise FileNotFoundError(
                f'Official Real3D train_cut/*.asc data not found in {train_cut_dir}. '
                'Choose --real3d_format pcd for the PCD dataset or set --real3d_root.'
            )

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path = self.pcd_paths[idx]
        input_points = np.loadtxt(pcd_path, dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_points[:, 0:3])
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
        unorganized_pc = np.asarray(pcd.points)
        return unorganized_pc, 0, 0, pcd_path


class OfficialReal3DTest(OfficialReal3D):
    def __init__(self, class_name, data_root):
        super().__init__('test', class_name, data_root)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f'Official Real3D test/*.txt data not found in {self.data_path}. '
                'Choose --real3d_format pcd for the PCD dataset or set --real3d_root.'
            )
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()
        if not self.pcd_paths:
            raise FileNotFoundError(f'No official Real3D test/*.txt files found in {self.data_path}.')

    def load_dataset(self):
        pcd_tot_paths, gt_tot_paths, tot_labels = [], [], []
        for defect_type in os.listdir(self.data_path):
            defect_dir = os.path.join(self.data_path, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            pcd_paths = sorted(glob.glob(os.path.join(defect_dir, '*.txt')))
            pcd_tot_paths.extend(pcd_paths)
            if defect_type == 'good':
                gt_tot_paths.extend([0] * len(pcd_paths))
                tot_labels.extend([0] * len(pcd_paths))
            else:
                # In the official format, each test TXT contains its own GT in column four.
                gt_tot_paths.extend(pcd_paths)
                tot_labels.extend([1] * len(pcd_paths))
        return pcd_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, gt, label = self.pcd_paths[idx], self.gt_paths[idx], self.labels[idx]
        input_points = np.loadtxt(pcd_path, dtype=np.float32)

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        idx1 = input_points[:, 3] == 0
        idx2 = input_points[:, 3] == 1
        pcd1.points = o3d.utility.Vector3dVector(input_points[idx1, 0:3])
        pcd2.points = o3d.utility.Vector3dVector(input_points[idx2, 0:3])

        pcd = pcd1 + pcd2
        pcd_new = pcd.voxel_down_sample(voxel_size=voxel_size_setting)
        pcd1_vec, pcd2_vec = [], []
        pc_len = len(pcd1.points)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for point in np.asarray(pcd_new.points):
            _, indices, _ = pcd_tree.search_knn_vector_3d(point, 1)
            if indices[0] < pc_len:
                pcd1_vec.append(point)
            else:
                pcd2_vec.append(point)

        pcd1.points = o3d.utility.Vector3dVector(np.asarray(pcd1_vec))
        if len(pcd2_vec) != 0:
            pcd2.points = o3d.utility.Vector3dVector(np.asarray(pcd2_vec))
        pcd = pcd1 + pcd2
        unorganized_pc = np.asarray(pcd.points)

        if gt == 0:
            gt_mask = torch.zeros([1, unorganized_pc.shape[0]])
        else:
            gt_mask = torch.zeros(unorganized_pc.shape[0])
            gt_mask[len(pcd1.points):len(pcd.points)] = 1.0
            gt_mask = torch.where(gt_mask > 0.5, 1., .0)
            gt_mask = gt_mask.unsqueeze(0)
            gt_mask = gt_mask.unsqueeze(0)
        return unorganized_pc, gt_mask[:1], label, pcd_path


def get_real_loader(split, class_name, data_format='pcd', data_root=None):
    if data_format == 'official':
        root = data_root or OFFICIAL_DATASETS_PATH
        if split == 'train':
            dataset = OfficialReal3DTrain(class_name=class_name, data_root=root)
        elif split == 'test':
            dataset = OfficialReal3DTest(class_name=class_name, data_root=root)
        else:
            raise ValueError(f'Unknown Real3D split: {split}')
    elif data_format == 'pcd':
        if data_root is not None:
            raise ValueError('--real3d_root is only used with --real3d_format official.')
        if split == 'train':
            dataset = Real3DTrain(class_name=class_name)
        elif split == 'test':
            dataset = Real3DTest(class_name=class_name)
        else:
            raise ValueError(f'Unknown Real3D split: {split}')
    else:
        raise ValueError(f'Unknown Real3D format: {data_format}')

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
