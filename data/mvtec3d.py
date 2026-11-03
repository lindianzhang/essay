import os
from PIL import Image
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import torch


DATASETS_PATH = '../Semicore/data/mvtec3ds'


def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]



class MVTec3D(Dataset):

    def __init__(self, split, class_name, data_root=None):
        self.cls = class_name
        self.data_root = Path(data_root or os.environ.get('MVTEC3D_ROOT') or DATASETS_PATH)
        self.data_path = self.data_root / self.cls / split

    @staticmethod
    def read_xyz(path):
        organized_pc = read_tiff_organized_pc(str(path))
        if organized_pc.ndim != 3 or organized_pc.shape[2] != 3:
            raise ValueError(f'XYZ TIFF must have shape [H, W, 3]: {path}')
        resized = resize_organized_pc(organized_pc.astype(np.float32, copy=False))
        if resized.shape != (3, 224, 224):
            raise ValueError(f'XYZ TIFF preprocessing did not produce [3, 224, 224]: {path}')
        return organized_pc_to_unorganized_pc(resized.permute(1, 2, 0))


class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, data_root=None):
        super().__init__(split="train", class_name=class_name, data_root=data_root)
        self.class_name = class_name
        self.pcd_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        paths = sorted((self.data_path / 'good' / 'xyz').glob('*.tiff'))
        if not paths:
            raise FileNotFoundError(f'No normal XYZ TIFF files in {self.data_path}')
        return [str(path) for path in paths], [0] * len(paths)

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, label = self.pcd_paths[idx], self.labels[idx]
        return self.read_xyz(pcd_path), label, label, pcd_path


class MVTec3DTest(MVTec3D):
    def __init__(self, class_name, data_root=None):
        super().__init__(split="test", class_name=class_name, data_root=data_root)
        self.pcd_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

    def load_dataset(self):
        pcd_paths, gt_paths, labels = [], [], []
        if not self.data_path.is_dir():
            raise FileNotFoundError(f'MVTec 3D test directory not found: {self.data_path}')
        for defect_dir in sorted(path for path in self.data_path.iterdir() if path.is_dir()):
            xyz_paths = sorted((defect_dir / 'xyz').glob('*.tiff'))
            if not xyz_paths:
                raise FileNotFoundError(f'No XYZ TIFF files in {defect_dir / "xyz"}')
            if defect_dir.name == 'good':
                for xyz_path in xyz_paths:
                    pcd_paths.append(str(xyz_path))
                    gt_paths.append(0)
                    labels.append(0)
                continue
            masks = {}
            for mask_path in (defect_dir / 'gt').glob('*.png'):
                if mask_path.stem in masks:
                    raise ValueError(f'Duplicate GT mask identifier: {mask_path.stem}')
                masks[mask_path.stem] = mask_path
            xyz_ids = {path.stem for path in xyz_paths}
            if len(xyz_ids) != len(xyz_paths):
                raise ValueError(f'Duplicate XYZ TIFF sample identifiers in {defect_dir}')
            if set(masks) != xyz_ids:
                raise ValueError(f'XYZ TIFF and GT PNG identifiers differ in {defect_dir}')
            for xyz_path in xyz_paths:
                pcd_paths.append(str(xyz_path))
                gt_paths.append(str(masks[xyz_path.stem]))
                labels.append(1)
        return pcd_paths, gt_paths, labels

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path, gt, label = self.pcd_paths[idx], self.gt_paths[idx], self.labels[idx]
        unorganized_pc = self.read_xyz(pcd_path)

        if gt == 0:
            gt = torch.zeros([1, 224, 224])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        if unorganized_pc.shape != (224 * 224, 3) or gt.shape != (1, 224, 224):
            raise ValueError(f'Point grid and GT mask dimensions differ: {pcd_path}')
        return unorganized_pc, gt[:1], label, pcd_path


def get_mvtec_loader(split, class_name, data_root=None):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name, data_root=data_root)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name, data_root=data_root)
    else:
        raise ValueError(f'Unknown MVTec 3D split: {split}')

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader
