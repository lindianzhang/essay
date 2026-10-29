from data.mvtec3d import get_mvtec_loader,mvtec3d_classes
from data.real3d import get_real_loader,real3d_classes
from data.anomalyshape import get_shapenet_loader,shapenet3d_classes
from data.MulSen import get_mulsen_loader,mulsen_classes
from data.MiniShiftAD import get_minishift_loader,minishiftAD_classes
# from data.quan import get_quan_loader,quan_classes  # Module not available
import torch
from tqdm import tqdm
from feature_extractors.FPFH import FPFHFeatures
import numpy as np
import os
from feature_extractors.pointnet2_utils import *


class PatchCore():
    def __init__(self, ckp = '', image_size=224, args=None):
        self.args = args
        self.method  = FPFHFeatures(args=args)
        self.dataset_name = self.args.dataset
        self.level = self.args.level

    def get_dataloader(self,dataset_name,split,class_name,level='ALL'):
        if dataset_name == 'mvtec':
            return get_mvtec_loader(split, class_name=class_name)
        if dataset_name == 'real':
            return get_real_loader(split, class_name=class_name)
        if dataset_name == 'shapenet':
            return get_shapenet_loader(split, class_name=class_name)
        if dataset_name == 'mulsen':
            return get_mulsen_loader(split, class_name=class_name)
        if dataset_name == 'minishift':
            return get_minishift_loader(split, class_name=class_name,level=level)  

    def fit(self, class_name):
        train_loader = self.get_dataloader(self.dataset_name,'train',class_name,level=self.level)
        train_count = len(train_loader)
        use_ncrp = getattr(self.args, 'agg_method', 'mean') == 'ncrp'
        if use_ncrp and train_count < 2:
            raise ValueError('NCRP requires at least two normal training samples for independent calibration.')

        # Use the same memory-bank subset in Mean and NCRP ablations.
        calibration_indices = set()
        if train_count >= 2:
            calibration_count = min(train_count - 1, max(1, (train_count + 4) // 5))
            calibration_indices = set(
                np.random.RandomState(0).choice(train_count, calibration_count, replace=False).tolist()
            )

        for index, (pc, _, _, path) in enumerate(tqdm(train_loader, desc=f'Extracting train features for class {class_name}')):
            if index in calibration_indices:
                continue
            self.method.collect_features(pc)
            self.method.name_list.append(path)
        print(f'\n\nRunning coreset on class {class_name}...')
        self.method.run_coreset()

        if use_ncrp:
            # Score only held-out normal training samples; never use test data.
            for index, (pc, _, _, _) in enumerate(tqdm(train_loader, desc=f'Calibrating NCRP for class {class_name}')):
                if index in calibration_indices:
                    self.method.collect_calibration_scores(pc)
            self.method.fit_ncrp_score_calibration()


    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = self.get_dataloader(self.dataset_name,'test',class_name,level=self.level)
        with torch.no_grad():
            self.method.init_para()
            self.method.name_list = []
            self.method.test_patch_lib = []
        
            for pc, mask, label, path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                self.method.predict(pc, mask, label,path)
        method_name = "Simple3D"
        self.method.calculate_metrics()
        image_rocaucs[method_name] = round(self.method.image_rocauc, 3)
        pixel_rocaucs[method_name] = round(self.method.pixel_rocauc, 3)
        au_pros[method_name] = round(self.method.au_pro, 3)
        print(
            f'Class: {class_name}, {method_name} Image ROCAUC: {self.method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {self.method.pixel_rocauc:.3f}, {method_name} AU-PRO: {self.method.au_pro:.3f}')
        return image_rocaucs, pixel_rocaucs, au_pros

