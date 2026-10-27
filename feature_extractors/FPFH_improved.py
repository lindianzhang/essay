"""
Improved FPFH Features class that uses the enhanced compute_anomay_scores method
"""
from feature_extractors.FPFH import FPFHFeatures
from feature_extractors.features_improved import Features as FeaturesImproved

class FPFHFeaturesImproved(FeaturesImproved):
    """
    FPFH Features using improved anomaly detection methods
    Inherits from FeaturesImproved to get enhanced compute_anomay_scores
    """

    def __init__(self, args=None):
        self.args = args
        super().__init__(args=args)

    # Import all FPFH-specific methods from FPFHFeatures
    def get_fpfh_features(self, unorganized_pc, voxel_size=0.1):
        # Reuse the implementation from FPFHFeatures
        from feature_extractors.FPFH import FPFHFeatures
        temp_instance = FPFHFeatures(args=self.args)
        return temp_instance.get_fpfh_features(unorganized_pc, voxel_size)

    def get_features(self, unorganized_pc):
        from feature_extractors.FPFH import FPFHFeatures
        temp_instance = FPFHFeatures(args=self.args)
        return temp_instance.get_features(unorganized_pc)

    def collect_features(self, pc):
        from feature_extractors.FPFH import FPFHFeatures
        temp_instance = FPFHFeatures(args=self.args)
        temp_instance.patch_lib = self.patch_lib
        temp_instance.collect_features(pc)
        self.patch_lib = temp_instance.patch_lib

    def predict(self, pc, mask, label, path=None):
        from feature_extractors.FPFH import FPFHFeatures
        temp_instance = FPFHFeatures(args=self.args)
        agg_point_feature, unorganized_pc, unorganized_pc_no_zeros, center = temp_instance.get_features(pc)
        # Use the improved compute_anomay_scores from FeaturesImproved
        self.compute_anomay_scores(agg_point_feature, mask, label, path, unorganized_pc, unorganized_pc_no_zeros, center)
