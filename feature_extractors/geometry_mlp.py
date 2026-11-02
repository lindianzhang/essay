import torch
from torch import nn


class GeometryFusionMLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, use_position=True, use_curvature=True):
        super().__init__()
        self.use_position = use_position
        self.use_curvature = use_curvature
        self.geometry_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.position_proj = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU())
        self.curvature_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 3))

    def forward(self, geometry, position, curvature):
        geometry_feature = self.geometry_proj(geometry)
        position_feature = self.position_proj(position)
        curvature_feature = self.curvature_proj(curvature)
        branches = torch.stack((geometry_feature, position_feature, curvature_feature), dim=1)
        gate_input = torch.cat((geometry_feature,
                                position_feature if self.use_position else torch.zeros_like(position_feature),
                                curvature_feature if self.use_curvature else torch.zeros_like(curvature_feature)), dim=1)
        gate_logits = self.gate(gate_input)
        enabled = torch.tensor([True, self.use_position, self.use_curvature], device=geometry.device)
        weights = torch.softmax(gate_logits.masked_fill(~enabled, float('-inf')), dim=1)
        embedding = (branches * weights.unsqueeze(-1)).sum(dim=1) + geometry_feature
        return embedding, weights


class ConcatFusionMLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64,
                 use_position=True, use_curvature=True):
        super().__init__()
        self.use_position = use_position
        self.use_curvature = use_curvature
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, geometry, position, curvature):
        if not self.use_position:
            position = torch.zeros_like(position)
        if not self.use_curvature:
            curvature = torch.zeros_like(curvature)

        x = torch.cat([geometry, position, curvature], dim=1)
        return self.net(x), None
