"""
Neural Predicate Layer
=======================
Maps entity features to neural predicate probabilities.

This module provides learnable neural networks that ground predicates
from entity features extracted by the scene encoder.

Examples:
    near(X, Y)      → MLP(entity_pair_features) → P(near)
    flammable(X)    → MLP(entity_features) → P(flammable)
    sharp(X)        → MLP(entity_features) → P(sharp)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class UnaryPredicateNet(nn.Module):
    """
    Neural network for unary predicates (properties of single entities).

    Examples:
        flammable(X), sharp(X), is_fire(X), is_person(X), etc.

    Architecture:
        features → Linear → ReLU → Linear → Sigmoid → probability
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: tensor(batch, feature_dim) or tensor(feature_dim,)
        Returns:
            probabilities: tensor(batch,) or tensor scalar
        """
        # Handle both single and batch inputs
        squeeze = False
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze = True

        prob = self.net(features).squeeze(-1)

        if squeeze:
            prob = prob.squeeze(0)

        return prob


class BinaryPredicateNet(nn.Module):
    """
    Neural network for binary predicates (relations between entity pairs).

    Examples:
        near(X, Y), touching(X, Y), above(X, Y), holding(X, Y), etc.

    Architecture:
        concat(entity_a_features, entity_b_features, spatial_features)
        → Linear → ReLU → Linear → Sigmoid → probability
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        use_spatial_features: bool = True,
    ):
        super().__init__()
        self.use_spatial = use_spatial_features

        # Input: concatenated features of two entities
        input_dim = feature_dim * 2
        if use_spatial_features:
            # Add relative position features (dx, dy, dz, distance)
            input_dim += 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        spatial_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features_a: tensor(batch, feature_dim)
            features_b: tensor(batch, feature_dim)
            spatial_feat: Optional tensor(batch, 4) - [dx, dy, dz, dist]
        Returns:
            probabilities: tensor(batch,)
        """
        # Ensure batch dimension
        if features_a.dim() == 1:
            features_a = features_a.unsqueeze(0)
        if features_b.dim() == 1:
            features_b = features_b.unsqueeze(0)

        # Concatenate features
        combined = torch.cat([features_a, features_b], dim=-1)

        # Add spatial features if provided
        if self.use_spatial and spatial_feat is not None:
            if spatial_feat.dim() == 1:
                spatial_feat = spatial_feat.unsqueeze(0)
            combined = torch.cat([combined, spatial_feat], dim=-1)

        return self.net(combined).squeeze(-1)


class NeuralPredicateLayer(nn.Module):
    """
    Complete neural predicate layer managing all predicates.

    This is a collection of neural networks, one per predicate type,
    that ground all predicates from scene entity features.
    """

    # Standard unary predicates for embodied safety
    DEFAULT_UNARY_PREDICATES = [
        # Material/physical properties
        "flammable",
        "liquid",
        "sharp",
        "fragile",
        "heavy",
        "toxic",
        # Functional types
        "is_fire",
        "is_hot",
        "is_hot_liquid",
        "is_electrical",
        "is_container",
        # Living entities
        "is_person",
        "is_child",
        "is_cloth",
        "is_oil",
        # States
        "is_wet",
        "is_exposed",
        "is_unstable",
        "is_medication",
        "on_floor",
        "running",
    ]

    # Standard binary predicates (spatial and relational)
    DEFAULT_BINARY_PREDICATES = [
        "near",
        "touching",
        "above",
        "holding",
        "on",
        "inside",
    ]

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        unary_predicates: Optional[List[str]] = None,
        binary_predicates: Optional[List[str]] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Define predicates
        self.unary_preds = unary_predicates or self.DEFAULT_UNARY_PREDICATES
        self.binary_preds = binary_predicates or self.DEFAULT_BINARY_PREDICATES

        # Create networks for unary predicates
        self.unary_nets = nn.ModuleDict({
            name: UnaryPredicateNet(feature_dim, hidden_dim)
            for name in self.unary_preds
        })

        # Create networks for binary predicates
        self.binary_nets = nn.ModuleDict({
            name: BinaryPredicateNet(feature_dim, hidden_dim)
            for name in self.binary_preds
        })

    def forward(
        self,
        entity_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Ground all unary predicates for all entities.

        Args:
            entity_features: {entity_id: tensor(feature_dim)}

        Returns:
            grounded_probs: {"flammable(obj1)": tensor(prob), ...}
        """
        grounded: Dict[str, torch.Tensor] = {}

        for entity_id, features in entity_features.items():
            for pred_name, net in self.unary_nets.items():
                prob = net(features)
                grounded[f"{pred_name}({entity_id})"] = prob

        return grounded

    def ground_binary(
        self,
        entity_features: Dict[str, torch.Tensor],
        entity_pairs: List[Tuple[str, str]],
        positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Ground all binary predicates for specified entity pairs.

        Args:
            entity_features: {entity_id: tensor(feature_dim)}
            entity_pairs: [(id_a, id_b), ...]
            positions: Optional {entity_id: tensor(3,)} for spatial features

        Returns:
            grounded_probs: {"near(obj1,obj2)": tensor(prob), ...}
        """
        grounded: Dict[str, torch.Tensor] = {}

        for id_a, id_b in entity_pairs:
            if id_a not in entity_features or id_b not in entity_features:
                continue

            feat_a = entity_features[id_a]
            feat_b = entity_features[id_b]

            # Compute spatial features if positions available
            spatial_feat = None
            if positions is not None:
                if id_a in positions and id_b in positions:
                    pos_a = positions[id_a]
                    pos_b = positions[id_b]
                    diff = pos_b - pos_a  # Vector from a to b
                    dist = torch.norm(diff)
                    spatial_feat = torch.cat([diff, dist.unsqueeze(0)])

            for pred_name, net in self.binary_nets.items():
                prob = net(feat_a, feat_b, spatial_feat)
                grounded[f"{pred_name}({id_a},{id_b})"] = prob

        return grounded

    def ground_all(
        self,
        entity_features: Dict[str, torch.Tensor],
        entity_pairs: Optional[List[Tuple[str, str]]] = None,
        positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Ground all predicates (unary + binary).

        Args:
            entity_features: {entity_id: tensor(feature_dim)
            entity_pairs: Optional list of pairs for binary predicates
            positions: Optional positions for spatial features

        Returns:
            grounded_probs: {"predicate(args)": tensor(prob), ...}
        """
        grounded = self.forward(entity_features)

        if entity_pairs:
            binary_grounded = self.ground_binary(
                entity_features, entity_pairs, positions
            )
            grounded.update(binary_grounded)

        return grounded


class CreateRiskPredicateLayer(NeuralPredicateLayer):
    """
    Pre-configured predicate layer for risk assessment scenarios.

    Includes all predicates needed for the embodied safety use cases.
    """

    RISK_UNARY = [
        # Fire/burn hazards
        "flammable",
        "is_fire",
        "is_hot",
        "is_hot_liquid",
        "is_cloth",
        "is_oil",
        # Chemical hazards
        "liquid",
        "toxic",
        # Physical hazards
        "sharp",
        "fragile",
        "heavy",
        # Electrical hazards
        "is_electrical",
        # Containment
        "is_container",
        # Demographics
        "is_person",
        "is_child",
        "is_medication",
        # State-based
        "is_wet",
        "is_exposed",
        "is_unstable",
        "on_floor",
        "running",
    ]

    RISK_BINARY = [
        "near",
        "touching",
        "above",
        "holding",
        "on",
        "inside",
    ]

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            unary_predicates=self.RISK_UNARY,
            binary_predicates=self.RISK_BINARY,
        )


def create_risk_predicate_layer(feature_dim: int = 256) -> NeuralPredicateLayer:
    """Factory function for risk assessment predicate layer."""
    return CreateRiskPredicateLayer(feature_dim=feature_dim)
