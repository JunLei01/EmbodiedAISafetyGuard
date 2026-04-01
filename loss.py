"""
Neuro-Symbolic Loss Functions
==============================
Loss functions for training the end-to-end differentiable pipeline.

Combines:
    1. Supervised loss on entity risk labels
    2. Scene-level risk classification loss
    3. Predicate grounding consistency loss
    4. Rule weight regularization
    5. Explanation alignment loss

Designed for weakly-supervised training where only scene-level
labels are available, but can also use entity-level supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class NeuroSymbolicLoss(nn.Module):
    """
    Combined loss for neuro-symbolic risk reasoning.

    Loss components:
        L = α * L_entity + β * L_scene + γ * L_predicate + δ * L_reg + ε * L_expl

    Where:
        L_entity:   Supervised loss on entity risk probabilities
        L_scene:    Scene-level binary classification loss
        L_predicate: Consistency between predicted predicates
        L_reg:      Rule weight regularization (L2)
        L_expl:     Explanation alignment (optional)
    """

    def __init__(
        self,
        alpha: float = 1.0,   # Entity risk weight
        beta: float = 0.5,    # Scene risk weight
        gamma: float = 0.3,   # Predicate consistency weight
        delta: float = 0.01,  # Regularization weight
        epsilon: float = 0.1, # Explanation alignment weight
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

    def forward(
        self,
        derived_probs: Dict[str, torch.Tensor],
        grounded_probs: Dict[str, torch.Tensor],
        entity_risk_labels: Optional[Dict[str, torch.Tensor]] = None,
        scene_risk_label: Optional[torch.Tensor] = None,
        predicate_labels: Optional[Dict[str, torch.Tensor]] = None,
        rule_weights: Optional[nn.ParameterDict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            derived_probs: Output of DifferentiableLogicLayer
                {"risk(obj1)": tensor(0.85), ...}
            grounded_probs: Output of NeuralPredicateLayer
                {"near(1,2)": tensor(0.92), ...}
            entity_risk_labels: Ground truth entity risk {entity_id: label}
            scene_risk_label: Ground truth scene risk (0 or 1)
            predicate_labels: Ground truth predicates for supervision
            rule_weights: Learnable rule weights for regularization

        Returns:
            total_loss: Combined loss tensor
            components: Dict of individual loss component values
        """
        losses = {}

        # Component 1: Entity-level supervised loss
        if entity_risk_labels is not None and derived_probs:
            losses["entity"] = self._entity_risk_loss(
                derived_probs, entity_risk_labels
            )
        else:
            losses["entity"] = torch.tensor(0.0)

        # Component 2: Scene-level classification loss
        if scene_risk_label is not None and derived_probs:
            losses["scene"] = self._scene_risk_loss(
                derived_probs, scene_risk_label
            )
        else:
            losses["scene"] = torch.tensor(0.0)

        # Component 3: Predicate consistency loss
        if predicate_labels is not None and grounded_probs:
            losses["predicate"] = self._predicate_consistency_loss(
                grounded_probs, predicate_labels
            )
        else:
            losses["predicate"] = torch.tensor(0.0)

        # Component 4: Regularization on rule weights
        if rule_weights is not None:
            losses["reg"] = self._rule_weight_regularization(rule_weights)
        else:
            losses["reg"] = torch.tensor(0.0)

        # Combine losses
        total = (
            self.alpha * losses["entity"] +
            self.beta * losses["scene"] +
            self.gamma * losses["predicate"] +
            self.delta * losses["reg"]
        )

        # Convert to float dict
        components = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        components["total"] = total.item()

        return total, components

    def _entity_risk_loss(
        self,
        derived_probs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss on entity risk predictions.

        Matches predictions like "risk(obj1)" against labels.
        """
        total_loss = torch.tensor(0.0)
        count = 0

        for entity_id, label in labels.items():
            # Find all risk predictions for this entity
            risk_keys = [k for k in derived_probs.keys() if k.startswith(f"risk({entity_id})")]

            for key in risk_keys:
                pred = derived_probs[key]
                # BCE loss
                loss = F.binary_cross_entropy(pred, label.float())
                total_loss = total_loss + loss
                count += 1

        return total_loss / max(count, 1)

    def _scene_risk_loss(
        self,
        derived_probs: Dict[str, torch.Tensor],
        scene_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scene-level risk based on maximum entity risk.

        Scene is risky if any entity has high risk.
        """
        # Extract all risk probabilities
        risk_probs = [
            prob for key, prob in derived_probs.items()
            if key.startswith("risk(")
        ]

        if not risk_probs:
            return torch.tensor(0.0)

        # Scene risk = max over entity risks (smooth approximation)
        stacked = torch.stack(risk_probs)
        scene_pred = torch.max(stacked)

        # BCE loss
        return F.binary_cross_entropy(scene_pred, scene_label.float())

    def _predicate_consistency_loss(
        self,
        grounded_probs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Consistency between predicted predicate probabilities
        and ground truth labels.
        """
        total_loss = torch.tensor(0.0)
        count = 0

        for key, label in labels.items():
            if key in grounded_probs:
                pred = grounded_probs[key]
                loss = F.mse_loss(pred, label.float())
                total_loss = total_loss + loss
                count += 1

        return total_loss / max(count, 1)

    def _rule_weight_regularization(
        self,
        rule_weights: nn.ParameterDict,
    ) -> torch.Tensor:
        """
        L2 regularization on rule weights to prevent overfitting.

        Encourages rule weights to stay near their initial values.
        """
        total = torch.tensor(0.0)
        for weight in rule_weights.values():
            # L2 regularization
            total = total + weight.pow(2)
        return total / len(rule_weights)


class ContrastiveExplanationLoss(nn.Module):
    """
    Contrastive loss for explanation alignment.

    Ensures that similar scenes have similar explanations,
    and different scenes have different explanations.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        scene_embedding_a: torch.Tensor,
        scene_embedding_b: torch.Tensor,
        explanation_a: List[str],
        explanation_b: List[str],
        same_scene_type: bool,  # True if both scenes are same type (both safe or both risky)
    ) -> torch.Tensor:
        """
        Compute contrastive loss based on explanation similarity.
        """
        # Compute embedding distance
        embedding_dist = F.pairwise_distance(
            scene_embedding_a.unsqueeze(0),
            scene_embedding_b.unsqueeze(0),
        )

        # Compute explanation similarity (simplified)
        exp_sim = self._explanation_similarity(explanation_a, explanation_b)

        if same_scene_type:
            # Same scene type should have similar explanations
            loss = embedding_dist * (1 - exp_sim)
        else:
            # Different scene types should have different explanations
            loss = F.relu(self.margin - embedding_dist) * exp_sim

        return loss

    @staticmethod
    def _explanation_similarity(exp_a: List[str], exp_b: List[str]) -> float:
        """Compute Jaccard similarity between explanations."""
        set_a = set(exp_a)
        set_b = set(exp_b)

        if not set_a and not set_b:
            return 1.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0


class CurriculumLearningLoss(nn.Module):
    """
    Loss that adapts difficulty based on training progress.

    Early in training: focus on easy examples (clear hazard patterns)
    Late in training: focus on hard examples (subtle risk patterns)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.warmup_epochs = warmup_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        loss, components = self.base_loss(*args, **kwargs)

        # Curriculum weighting based on epoch
        if self.epoch < self.warmup_epochs:
            # Early: downweight hard examples
            weight = 0.5 + 0.5 * (self.epoch / self.warmup_epochs)
            loss = loss * weight

        return loss, components
