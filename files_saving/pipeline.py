"""
End-to-End Differentiable Pipeline
===================================
Combines all components into a single differentiable model:

    Image/Scene → Scene Graph → Neural Predicates → Logic Inference → Risk

The entire pipeline is differentiable, allowing gradients to flow from
the final risk probability back through logic rules to neural predicate
parameters and even the scene encoder.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from files_saving.scene_graph import SceneGraph, SceneFeatureEncoder
from neural_predicates import NeuralPredicateLayer, create_risk_predicate_layer
from knowledge_base import KnowledgeBase, create_risk_kb
from logic_layer import DifferentiableLogicLayer, ProofTracer
from loss import NeuroSymbolicLoss


class NeuroSymbolicRiskReasoner(nn.Module):
    """
    Complete neuro-symbolic risk reasoning pipeline.
    
    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Scene Graph (from VLM or structured input)          │
    │  entities: person1, fire1, knife1, ...               │
    │  relations: near(person1, fire1), ...                │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  Scene Feature Encoder                               │
    │  entity_id → feature_vector (256-dim)                │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  Neural Predicate Layer                              │
    │  near(X,Y)=0.92  flammable(X)=0.87  sharp(Y)=0.95  │
    │  (each is a small MLP: features → probability)      │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  Differentiable Logic Layer (DeepProbLog-style)      │
    │  fire_risk(X) :- near(X,Y), flammable(X), fire(Y).  │
    │  risk(X) :- fire_risk(X).                            │
    │  Semiring: AND=product, OR=noisy-OR                  │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼───────────────────────────────────┐
    │  Risk Probabilities                                  │
    │  risk(person1) = 0.78                                │
    │  fire_risk(person1) = 0.73                           │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        knowledge_base: Optional[KnowledgeBase] = None,
        learn_rule_weights: bool = True,
    ):
        super().__init__()

        self.scene_encoder = SceneFeatureEncoder(feature_dim=feature_dim)
        self.neural_predicates = create_risk_predicate_layer(feature_dim=feature_dim)
        self.kb = knowledge_base or create_risk_kb()
        self.logic_layer = DifferentiableLogicLayer(
            knowledge_base=self.kb,
            learn_rule_weights=learn_rule_weights,
        )
        self.loss_fn = NeuroSymbolicLoss()

    def forward(
        self,
        scene_graph: SceneGraph,
        batch_size: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Full forward pass: scene → risk probabilities.
        
        Returns:
            derived_probs:  {key: prob} for derived predicates (risk, fire_risk, etc.)
            grounded_probs: {key: prob} for neural predicates (near, sharp, etc.)
        """
        # Step 1: Encode entities into feature vectors
        entity_features = self.scene_encoder(scene_graph, batch_size)

        # Step 2: Ground all neural predicates
        grounded_probs = self.neural_predicates.ground_all(
            entity_features=entity_features,
            entity_pairs=scene_graph.entity_pairs,
        )

        # Step 3: Run differentiable logic inference
        derived_probs = self.logic_layer(
            grounded_facts=grounded_probs,
            entities=scene_graph.entity_ids,
        )

        return derived_probs, grounded_probs

    def compute_loss(
        self,
        scene_graph: SceneGraph,
        entity_risk_labels: Optional[Dict[str, torch.Tensor]] = None,
        scene_risk_label: Optional[torch.Tensor] = None,
        predicate_labels: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Forward pass + loss computation.
        
        Returns:
            loss, loss_components, derived_probs
        """
        derived_probs, grounded_probs = self.forward(scene_graph, batch_size)

        loss, components = self.loss_fn(
            derived_probs=derived_probs,
            grounded_probs=grounded_probs,
            entity_risk_labels=entity_risk_labels,
            scene_risk_label=scene_risk_label,
            predicate_labels=predicate_labels,
        )

        return loss, components, derived_probs

    def explain(
        self,
        scene_graph: SceneGraph,
        entity_id: str,
        batch_size: int = 1,
    ) -> List[str]:
        """
        Generate human-readable explanation for an entity's risk.
        
        Returns list of explanation strings.
        """
        derived, grounded = self.forward(scene_graph, batch_size)
        query = f"risk({entity_id})"

        all_facts = {**grounded, **derived}
        explanation = ProofTracer.trace(
            query=query,
            result=derived,
            grounded_facts=all_facts,
            kb=self.kb,
        )

        return explanation

    def get_all_risk_scores(
        self,
        scene_graph: SceneGraph,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """Get risk scores for all entities as a simple dict."""
        derived, _ = self.forward(scene_graph, batch_size)
        risks = {}
        for key, prob in derived.items():
            if key.startswith("risk("):
                entity_id = key[5:-1]  # extract from "risk(entity_id)"
                risks[entity_id] = prob.item()
        return risks


class PipelineConfig:
    """Configuration for the full pipeline."""

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        learn_rule_weights: bool = True,
        loss_alpha: float = 1.0,
        loss_beta: float = 0.5,
        loss_gamma: float = 0.3,
        loss_delta: float = 0.01,
        loss_epsilon: float = 0.1,
    ):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.learn_rule_weights = learn_rule_weights
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_gamma = loss_gamma
        self.loss_delta = loss_delta
        self.loss_epsilon = loss_epsilon

    def build_model(self) -> NeuroSymbolicRiskReasoner:
        model = NeuroSymbolicRiskReasoner(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            learn_rule_weights=self.learn_rule_weights,
        )
        model.loss_fn = NeuroSymbolicLoss(
            alpha=self.loss_alpha,
            beta=self.loss_beta,
            gamma=self.loss_gamma,
            delta=self.loss_delta,
            epsilon=self.loss_epsilon,
        )
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )