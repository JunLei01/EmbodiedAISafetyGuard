"""
Scene Graph / VLM Feature Extractor
====================================
Extracts entity features from images or scene descriptions.

In a full system, this would use:
    - A VLM (e.g., LLaVA, InternVL) for open-vocabulary detection
    - A Scene Graph Generator (e.g., Neural Motifs, VCTree)
    - Or a pretrained backbone (ResNet, ViT) + RoI features

Here we provide:
    1. A pluggable interface for real VLM backends
    2. A feature encoder for structured scene graphs
    3. A synthetic feature generator for testing
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Entity:
    """An entity detected in a scene."""
    id: str
    category: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    attributes: Dict[str, float] = field(default_factory=dict)
    visual_features: Optional[torch.Tensor] = None


@dataclass
class SceneGraph:
    """A scene graph representing entities and spatial relations."""
    entities: List[Entity]
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    # relations: [(subject_id, predicate, object_id)]

    @property
    def entity_ids(self) -> List[str]:
        return [e.id for e in self.entities]

    @property
    def entity_pairs(self) -> List[Tuple[str, str]]:
        """All directed entity pairs."""
        ids = self.entity_ids
        return [(a, b) for a in ids for b in ids if a != b]

    def get_entity(self, eid: str) -> Optional[Entity]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None


class SceneFeatureEncoder(nn.Module):
    """
    Encodes scene graph entities into feature vectors.
    
    For each entity, encodes:
    - Category embedding
    - Spatial features (from bbox)
    - Attribute features
    - Optional: visual features from VLM backbone
    
    Output: feature_dim-dimensional vector per entity
    """

    # Common object categories for risk assessment
    CATEGORIES = [
        "person", "child", "fire", "knife", "sword", "scissors",
        "water", "liquid", "chemical", "gas", "bottle", "container",
        "electrical_device", "wire", "stove", "oven", "match",
        "glass", "ceramic", "heavy_object", "vehicle", "tool",
        "medication", "cleaning_product", "unknown",
    ]

    def __init__(
        self,
        feature_dim: int = 256,
        num_attributes: int = 16,
        use_visual_features: bool = False,
        visual_feature_dim: int = 2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Category embedding
        self.category_embed = nn.Embedding(
            len(self.CATEGORIES) + 1, feature_dim // 4  # +1 for unknown
        )
        self.cat2idx = {c: i for i, c in enumerate(self.CATEGORIES)}

        # Spatial encoder (bbox → features)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, feature_dim // 4),
            nn.ReLU(),
        )

        # Attribute encoder
        self.attribute_encoder = nn.Sequential(
            nn.Linear(num_attributes, feature_dim // 4),
            nn.ReLU(),
        )

        # Optional visual feature projector
        self.use_visual = use_visual_features
        if use_visual_features:
            self.visual_projector = nn.Sequential(
                nn.Linear(visual_feature_dim, feature_dim // 4),
                nn.ReLU(),
            )
            fusion_dim = feature_dim  # All four components
        else:
            fusion_dim = 3 * (feature_dim // 4)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self, scene_graph: SceneGraph, batch_size: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all entities in a scene graph.
        
        Returns: {entity_id: tensor(batch, feature_dim)}
        """
        device = next(self.parameters()).device
        features = {}

        for entity in scene_graph.entities:
            # Category
            cat_idx = self.cat2idx.get(entity.category, len(self.CATEGORIES))
            cat_tensor = torch.tensor([cat_idx], device=device).expand(batch_size)
            cat_feat = self.category_embed(cat_tensor)

            # Spatial (bbox)
            if entity.bbox is not None:
                bbox = torch.tensor(
                    [list(entity.bbox)], device=device
                ).expand(batch_size, -1)
            else:
                bbox = torch.zeros(batch_size, 4, device=device)
            spatial_feat = self.spatial_encoder(bbox)

            # Attributes
            attr_vec = self._encode_attributes(entity.attributes, device)
            attr_vec = attr_vec.unsqueeze(0).expand(batch_size, -1)
            attr_feat = self.attribute_encoder(attr_vec)

            # Fuse
            parts = [cat_feat, spatial_feat, attr_feat]

            if self.use_visual and entity.visual_features is not None:
                vis_feat = self.visual_projector(
                    entity.visual_features.expand(batch_size, -1)
                )
                parts.append(vis_feat)

            combined = torch.cat(parts, dim=-1)
            features[entity.id] = self.fusion(combined)

        return features

    def _encode_attributes(
        self, attributes: Dict[str, float], device: torch.device
    ) -> torch.Tensor:
        """Encode entity attributes as a fixed-size vector."""
        attr_names = [
            "temperature", "size", "weight", "sharpness",
            "flammability", "toxicity", "fragility", "conductivity",
            "velocity", "height", "wetness", "age",
            "brightness", "opacity", "pressure", "volume",
        ]
        vec = torch.zeros(len(attr_names), device=device)
        for i, name in enumerate(attr_names):
            if name in attributes:
                vec[i] = attributes[name]
        return vec


# ─────────────────────────────────────────────────────────
# Synthetic scene generation (for testing & demos)
# ─────────────────────────────────────────────────────────

def create_synthetic_scene(scenario: str = "kitchen_fire") -> SceneGraph:
    """Create synthetic scene graphs for testing."""

    scenarios = {
        "kitchen_fire": SceneGraph(
            entities=[
                Entity("person1", "person", (0.1, 0.2, 0.4, 0.8),
                       {"temperature": 0.3, "velocity": 0.0}),
                Entity("fire1", "fire", (0.5, 0.3, 0.7, 0.6),
                       {"temperature": 1.0, "brightness": 0.9}),
                Entity("cloth1", "unknown", (0.15, 0.3, 0.35, 0.5),
                       {"flammability": 0.9, "weight": 0.1}),
                Entity("bottle1", "bottle", (0.6, 0.6, 0.7, 0.8),
                       {"toxicity": 0.0, "volume": 0.5}),
            ],
            relations=[
                ("person1", "near", "fire1"),
                ("person1", "holding", "cloth1"),
                ("bottle1", "near", "fire1"),
            ]
        ),
        "child_kitchen": SceneGraph(
            entities=[
                Entity("child1", "child", (0.2, 0.3, 0.4, 0.9),
                       {"age": 0.2, "velocity": 0.5}),
                Entity("knife1", "knife", (0.5, 0.5, 0.6, 0.55),
                       {"sharpness": 0.95, "weight": 0.15}),
                Entity("stove1", "stove", (0.6, 0.1, 0.9, 0.5),
                       {"temperature": 0.8}),
                Entity("glass1", "glass", (0.45, 0.6, 0.5, 0.65),
                       {"fragility": 0.9}),
            ],
            relations=[
                ("child1", "near", "knife1"),
                ("child1", "near", "stove1"),
                ("child1", "near", "glass1"),
            ]
        ),
        "chemical_spill": SceneGraph(
            entities=[
                Entity("worker1", "person", (0.3, 0.2, 0.5, 0.8),
                       {"velocity": 0.1}),
                Entity("chemical1", "chemical", (0.6, 0.4, 0.7, 0.6),
                       {"toxicity": 0.95, "volume": 0.8}),
                Entity("wire1", "wire", (0.5, 0.5, 0.8, 0.55),
                       {"conductivity": 0.9}),
                Entity("water1", "liquid", (0.55, 0.6, 0.75, 0.75),
                       {"conductivity": 0.7, "volume": 0.6}),
            ],
            relations=[
                ("worker1", "near", "chemical1"),
                ("chemical1", "near", "water1"),
                ("wire1", "near", "water1"),
            ]
        ),
        "safe_office": SceneGraph(
            entities=[
                Entity("person1", "person", (0.2, 0.2, 0.5, 0.8),
                       {"velocity": 0.0}),
                Entity("laptop1", "electrical_device", (0.5, 0.4, 0.7, 0.5),
                       {"temperature": 0.2}),
                Entity("mug1", "container", (0.6, 0.5, 0.65, 0.55),
                       {"temperature": 0.3, "fragility": 0.3}),
            ],
            relations=[
                ("person1", "near", "laptop1"),
                ("person1", "near", "mug1"),
            ]
        ),
    }

    if scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Available: {list(scenarios.keys())}"
        )

    return scenarios[scenario]