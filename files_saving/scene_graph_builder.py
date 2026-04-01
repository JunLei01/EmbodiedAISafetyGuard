"""
Neural Predicate → Scene Graph Builder
=======================================
Bridges VLM / LLaVa-3D neural predicate output to the differentiable logic layer.

Pipeline position:
    LLaVa-3D / 3D Object Detection
            ↓
    Neural Predicate Grounding (text output)
        object(1, gasoline_can, flammable).
        attribute(2, temperature, 150).
        position(1, 1.2, 0.5, 0.8).
            ↓
    ┌─────────────────────────────────────────┐
    │  >>> THIS MODULE <<<                    │
    │  NeuralPredicateParser                  │
    │  SceneGraphBuilder                      │
    │  SpatialRelationInferrer                │
    │  PredicateGrounder                      │
    └─────────────────────────────────────────┘
            ↓
    Scene Graph + Grounded Fact Probabilities
    {
        "near(1,2)":       tensor(0.92),
        "flammable(1)":    tensor(0.95),
        "fire_source(2)":  tensor(1.00),
        ...
    }
            ↓
    DifferentiableLogicLayer (forward chaining)

Design Principles:
    1. Prolog-like text predicates are parsed into structured Python objects
    2. Spatial relations (near, on, above, inside) are INFERRED from 3D positions
       using differentiable distance kernels — not hardcoded thresholds
    3. Semantic type properties (flammable, sharp, fire_source) are mapped to
       neural predicate probabilities with configurable confidence
    4. The entire grounding is differentiable: position → distance → soft relation
       probability, enabling end-to-end gradient flow
"""

import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Dict, List, Tuple, Optional, Set, Any, Union, NamedTuple
)
from dataclasses import dataclass, field
from enum import Enum, auto


# Part 1: Data Structures

class PredicateType(Enum):
    """Types of neural predicates from VLM output."""
    OBJECT     = auto()   # object(id, class, semantic_type)
    ATTRIBUTE  = auto()   # attribute(id, attr_name, value)
    POSITION   = auto()   # position(id, x, y, z)
    RELATION   = auto()   # relation(id1, rel_type, id2)
    STATE      = auto()   # state(id, state_name, value)
    ACTION     = auto()   # action(agent_id, action_name, target_id)


@dataclass
class RawPredicate:
    """A single parsed predicate from VLM text output."""
    pred_type: PredicateType
    functor: str              # e.g., "object", "attribute", "position"
    arguments: List[Any]      # parsed arguments (mixed types)
    confidence: float = 1.0   # VLM confidence score if available
    raw_text: str = ""        # original text for debugging

    def __repr__(self):
        args = ", ".join(str(a) for a in self.arguments)
        conf = f" [{self.confidence:.2f}]" if self.confidence < 1.0 else ""
        return f"{self.functor}({args}){conf}"


@dataclass
class SceneObject:
    """
    A fully parsed scene object with all associated predicates.
    
    Aggregates information from multiple predicate lines:
        object(1, gasoline_can, flammable)  → id, category, semantic_types
        attribute(1, temperature, 25)       → attributes
        position(1, 1.2, 0.5, 0.8)         → position_3d
        state(1, open, true)                → states
    """
    obj_id: str
    category: str
    semantic_types: Set[str] = field(default_factory=set)
    # e.g., {"flammable", "liquid", "container"}

    position_3d: Optional[Tuple[float, float, float]] = None
    # (x, y, z) in world coordinates

    attributes: Dict[str, float] = field(default_factory=dict)
    # e.g., {"temperature": 150.0, "weight": 2.5}

    states: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"open": True, "powered": True}

    detection_confidence: float = 1.0
    # VLM detection confidence for this object

    def __repr__(self):
        types_str = ", ".join(sorted(self.semantic_types))
        pos_str = (
            f"({self.position_3d[0]:.2f}, {self.position_3d[1]:.2f}, "
            f"{self.position_3d[2]:.2f})"
            if self.position_3d else "(?)"
        )
        return (
            f"Obj[{self.obj_id}] {self.category} "
            f"types={{{types_str}}} pos={pos_str}"
        )


@dataclass
class ExplicitRelation:
    """A relation explicitly stated by the VLM (not inferred from positions)."""
    subject_id: str
    relation: str        # e.g., "on", "holding", "inside"
    object_id: str
    confidence: float = 1.0


@dataclass
class ParsedScene:
    """
    Complete parsed scene from VLM neural predicate output.
    Intermediate representation before SceneGraph construction.
    """
    objects: Dict[str, SceneObject] = field(default_factory=dict)
    explicit_relations: List[ExplicitRelation] = field(default_factory=list)
    raw_predicates: List[RawPredicate] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)

    @property
    def object_ids(self) -> List[str]:
        return list(self.objects.keys())

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def summary(self) -> str:
        lines = [f"ParsedScene: {self.num_objects} objects"]
        for obj in self.objects.values():
            lines.append(f"  {obj}")
        if self.explicit_relations:
            lines.append(f"  Explicit relations: {len(self.explicit_relations)}")
            for rel in self.explicit_relations:
                lines.append(
                    f"    {rel.subject_id} --{rel.relation}--> {rel.object_id}"
                )
        if self.parse_warnings:
            lines.append(f"  Warnings: {len(self.parse_warnings)}")
            for w in self.parse_warnings:
                lines.append(f"    ⚠ {w}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Part 2: Neural Predicate Parser
# ═══════════════════════════════════════════════════════════

class NeuralPredicateParser:
    """
    Parses Prolog-style neural predicate text output from VLM/LLaVa-3D
    into structured Python objects.
    
    Supported predicate formats:
        object(ID, Category, SemanticType).
        attribute(ID, AttrName, Value).
        position(ID, X, Y, Z).
        relation(ID1, RelType, ID2).
        state(ID, StateName, Value).
        action(AgentID, ActionName, TargetID).
    
    Also supports:
        - Confidence annotations: predicate(...) [0.95].
        - Comments: % this is a comment
        - Multi-type objects: object(1, gasoline_can, [flammable, liquid]).
        - Numeric/string value auto-detection
    """

    # Regex patterns for predicate parsing
    # Matches: functor(arg1, arg2, ...) with optional [confidence] and trailing .
    _PRED_PATTERN = re.compile(
        r"(\w+)\s*\(\s*"       # functor(
        r"(.+?)"               # arguments (non-greedy)
        r"\s*\)"               # )
        r"(?:\s*\[(\d*\.?\d+)\])?"  # optional [confidence]
        r"\s*\.?"              # optional trailing period
    )

    # Predicate functor → type mapping
    _FUNCTOR_MAP = {
        "object":    PredicateType.OBJECT,
        "obj":       PredicateType.OBJECT,
        "attribute": PredicateType.ATTRIBUTE,
        "attr":      PredicateType.ATTRIBUTE,
        "position":  PredicateType.POSITION,
        "pos":       PredicateType.POSITION,
        "location":  PredicateType.POSITION,
        "relation":  PredicateType.RELATION,
        "rel":       PredicateType.RELATION,
        "state":     PredicateType.STATE,
        "action":    PredicateType.ACTION,
        "act":       PredicateType.ACTION,
    }

    def parse(self, text: str) -> ParsedScene:
        """
        Parse complete VLM neural predicate text output into a ParsedScene.
        
        Args:
            text: Multi-line string of Prolog-style predicates.
                  Each line is one predicate, comments start with %.
                  
        Returns:
            ParsedScene with all objects, attributes, positions, and relations.
        """
        scene = ParsedScene()
        lines = text.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("%") or line.startswith("//"):
                continue

            # Strip inline comments
            if "%" in line:
                line = line[:line.index("%")].strip()

            try:
                predicate = self._parse_predicate_line(line)
                if predicate is not None:
                    scene.raw_predicates.append(predicate)
                    self._integrate_predicate(predicate, scene)
            except Exception as e:
                scene.parse_warnings.append(
                    f"Line {line_num}: Failed to parse '{line}' — {e}"
                )

        return scene

    def parse_predicate_list(
        self, predicates: List[str]
    ) -> ParsedScene:
        """Parse a list of predicate strings (one per entry)."""
        return self.parse("\n".join(predicates))

    def parse_dict_format(
        self, data: Dict[str, List]
    ) -> ParsedScene:
        """
        Parse from a structured dictionary format.
        
        Example:
            {
                "objects": [
                    {"id": 1, "category": "gasoline_can", "types": ["flammable"]},
                    {"id": 2, "category": "stove", "types": ["fire_source"]},
                ],
                "attributes": [
                    {"id": 1, "name": "temperature", "value": 25},
                ],
                "positions": [
                    {"id": 1, "x": 1.2, "y": 0.5, "z": 0.8},
                ],
                "relations": [
                    {"subject": 1, "relation": "on", "object": 2},
                ],
            }
        """
        scene = ParsedScene()

        # Parse objects
        for obj_data in data.get("objects", []):
            oid = str(obj_data["id"])
            cat = obj_data.get("category", "unknown")
            types = set(obj_data.get("types", []))
            conf = obj_data.get("confidence", 1.0)
            scene.objects[oid] = SceneObject(
                obj_id=oid, category=cat,
                semantic_types=types, detection_confidence=conf,
            )

        # Parse attributes
        for attr_data in data.get("attributes", []):
            oid = str(attr_data["id"])
            if oid in scene.objects:
                name = attr_data["name"]
                value = float(attr_data["value"])
                scene.objects[oid].attributes[name] = value

        # Parse positions
        for pos_data in data.get("positions", []):
            oid = str(pos_data["id"])
            if oid in scene.objects:
                scene.objects[oid].position_3d = (
                    float(pos_data["x"]),
                    float(pos_data["y"]),
                    float(pos_data["z"]),
                )

        # Parse relations
        for rel_data in data.get("relations", []):
            scene.explicit_relations.append(ExplicitRelation(
                subject_id=str(rel_data["subject"]),
                relation=rel_data["relation"],
                object_id=str(rel_data["object"]),
                confidence=rel_data.get("confidence", 1.0),
            ))

        return scene

    # ─── Internal parsing methods ───

    def _parse_predicate_line(self, line: str) -> Optional[RawPredicate]:
        """Parse a single predicate line."""
        match = self._PRED_PATTERN.match(line)
        if not match:
            return None

        functor = match.group(1).lower()
        args_str = match.group(2)
        confidence = float(match.group(3)) if match.group(3) else 1.0

        # Determine predicate type
        pred_type = self._FUNCTOR_MAP.get(functor)
        if pred_type is None:
            # Unknown functor → treat as generic relation if 2-3 args
            pred_type = PredicateType.RELATION

        # Parse arguments
        arguments = self._parse_arguments(args_str)

        return RawPredicate(
            pred_type=pred_type,
            functor=functor,
            arguments=arguments,
            confidence=confidence,
            raw_text=line,
        )

    def _parse_arguments(self, args_str: str) -> List[Any]:
        """
        Parse comma-separated arguments with type inference.
        
        Handles:
            - Integers: 1, 42
            - Floats: 1.2, 0.5
            - Booleans: true, false
            - Lists: [flammable, liquid]
            - Strings: gasoline_can, fire_source
        """
        # Handle list arguments: [a, b, c]
        # First, temporarily replace lists with placeholders
        lists = {}
        list_pattern = re.compile(r"\[([^\]]*)\]")
        placeholder_idx = 0

        def replace_list(m):
            nonlocal placeholder_idx
            key = f"__LIST_{placeholder_idx}__"
            items = [
                self._parse_single_value(item.strip())
                for item in m.group(1).split(",")
                if item.strip()
            ]
            lists[key] = items
            placeholder_idx += 1
            return key

        args_str = list_pattern.sub(replace_list, args_str)

        # Split by comma and parse each argument
        raw_args = [a.strip() for a in args_str.split(",")]
        parsed = []
        for arg in raw_args:
            if arg in lists:
                parsed.append(lists[arg])
            else:
                parsed.append(self._parse_single_value(arg))
        return parsed

    @staticmethod
    def _parse_single_value(val: str) -> Any:
        """Parse a single value with type inference."""
        val = val.strip().strip("'\"")

        # Boolean
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False

        # Integer
        try:
            return int(val)
        except ValueError:
            pass

        # Float
        try:
            return float(val)
        except ValueError:
            pass

        # String (identifier)
        return val

    def _integrate_predicate(
        self, pred: RawPredicate, scene: ParsedScene
    ):
        """Integrate a parsed predicate into the scene."""

        if pred.pred_type == PredicateType.OBJECT:
            self._integrate_object(pred, scene)
        elif pred.pred_type == PredicateType.ATTRIBUTE:
            self._integrate_attribute(pred, scene)
        elif pred.pred_type == PredicateType.POSITION:
            self._integrate_position(pred, scene)
        elif pred.pred_type == PredicateType.RELATION:
            self._integrate_relation(pred, scene)
        elif pred.pred_type == PredicateType.STATE:
            self._integrate_state(pred, scene)
        elif pred.pred_type == PredicateType.ACTION:
            self._integrate_action(pred, scene)

    def _integrate_object(self, pred: RawPredicate, scene: ParsedScene):
        """
        object(ID, Category, SemanticType)
        object(ID, Category, [Type1, Type2, ...])
        """
        args = pred.arguments
        if len(args) < 2:
            scene.parse_warnings.append(
                f"object predicate needs ≥2 args, got {len(args)}: {pred}"
            )
            return

        oid = str(args[0])
        category = str(args[1])

        # Semantic types: can be a single string or a list
        semantic_types = set()
        if len(args) >= 3:
            type_arg = args[2]
            if isinstance(type_arg, list):
                semantic_types = {str(t) for t in type_arg}
            else:
                semantic_types = {str(type_arg)}

        if oid in scene.objects:
            # Merge with existing object
            scene.objects[oid].semantic_types |= semantic_types
            if category != "unknown":
                scene.objects[oid].category = category
        else:
            scene.objects[oid] = SceneObject(
                obj_id=oid,
                category=category,
                semantic_types=semantic_types,
                detection_confidence=pred.confidence,
            )

    def _integrate_attribute(self, pred: RawPredicate, scene: ParsedScene):
        """attribute(ID, AttrName, Value)"""
        args = pred.arguments
        if len(args) < 3:
            scene.parse_warnings.append(
                f"attribute predicate needs 3 args: {pred}"
            )
            return

        oid = str(args[0])
        attr_name = str(args[1])
        value = float(args[2]) if isinstance(args[2], (int, float)) else 0.0

        # Auto-create object if not yet seen
        if oid not in scene.objects:
            scene.objects[oid] = SceneObject(
                obj_id=oid, category="unknown"
            )

        scene.objects[oid].attributes[attr_name] = value

    def _integrate_position(self, pred: RawPredicate, scene: ParsedScene):
        """position(ID, X, Y, Z)"""
        args = pred.arguments
        if len(args) < 4:
            scene.parse_warnings.append(
                f"position predicate needs 4 args (id, x, y, z): {pred}"
            )
            return

        oid = str(args[0])
        x, y, z = float(args[1]), float(args[2]), float(args[3])

        if oid not in scene.objects:
            scene.objects[oid] = SceneObject(
                obj_id=oid, category="unknown"
            )

        scene.objects[oid].position_3d = (x, y, z)

    def _integrate_relation(self, pred: RawPredicate, scene: ParsedScene):
        """
        relation(ID1, RelType, ID2)
        
        Also handles non-standard functors used directly as relations:
            on(pan, stove).
            near(water, strip).
            hot(pan).          ← unary, treated as semantic type
        """
        args = pred.arguments

        if pred.functor in self._FUNCTOR_MAP:
            # Standard relation(id1, rel, id2) format
            if len(args) >= 3:
                scene.explicit_relations.append(ExplicitRelation(
                    subject_id=str(args[0]),
                    relation=str(args[1]),
                    object_id=str(args[2]),
                    confidence=pred.confidence,
                ))
            return

        # Non-standard functor used as the relation name itself
        # e.g., on(pan, stove) → relation between pan and stove
        if len(args) == 2:
            # Binary: functor(subject, object) → explicit relation
            scene.explicit_relations.append(ExplicitRelation(
                subject_id=str(args[0]),
                relation=pred.functor,
                object_id=str(args[1]),
                confidence=pred.confidence,
            ))
        elif len(args) == 1:
            # Unary: functor(object) → semantic type
            # e.g., hot(pan) → pan has semantic type "hot"
            oid = str(args[0])
            if oid not in scene.objects:
                scene.objects[oid] = SceneObject(
                    obj_id=oid, category="unknown"
                )
            scene.objects[oid].semantic_types.add(pred.functor)

    def _integrate_state(self, pred: RawPredicate, scene: ParsedScene):
        """state(ID, StateName, Value)"""
        args = pred.arguments
        if len(args) < 3:
            scene.parse_warnings.append(
                f"state predicate needs 3 args: {pred}"
            )
            return

        oid = str(args[0])
        state_name = str(args[1])
        value = args[2]

        if oid not in scene.objects:
            scene.objects[oid] = SceneObject(
                obj_id=oid, category="unknown"
            )

        scene.objects[oid].states[state_name] = value

    def _integrate_action(self, pred: RawPredicate, scene: ParsedScene):
        """action(AgentID, ActionName, TargetID)"""
        args = pred.arguments
        if len(args) >= 3:
            # Actions become relations
            scene.explicit_relations.append(ExplicitRelation(
                subject_id=str(args[0]),
                relation=str(args[1]),
                object_id=str(args[2]),
                confidence=pred.confidence,
            ))


# ═══════════════════════════════════════════════════════════
# Part 3: Spatial Relation Inference (Differentiable)
# ═══════════════════════════════════════════════════════════

class SpatialRelationConfig:
    """
    Configuration for spatial relation inference thresholds.
    
    All distances are in world-coordinate units (meters).
    Probabilities are computed via differentiable sigmoid kernels,
    NOT hard thresholds, to allow gradient flow.
    """
    def __init__(
        self,
        near_radius: float = 0.5,
        near_sharpness: float = 8.0,
        touching_radius: float = 0.15,
        touching_sharpness: float = 20.0,
        above_height_thresh: float = 0.3,
        above_sharpness: float = 10.0,
        above_horizontal_max: float = 0.3,
    ):
        # near(X, Y): sigmoid kernel around distance = near_radius
        self.near_radius = near_radius
        self.near_sharpness = near_sharpness

        # touching(X, Y): very close proximity
        self.touching_radius = touching_radius
        self.touching_sharpness = touching_sharpness

        # above(X, Y): X is higher than Y
        self.above_height_thresh = above_height_thresh
        self.above_sharpness = above_sharpness
        self.above_horizontal_max = above_horizontal_max


class SpatialRelationInferrer(nn.Module):
    """
    Infers spatial relations from 3D positions using differentiable kernels.
    
    Key insight: Instead of hard distance thresholds (which kill gradients),
    we use sigmoid-based soft thresholds:
    
        P(near(X,Y)) = σ( sharpness * (radius - distance(X,Y)) )
    
    When distance < radius → positive input → P ≈ 1
    When distance > radius → negative input → P ≈ 0
    The transition is smooth and differentiable.
    
    This means position perturbations from the VLM can propagate gradients
    through spatial relation probabilities to the final risk score.
    """

    def __init__(self, config: Optional[SpatialRelationConfig] = None):
        super().__init__()
        self.config = config or SpatialRelationConfig()

        # Learnable sharpness parameters (temperature for each relation)
        self.near_sharpness = nn.Parameter(
            torch.tensor(self.config.near_sharpness)
        )
        self.touching_sharpness = nn.Parameter(
            torch.tensor(self.config.touching_sharpness)
        )
        self.above_sharpness = nn.Parameter(
            torch.tensor(self.config.above_sharpness)
        )

        # Learnable radius parameters
        self.near_radius = nn.Parameter(
            torch.tensor(self.config.near_radius)
        )
        self.touching_radius = nn.Parameter(
            torch.tensor(self.config.touching_radius)
        )

    def compute_distance(
        self,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euclidean distance between two 3D positions.
        
        Args:
            pos_a: tensor(3,) or tensor(batch, 3)
            pos_b: tensor(3,) or tensor(batch, 3)
        Returns:
            distance: tensor(,) or tensor(batch,)
        """
        return torch.norm(pos_a - pos_b, dim=-1)

    def compute_horizontal_distance(
        self,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """Horizontal (XZ-plane) distance, ignoring height."""
        diff = pos_a - pos_b
        # Assuming Y is up: horizontal = sqrt(x² + z²)
        return torch.norm(diff[..., [0, 2]], dim=-1)

    def infer_near(
        self,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        P(near(A, B)) via differentiable sigmoid kernel.
        
        P = σ(sharpness * (radius - dist))
        """
        dist = self.compute_distance(pos_a, pos_b)
        # Clamp sharpness to positive range
        sharp = self.near_sharpness.clamp(min=1.0)
        radius = self.near_radius.clamp(min=0.05)
        return torch.sigmoid(sharp * (radius - dist))

    def infer_touching(
        self,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """P(touching(A, B)) — very close proximity."""
        dist = self.compute_distance(pos_a, pos_b)
        sharp = self.touching_sharpness.clamp(min=1.0)
        radius = self.touching_radius.clamp(min=0.01)
        return torch.sigmoid(sharp * (radius - dist))

    def infer_above(
        self,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        P(above(A, B)) — A is higher than B AND horizontally close.
        
        P = σ(sharp * (height_diff - thresh)) * σ(sharp * (max_horiz - horiz_dist))
        """
        # Height difference: positive means A is higher
        height_diff = pos_a[..., 1] - pos_b[..., 1]
        horiz_dist = self.compute_horizontal_distance(pos_a, pos_b)

        sharp = self.above_sharpness.clamp(min=1.0)

        p_higher = torch.sigmoid(
            sharp * (height_diff - self.config.above_height_thresh)
        )
        p_aligned = torch.sigmoid(
            sharp * (self.config.above_horizontal_max - horiz_dist)
        )

        return p_higher * p_aligned

    def infer_all_spatial(
        self,
        positions: Dict[str, torch.Tensor],
        object_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Infer all spatial relations for all object pairs.
        
        Args:
            positions: {obj_id: tensor(3,)} 3D positions
            object_pairs: specific pairs to compute (default: all pairs)
            
        Returns:
            {
                "near(1,2)": tensor(prob),
                "touching(1,2)": tensor(prob),
                "above(1,2)": tensor(prob),
                ...
            }
        """
        obj_ids = list(positions.keys())
        if object_pairs is None:
            object_pairs = [
                (a, b) for a in obj_ids for b in obj_ids if a != b
            ]

        relations = {}
        for id_a, id_b in object_pairs:
            if id_a not in positions or id_b not in positions:
                continue

            pos_a = positions[id_a]
            pos_b = positions[id_b]

            relations[f"near({id_a},{id_b})"] = self.infer_near(pos_a, pos_b)
            relations[f"touching({id_a},{id_b})"] = self.infer_touching(pos_a, pos_b)
            relations[f"above({id_a},{id_b})"] = self.infer_above(pos_a, pos_b)

        return relations


# Part 4: Semantic Type → Neural Predicate Mapping

class SemanticTypeMapper:
    """
    Maps VLM semantic type labels to neural predicate names used
    by the knowledge base / logic layer.
    
    The VLM might output diverse labels:
        "flammable", "inflammable", "combustible", "burns_easily"
    All should map to the neural predicate: flammable(X)
    
    This mapper handles synonyms, hierarchical types, and
    confidence calibration for the type → predicate mapping.
    """

    # Default mapping: {semantic_type: (predicate_name, base_confidence)}
    DEFAULT_TYPE_MAP: Dict[str, Tuple[str, float]] = {
        # Fire-related
        "flammable":     ("flammable", 0.95),
        "inflammable":   ("flammable", 0.95),
        "combustible":   ("flammable", 0.90),
        "burns_easily":  ("flammable", 0.85),
        "fire_source":   ("is_fire", 1.0),
        "fire":          ("is_fire", 1.0),
        "flame":         ("is_fire", 0.95),
        "ignition":      ("is_fire", 0.90),
        "hot":           ("is_fire", 0.70),

        # Sharp / cutting
        "sharp":         ("sharp", 0.95),
        "blade":         ("sharp", 0.90),
        "pointed":       ("sharp", 0.80),
        "cutting_tool":  ("sharp", 0.85),
        "edged":         ("sharp", 0.85),

        # Liquid
        "liquid":        ("liquid", 0.95),
        "fluid":         ("liquid", 0.90),
        "water":         ("liquid", 0.95),
        "wet":           ("liquid", 0.70),
        "spill":         ("liquid", 0.80),

        # Electrical
        "electrical":    ("is_electrical", 0.95),
        "electric":      ("is_electrical", 0.95),
        "powered":       ("is_electrical", 0.90),
        "electronic":    ("is_electrical", 0.85),
        "wired":         ("is_electrical", 0.80),
        "power_strip":   ("is_electrical", 0.95),

        # Toxic / chemical
        "toxic":         ("toxic", 0.95),
        "poisonous":     ("toxic", 0.90),
        "chemical":      ("toxic", 0.75),
        "hazardous":     ("toxic", 0.80),
        "corrosive":     ("toxic", 0.85),

        # Fragile
        "fragile":       ("fragile", 0.95),
        "brittle":       ("fragile", 0.90),
        "breakable":     ("fragile", 0.85),
        "glass":         ("fragile", 0.80),
        "ceramic":       ("fragile", 0.75),

        # Heavy
        "heavy":         ("heavy", 0.90),
        "massive":       ("heavy", 0.85),
        "dense":         ("heavy", 0.75),

        # Person types
        "person":        ("is_person", 0.95),
        "human":         ("is_person", 0.95),
        "adult":         ("is_person", 0.90),
        "child":         ("is_child", 0.95),
        "infant":        ("is_child", 0.95),
        "toddler":       ("is_child", 0.90),
        "kid":           ("is_child", 0.85),
        "elderly":       ("is_person", 0.90),

        # Container
        "container":     ("is_container", 0.90),
        "box":           ("is_container", 0.80),
        "bottle":        ("is_container", 0.85),
        "sealed":        ("is_container", 0.80),
        "enclosed":      ("is_container", 0.85),

        # Motion
        "running":       ("running", 0.90),
        "moving_fast":   ("running", 0.85),
        "rushing":       ("running", 0.80),
    }

    def __init__(
        self,
        type_map: Optional[Dict[str, Tuple[str, float]]] = None,
    ):
        self.type_map = type_map or dict(self.DEFAULT_TYPE_MAP)

    def register_mapping(
        self,
        semantic_type: str,
        predicate_name: str,
        confidence: float = 0.90,
    ):
        """Add or override a type mapping."""
        self.type_map[semantic_type.lower()] = (predicate_name, confidence)

    def map_types(
        self,
        semantic_types: Set[str],
        detection_confidence: float = 1.0,
    ) -> Dict[str, float]:
        """
        Map a set of semantic types to neural predicate probabilities.
        
        Args:
            semantic_types: {"flammable", "liquid"}
            detection_confidence: VLM's confidence in the object detection
            
        Returns:
            {"flammable": 0.95, "liquid": 0.95}  (predicate_name → probability)
        """
        predicates = {}
        for stype in semantic_types:
            key = stype.lower().strip()
            if key in self.type_map:
                pred_name, base_conf = self.type_map[key]
                # Combined confidence = base × detection
                combined = base_conf * detection_confidence
                # Keep highest confidence if multiple types map to same predicate
                if pred_name not in predicates or combined > predicates[pred_name]:
                    predicates[pred_name] = combined
            # else: unknown type, skip (could log warning)

        return predicates

    def map_category(self, category: str) -> Dict[str, float]:
        """
        Map object category to default semantic predicates.
        
        Categories like "knife" → {"sharp": 0.90}
        Categories like "child" → {"is_child": 0.95, "is_person": 0.95}
        """
        return self.map_types({category.lower()})


class SceneGraphBuilder(nn.Module):
    """
    Main module: Constructs a grounded fact probability dictionary
    from parsed neural predicates, ready for the DifferentiableLogicLayer.
    
    This is the central bridge between:
        VLM output (text predicates) ←→ Logic layer (tensor probabilities)
    
    Pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: Raw Prolog-style predicates from VLM               │
    │    object(1, gasoline_can, flammable).                     │
    │    position(1, 1.2, 0.5, 0.8).                             │
    │    position(2, 1.0, 0.5, 0.8).                             │
    └──────────────────┬──────────────────────────────────────────┘
                       │ NeuralPredicateParser.parse()
                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  ParsedScene                                               │
    │    objects: {1: SceneObject(gasoline_can, {flammable})}    │
    │    positions: {1: (1.2, 0.5, 0.8), 2: (1.0, 0.5, 0.8)}   │
    └──────────────────┬──────────────────────────────────────────┘
                       │ SemanticTypeMapper + SpatialRelationInferrer
                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  Grounded Fact Probabilities (Dict[str, Tensor])           │
    │    "flammable(1)":      tensor(0.95)                       │
    │    "is_fire(2)":        tensor(1.00)                       │
    │    "near(1,2)":         tensor(0.92)   ← from positions   │
    │    "on(pan,stove)":     tensor(0.90)   ← explicit rel     │
    └──────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
              DifferentiableLogicLayer.forward()
    """

    def __init__(
        self,
        spatial_config: Optional[SpatialRelationConfig] = None,
        type_mapper: Optional[SemanticTypeMapper] = None,
        require_grad_positions: bool = True,
    ):
        """
        Args:
            spatial_config: Configuration for spatial relation kernels
            type_mapper: Semantic type → predicate mapper
            require_grad_positions: If True, 3D positions are torch tensors
                with requires_grad=True, enabling gradient flow from
                the logic layer back to position estimates.
        """
        super().__init__()
        self.spatial_inferrer = SpatialRelationInferrer(spatial_config)
        self.type_mapper = type_mapper or SemanticTypeMapper()
        self.require_grad_positions = require_grad_positions

    def build(
        self,
        predicate_text: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Full pipeline: text predicates → grounded fact probabilities.
        
        Args:
            predicate_text: Multi-line Prolog-style predicates from VLM
            device: torch device for output tensors
            
        Returns:
            grounded_facts: {
                "near(1,2)": tensor(prob),
                "flammable(1)": tensor(prob),
                ...
            }
            entity_ids: ["1", "2", "3", ...]
        """
        # Step 1: Parse text → structured scene
        parser = NeuralPredicateParser()
        parsed = parser.parse(predicate_text)
        return self.build_from_parsed(parsed, device)

    def build_from_parsed(
        self,
        parsed: ParsedScene,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Build grounded facts from an already-parsed scene."""
        grounded: Dict[str, torch.Tensor] = {}
        entity_ids = parsed.object_ids

        # ── Step 2: Ground semantic types as unary predicates ──
        for oid, obj in parsed.objects.items():
            # Map semantic types to predicate probabilities
            type_preds = self.type_mapper.map_types(
                obj.semantic_types, obj.detection_confidence
            )
            # Also map category name itself
            cat_preds = self.type_mapper.map_category(obj.category)

            # Merge (keep max confidence)
            all_preds = dict(cat_preds)
            for pred_name, prob in type_preds.items():
                if pred_name not in all_preds or prob > all_preds[pred_name]:
                    all_preds[pred_name] = prob

            # Create tensor entries
            for pred_name, prob in all_preds.items():
                key = f"{pred_name}({oid})"
                grounded[key] = torch.tensor(
                    prob, dtype=torch.float32, device=device
                )

        # ── Step 3: Infer spatial relations from 3D positions ──
        positions: Dict[str, torch.Tensor] = {}
        for oid, obj in parsed.objects.items():
            if obj.position_3d is not None:
                pos = torch.tensor(
                    obj.position_3d,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=self.require_grad_positions,
                )
                positions[oid] = pos

        if len(positions) >= 2:
            spatial_rels = self.spatial_inferrer.infer_all_spatial(positions)
            grounded.update(spatial_rels)

        # ── Step 4: Add explicit relations (from VLM direct output) ──
        for rel in parsed.explicit_relations:
            key = f"{rel.relation}({rel.subject_id},{rel.object_id})"
            prob = torch.tensor(
                rel.confidence, dtype=torch.float32, device=device
            )
            # If spatial inference also produced this relation,
            # take the max (VLM explicit > inferred)
            if key in grounded:
                grounded[key] = torch.max(grounded[key], prob)
            else:
                grounded[key] = prob

        # ── Step 5: Ground attribute-derived predicates ──
        for oid, obj in parsed.objects.items():
            attr_preds = self._attributes_to_predicates(obj)
            for pred_name, prob in attr_preds.items():
                key = f"{pred_name}({oid})"
                # Only set if not already set by semantic types
                if key not in grounded:
                    grounded[key] = torch.tensor(
                        prob, dtype=torch.float32, device=device
                    )

        return grounded, entity_ids

    def build_from_dict(
        self,
        data: Dict[str, List],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Build from structured dictionary format."""
        parser = NeuralPredicateParser()
        parsed = parser.parse_dict_format(data)
        return self.build_from_parsed(parsed, device)

    def _attributes_to_predicates(
        self, obj: SceneObject
    ) -> Dict[str, float]:
        """
        Derive predicate probabilities from numeric attributes.
        
        E.g., temperature=150 → is_fire probability via sigmoid:
            P(is_fire) = σ((temperature - threshold) / scale)
        
        This allows continuous attribute values to contribute to
        discrete predicate probabilities in a differentiable way.
        """
        preds: Dict[str, float] = {}
        attrs = obj.attributes

        # Temperature → fire risk
        if "temperature" in attrs:
            temp = attrs["temperature"]
            # P(hot) starts rising above 80°C, saturates at ~200°C
            p_hot = 1.0 / (1.0 + math.exp(-(temp - 80) / 30))
            if p_hot > 0.1:
                preds["is_fire"] = min(p_hot, 0.99)

        # Weight → heavy
        if "weight" in attrs:
            weight = attrs["weight"]
            # P(heavy) rises above 10kg
            p_heavy = 1.0 / (1.0 + math.exp(-(weight - 10) / 5))
            if p_heavy > 0.1:
                preds["heavy"] = min(p_heavy, 0.99)

        # Sharpness → sharp
        if "sharpness" in attrs:
            sharp = attrs["sharpness"]
            # Direct mapping: 0-1 scale
            if sharp > 0.3:
                preds["sharp"] = min(sharp, 0.99)

        # Toxicity → toxic
        if "toxicity" in attrs:
            tox = attrs["toxicity"]
            if tox > 0.2:
                preds["toxic"] = min(tox, 0.99)

        # Fragility → fragile
        if "fragility" in attrs:
            frag = attrs["fragility"]
            if frag > 0.3:
                preds["fragile"] = min(frag, 0.99)

        # Flammability → flammable
        if "flammability" in attrs:
            flam = attrs["flammability"]
            if flam > 0.2:
                preds["flammable"] = min(flam, 0.99)

        # Conductivity → is_electrical
        if "conductivity" in attrs:
            cond = attrs["conductivity"]
            if cond > 0.5:
                preds["is_electrical"] = min(cond, 0.99)

        # Velocity → running
        if "velocity" in attrs:
            vel = attrs["velocity"]
            # P(running) rises above 1.5 m/s
            p_run = 1.0 / (1.0 + math.exp(-(vel - 1.5) / 0.5))
            if p_run > 0.1:
                preds["running"] = min(p_run, 0.99)

        return preds

    def get_positions_tensor(
        self, parsed: ParsedScene, device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        """
        Extract 3D positions as differentiable tensors.
        Useful for connecting to a VLM that outputs position estimates
        as part of a larger differentiable pipeline.
        """
        positions = {}
        for oid, obj in parsed.objects.items():
            if obj.position_3d is not None:
                positions[oid] = torch.tensor(
                    obj.position_3d,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=self.require_grad_positions,
                )
        return positions

# Part 6: Visualization & Debugging

class SceneGraphVisualizer:
    """
    Generates human-readable visualizations of the scene graph
    for debugging and explainability.
    """

    @staticmethod
    def to_ascii(
        grounded_facts: Dict[str, torch.Tensor],
        entity_ids: List[str],
        threshold: float = 0.3,
    ) -> str:
        """
        Generate ASCII visualization of the scene graph.
        
        Shows only facts above the probability threshold.
        """
        lines = []
        lines.append("╔══════════════════════════════════════════╗")
        lines.append("║         GROUNDED SCENE GRAPH             ║")
        lines.append("╠══════════════════════════════════════════╣")

        # Group by predicate type
        unary = {}    # pred_name → [(obj_id, prob)]
        binary = {}   # pred_name → [(subj, obj, prob)]

        for key, prob_tensor in sorted(grounded_facts.items()):
            prob = prob_tensor.item() if isinstance(prob_tensor, torch.Tensor) else prob_tensor
            if prob < threshold:
                continue

            # Parse key: "pred_name(arg1)" or "pred_name(arg1,arg2)"
            paren = key.index("(")
            pred_name = key[:paren]
            args = key[paren + 1:-1].split(",")

            if len(args) == 1:
                unary.setdefault(pred_name, []).append((args[0], prob))
            elif len(args) == 2:
                binary.setdefault(pred_name, []).append(
                    (args[0], args[1], prob)
                )

        # Print unary predicates (entity properties)
        if unary:
            lines.append("║  Entity Properties:                      ║")
            lines.append("║  ─────────────────                        ║")
            for pred_name, entries in sorted(unary.items()):
                for obj_id, prob in entries:
                    bar = "█" * int(prob * 10)
                    lines.append(
                        f"║  {pred_name}({obj_id})"
                        f"{'': <20} {bar} {prob:.3f}   ║"
                    )

        # Print binary predicates (relations)
        if binary:
            lines.append("║                                          ║")
            lines.append("║  Relations:                               ║")
            lines.append("║  ──────────                                ║")
            for pred_name, entries in sorted(binary.items()):
                for subj, obj, prob in entries:
                    bar = "█" * int(prob * 10)
                    lines.append(
                        f"║  {subj} ──{pred_name}──▶ {obj}"
                        f"{'': <10} {bar} {prob:.3f}   ║"
                    )

        lines.append("╚══════════════════════════════════════════╝")
        return "\n".join(lines)

    @staticmethod
    def to_prolog(
        grounded_facts: Dict[str, torch.Tensor],
        threshold: float = 0.1,
    ) -> str:
        """Export grounded facts as Prolog-style probability annotations."""
        lines = ["%% Grounded Scene Graph (probabilities)"]
        for key, prob_tensor in sorted(grounded_facts.items()):
            prob = prob_tensor.item() if isinstance(prob_tensor, torch.Tensor) else prob_tensor
            if prob < threshold:
                continue
            lines.append(f"{prob:.4f}::{key}.")
        return "\n".join(lines)

    @staticmethod
    def to_adjacency_dict(
        grounded_facts: Dict[str, torch.Tensor],
        threshold: float = 0.3,
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Convert to adjacency representation for graph algorithms.
        
        Returns:
            {
                "1": {
                    "outgoing": [("near", "2", 0.92), ...],
                    "properties": [("flammable", 0.95), ...],
                },
                ...
            }
        """
        graph: Dict[str, Dict[str, list]] = {}

        for key, prob_tensor in grounded_facts.items():
            prob = prob_tensor.item() if isinstance(prob_tensor, torch.Tensor) else prob_tensor
            if prob < threshold:
                continue

            paren = key.index("(")
            pred_name = key[:paren]
            args = key[paren + 1:-1].split(",")

            if len(args) == 1:
                oid = args[0]
                graph.setdefault(oid, {"outgoing": [], "properties": []})
                graph[oid]["properties"].append((pred_name, prob))
            elif len(args) == 2:
                subj, obj = args
                graph.setdefault(subj, {"outgoing": [], "properties": []})
                graph[subj]["outgoing"].append((pred_name, obj, prob))

        return graph