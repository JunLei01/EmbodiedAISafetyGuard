# """
# ProcTHOR-10k Dataset Adapter
# ==============================
# Bridges ProcTHOR-10k ai2thor scenes to the Neuro-Symbolic Safety Guardrail pipeline.

# Complete Pipeline Integration:
#     ProcTHOR Scene → Scene Graph → Logical Predicates → Differentiable Logic Reasoning
#     → Safety Rule Activation → Action Violation Checking

# Usage:
#     import prior
#     dataset = prior.load_dataset("procthor-10k")

#     adapter = ProcTHORAdapter(dataset["train"])
#     scene = adapter.load_scene(scene_id="train_0")

#     # Run full neuro-symbolic pipeline
#     pipeline_info, safety_result = run_neuro_symbolic_pipeline(scene)

#     # Check actions
#     violations = safety_result.check_action("move_towards", "obj_0", "obj_1")
# """

# import os
# import json
# import numpy as np
# import torch
# from typing import Dict, List, Tuple, Optional, Any, Set
# from dataclasses import dataclass, field
# from pathlib import Path
# from collections import defaultdict

# from utils.scene_graph_builder import (
#     SceneGraphBuilder,
#     SpatialRelationConfig,
#     ParsedScene,
#     SceneObject,
#     ExplicitRelation,
# )
# from utils.differentiable_safety_reasoner import DifferentiableSafetyReasoner


# # ProcTHOR object category to safety semantic types mapping
# PROCTHOR_SEMANTIC_MAPPING = {
#     # Fire hazards
#     "stove": ["fire_source", "hot", "electrical"],
#     "oven": ["fire_source", "hot", "electrical"],
#     "microwave": ["fire_source", "electrical"],
#     "toaster": ["fire_source", "hot", "electrical"],
#     "candle": ["fire_source", "flammable"],
#     "fireplace": ["fire_source", "hot"],
#     "lighter": ["fire_source", "flammable"],
#     "match": ["fire_source", "flammable"],
#     "gas_can": ["flammable", "liquid"],
#     "gasoline_can": ["flammable", "liquid"],
#     "oil": ["flammable", "liquid"],

#     # Electrical hazards
#     "television": ["electrical", "is_electrical", "fragile"],
#     "tv": ["electrical", "is_electrical", "fragile"],
#     "computer": ["electrical", "is_electrical", "fragile"],
#     "monitor": ["electrical", "is_electrical", "fragile"],
#     "laptop": ["electrical", "is_electrical", "fragile"],
#     "phone": ["electrical", "is_electrical"],
#     "refrigerator": ["electrical", "is_electrical", "heavy"],
#     "dishwasher": ["electrical", "is_electrical"],
#     "washing_machine": ["electrical", "is_electrical", "is_wet"],
#     "dryer": ["electrical", "is_electrical", "hot"],
#     "power_strip": ["electrical", "is_electrical"],
#     "outlet": ["electrical", "is_electrical"],
#     "extension_cord": ["electrical", "is_electrical"],
#     "fan": ["electrical", "is_electrical"],
#     "printer": ["electrical", "is_electrical"],

#     # Sharp objects
#     "knife": ["sharp", "weapon"],
#     "scissors": ["sharp"],
#     "razor": ["sharp"],
#     "glass": ["sharp", "fragile", "glass"],
#     "bottle": ["sharp", "fragile", "glass"],
#     "wine_glass": ["fragile", "glass"],
#     "mirror": ["sharp", "fragile", "glass"],
#     "window": ["fragile", "glass"],
#     "plate": ["fragile", "breakable"],

#     # Chemical/toxic hazards
#     "bleach": ["chemical", "toxic", "liquid"],
#     "cleaning_solution": ["chemical", "toxic", "liquid"],
#     "detergent": ["chemical", "toxic", "liquid"],
#     "medicine": ["chemical", "toxic"],
#     "pills": ["chemical", "toxic"],
#     "pesticide": ["chemical", "toxic"],
#     "paint": ["chemical", "toxic", "liquid"],
#     "solvent": ["chemical", "toxic", "liquid", "flammable"],

#     # Heavy objects (crush hazard)
#     "book": ["heavy"],
#     "bookshelf": ["heavy", "furniture"],
#     "cabinet": ["heavy", "furniture"],
#     "dresser": ["heavy", "furniture"],
#     "wardrobe": ["heavy", "furniture"],
#     "tv_stand": ["heavy", "furniture"],
#     "desk": ["heavy", "furniture"],
#     "table": ["heavy", "furniture"],
#     "bathtub": ["heavy"],
#     "sink": ["heavy", "is_wet"],

#     # Slip/fall hazards
#     "rug": ["slip_hazard"],
#     "mat": ["slip_hazard", "on_floor"],
#     "stairs": ["fall_hazard"],
#     "shower": ["slip_hazard", "is_wet"],
#     "faucet": ["is_wet"],
#     "water": ["liquid", "is_wet", "slip_hazard"],
#     "puddle": ["liquid", "is_wet", "slip_hazard", "on_floor"],

#     # People (child safety)
#     "person": ["is_person"],
#     "child": ["is_person", "is_child"],
#     "baby": ["is_person", "is_child"],
#     "infant": ["is_person", "is_child"],
#     "elderly": ["is_person", "is_elderly"],

#     # Containers
#     "cup": ["is_container"],
#     "bowl": ["is_container"],
#     "pot": ["is_container", "hot"],
#     "pan": ["is_container", "hot"],
#     "box": ["is_container"],
#     "drawer": ["is_container", "locked"],
#     "safe": ["is_container", "locked"],

#     # Choking hazards (small objects)
#     "pen": ["small"],
#     "pencil": ["small"],
#     "coin": ["small", "choking"],
#     "button": ["small", "choking"],
#     "battery": ["small", "chemical", "toxic", "choking"],
#     "magnet": ["small", "choking"],

#     # General furniture (neutral)
#     "chair": ["furniture"],
#     "bed": ["furniture"],
#     "couch": ["furniture"],
#     "sofa": ["furniture"],
#     "shelf": ["furniture"],
#     "counter": ["furniture"],
#     "countertop": ["furniture"],
# }


# @dataclass
# class ProcTHORObject:
#     """Object from ProcTHOR scene."""
#     obj_id: str
#     category: str
#     position_3d: Tuple[float, float, float]  # (x, y, z)
#     rotation: Tuple[float, float, float]  # (pitch, yaw, roll)
#     bbox_3d: Optional[List[float]] = None  # [min_x, min_y, min_z, max_x, max_y, max_z]
#     attributes: Dict[str, Any] = field(default_factory=dict)

#     @property
#     def semantic_types(self) -> List[str]:
#         """Get safety-relevant semantic types based on category."""
#         category_lower = self.category.lower().replace(" ", "_")
#         return PROCTHOR_SEMANTIC_MAPPING.get(category_lower, [])


# @dataclass
# class ProcTHORScene:
#     """Complete loaded scene from ProcTHOR dataset."""
#     scene_id: str
#     room_type: str  # kitchen, bedroom, living room, bathroom, etc.
#     description: str

#     # Room structure
#     rooms: List[Dict[str, Any]] = field(default_factory=list)
#     walls: List[Dict[str, Any]] = field(default_factory=list)
#     floors: List[Dict[str, Any]] = field(default_factory=list)

#     # Objects
#     objects: List[ProcTHORObject] = field(default_factory=list)

#     # Agent information (if available)
#     agent_position: Optional[Tuple[float, float, float]] = None
#     agent_rotation: Optional[float] = None

#     @property
#     def num_objects(self) -> int:
#         return len(self.objects)

#     @property
#     def object_categories(self) -> Set[str]:
#         return set(obj.category for obj in self.objects)

#     @property
#     def hazard_objects(self) -> List[ProcTHORObject]:
#         """Return objects that are potential safety hazards."""
#         hazards = []
#         for obj in self.objects:
#             types = obj.semantic_types
#             if any(t in ["sharp", "hot", "fire_source", "chemical", "toxic", "electrical", "flammable"]
#                    for t in types):
#                 hazards.append(obj)
#         return hazards

#     def get_objects_by_type(self, semantic_type: str) -> List[ProcTHORObject]:
#         """Find all objects with a specific semantic type."""
#         return [obj for obj in self.objects if semantic_type in obj.semantic_types]

#     def summary(self) -> str:
#         lines = [
#             f"ProcTHORScene: {self.scene_id}",
#             f"  Room type: {self.room_type}",
#             f"  Description: {self.description}",
#             f"  Objects: {self.num_objects}",
#             f"  Categories: {', '.join(sorted(self.object_categories))}",
#             f"  Potential hazards: {len(self.hazard_objects)}",
#         ]
#         for obj in self.objects[:10]:  # Show first 10
#             types_str = ", ".join(obj.semantic_types[:3]) if obj.semantic_types else "none"
#             pos = obj.position_3d
#             lines.append(
#                 f"    [{obj.obj_id}] {obj.category} "
#                 f"types={{{types_str}}} "
#                 f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
#             )
#         if self.num_objects > 10:
#             lines.append(f"    ... and {self.num_objects - 10} more objects")
#         return "\n".join(lines)


# class ProcTHORAdapter:
#     """
#     Loads ProcTHOR-10k scenes and converts them to the safety pipeline format.

#     Usage:
#         import prior
#         dataset = prior.load_dataset("procthor-10k")

#         adapter = ProcTHORAdapter(dataset["train"])
#         scene = adapter.load_scene(scene_id="train_0")

#         # Convert to predicates
#         predicates = adapter.to_predicates(scene)

#         # Or get grounded facts directly
#         grounded_facts, entity_ids = adapter.to_grounded_facts(scene)
#     """

#     def __init__(self, dataset_split):
#         """
#         Args:
#             dataset_split: A dataset split from prior.load_dataset()
#                           (e.g., dataset["train"], dataset["val"])
#         """
#         self.dataset = dataset_split
#         # Handle various ID field names used by different ProcTHOR dataset versions
#         self._scene_map = {}
#         for i, scene in enumerate(dataset_split):
#             scene_id = None
#             if isinstance(scene, dict):
#                 for key in ["id", "scene_id", "name", "unique_id", "scene_name", "sceneId"]:
#                     if key in scene:
#                         scene_id = scene[key]
#                         break
#             # Fallback: use index if no ID found
#             if scene_id is None:
#                 scene_id = f"scene_{i}"
#             self._scene_map[str(scene_id)] = scene
#         print(f"ProcTHORAdapter: loaded {len(self.dataset)} scenes")

#     def list_scenes(self) -> List[str]:
#         """List all available scene IDs."""
#         return list(self._scene_map.keys())

#     def load_scene(self, scene_id: str) -> ProcTHORScene:
#         """
#         Load a ProcTHOR scene by ID.

#         Args:
#             scene_id: Scene identifier (e.g., "train_0", "val_42")

#         Returns:
#             ProcTHORScene with all objects and metadata
#         """
#         if scene_id not in self._scene_map:
#             raise KeyError(f"Scene {scene_id} not found in dataset")

#         raw_scene = self._scene_map[scene_id]

#         # Determine room type from objects or rooms
#         room_type = self._infer_room_type(raw_scene)

#         scene = ProcTHORScene(
#             scene_id=scene_id,
#             room_type=room_type,
#             description=raw_scene.get("description", f"ProcTHOR {room_type}"),
#         )

#         # Load rooms structure if available
#         if "rooms" in raw_scene:
#             scene.rooms = raw_scene["rooms"]
#         if "walls" in raw_scene:
#             scene.walls = raw_scene["walls"]
#         if "floors" in raw_scene:
#             scene.floors = raw_scene["floors"]

#         # Load objects
#         objects_data = raw_scene.get("objects", [])
#         for i, obj_data in enumerate(objects_data):
#             obj = self._parse_object(obj_data, obj_id=f"obj_{i}")
#             scene.objects.append(obj)

#         # Agent position if available
#         if "agent" in raw_scene:
#             agent_data = raw_scene["agent"]
#             if "position" in agent_data:
#                 scene.agent_position = tuple(agent_data["position"])
#             if "rotation" in agent_data:
#                 scene.agent_rotation = agent_data["rotation"]

#         return scene

#     def to_predicates(self, scene: ProcTHORScene) -> str:
#         """
#         Convert scene to Prolog-style predicates.

#         Output format:
#             % ProcTHOR scene: train_0
#             object(obj_0, stove, [fire_source, hot, electrical]).
#             position(obj_0, 2.5, 0.8, 1.2).
#             ...
#         """
#         lines = [
#             f"% ProcTHOR scene: {scene.scene_id}",
#             f"% Room type: {scene.room_type}",
#             f"% Objects: {scene.num_objects}",
#             "",
#             "% ── Objects ──",
#         ]

#         for obj in scene.objects:
#             types = obj.semantic_types
#             if len(types) == 1:
#                 types_str = types[0]
#             elif len(types) > 1:
#                 types_str = "[" + ", ".join(types) + "]"
#             else:
#                 types_str = "unknown"

#             lines.append(f"object({obj.obj_id}, {obj.category}, {types_str}).")

#         # Positions
#         lines.append("")
#         lines.append("% ── 3D Positions (meters) ──")
#         for obj in scene.objects:
#             x, y, z = obj.position_3d
#             lines.append(f"position({obj.obj_id}, {x:.3f}, {y:.3f}, {z:.3f}).")

#         # Attributes (temperature, etc.)
#         lines.append("")
#         lines.append("% ── Attributes ──")
#         for obj in scene.objects:
#             types = obj.semantic_types
#             if "hot" in types or "fire_source" in types:
#                 lines.append(f"attribute({obj.obj_id}, temperature, 100).")
#             if "is_wet" in types:
#                 lines.append(f"attribute({obj.obj_id}, wet, 1).")
#             if "electrical" in types:
#                 lines.append(f"attribute({obj.obj_id}, powered, 1).")

#         return "\n".join(lines)

#     def to_grounded_facts(
#         self,
#         scene: ProcTHORScene,
#         spatial_config: Optional[SpatialRelationConfig] = None,
#         device: torch.device = torch.device("cpu"),
#     ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
#         """
#         Full conversion: scene -> grounded facts ready for the reasoner.
#         """
#         builder = SceneGraphBuilder(
#             spatial_config=spatial_config,
#             require_grad_positions=False,  # ProcTHOR positions are fixed
#         )
#         parsed = self._to_parsed_scene(scene)
#         return builder.build_from_parsed(parsed, device)

#     def _to_parsed_scene(self, scene: ProcTHORScene) -> ParsedScene:
#         """Convert to ParsedScene for SceneGraphBuilder."""
#         parsed = ParsedScene()

#         for obj in scene.objects:
#             parsed.objects[obj.obj_id] = SceneObject(
#                 obj_id=obj.obj_id,
#                 category=obj.category,
#                 semantic_types=set(obj.semantic_types),
#                 position_3d=obj.position_3d,
#                 attributes=obj.attributes.copy(),
#                 detection_confidence=1.0,  # Ground truth from simulation
#             )

#         return parsed

#     def _parse_object(self, obj_data: Dict, obj_id: str) -> ProcTHORObject:
#         """Parse a single object from ProcTHOR format."""
#         # Get position
#         position = obj_data.get("position", [0.0, 0.0, 0.0])
#         if isinstance(position, dict):
#             position = (position.get("x", 0), position.get("y", 0), position.get("z", 0))
#         else:
#             position = tuple(position[:3])

#         # Get rotation
#         rotation = obj_data.get("rotation", [0.0, 0.0, 0.0])
#         if isinstance(rotation, dict):
#             rotation = (rotation.get("x", 0), rotation.get("y", 0), rotation.get("z", 0))
#         else:
#             rotation = tuple(rotation[:3])

#         # Get bounding box if available
#         bbox = obj_data.get("bbox", None)
#         if bbox and isinstance(bbox, dict):
#             bbox = [
#                 bbox.get("min", {}).get("x", 0), bbox.get("min", {}).get("y", 0), bbox.get("min", {}).get("z", 0),
#                 bbox.get("max", {}).get("x", 0), bbox.get("max", {}).get("y", 0), bbox.get("max", {}).get("z", 0),
#             ]

#         obj = ProcTHORObject(
#             obj_id=obj_id,
#             category=obj_data.get("objectType", obj_data.get("type", "unknown")),
#             position_3d=position,
#             rotation=rotation,
#             bbox_3d=bbox,
#             attributes=obj_data.get("attributes", {}),
#         )

#         return obj

#     def _infer_room_type(self, raw_scene: Dict) -> str:
#         """Infer the primary room type from scene metadata."""
#         # Direct room type
#         if "roomType" in raw_scene:
#             return raw_scene["roomType"]

#         # Infer from rooms list
#         rooms = raw_scene.get("rooms", [])
#         if rooms:
#             room_types = [r.get("roomType", "").lower() for r in rooms]
#             # Count occurrences and pick most common
#             type_counts = {}
#             for rt in room_types:
#                 type_counts[rt] = type_counts.get(rt, 0) + 1
#             if type_counts:
#                 return max(type_counts, key=type_counts.get)

#         # Infer from objects
#         objects = raw_scene.get("objects", [])
#         obj_types = set(obj.get("objectType", "").lower() for obj in objects)

#         # Kitchen indicators
#         kitchen_objs = {"stove", "oven", "refrigerator", "sink", "countertop", "dishwasher"}
#         if kitchen_objs & obj_types:
#             return "kitchen"

#         # Bathroom indicators
#         bathroom_objs = {"toilet", "sink", "bathtub", "shower", "mirror"}
#         if bathroom_objs & obj_types:
#             return "bathroom"

#         # Bedroom indicators
#         bedroom_objs = {"bed", "nightstand", "dresser", "wardrobe"}
#         if bedroom_objs & obj_types:
#             return "bedroom"

#         # Living room indicators
#         living_objs = {"sofa", "couch", "tv", "television", "coffee_table"}
#         if living_objs & obj_types:
#             return "living_room"

#         return "unknown"

#     def sample_scenes_by_type(self, room_type: str, n: int = 5) -> List[str]:
#         """
#         Sample scene IDs of a specific room type.

#         Args:
#             room_type: e.g., "kitchen", "bedroom", "bathroom", "living_room"
#             n: Number of scenes to sample

#         Returns:
#             List of scene IDs
#         """
#         matching = []
#         for scene_id, raw_scene in self._scene_map.items():
#             if self._infer_room_type(raw_scene) == room_type.lower():
#                 matching.append(scene_id)

#         import random
#         random.seed(42)
#         return random.sample(matching, min(n, len(matching)))

#     def get_statistics(self) -> Dict[str, Any]:
#         """Get dataset statistics."""
#         stats = {
#             "total_scenes": len(self.dataset),
#             "room_types": defaultdict(int),
#             "total_objects": 0,
#             "object_categories": set(),
#             "hazard_counts": defaultdict(int),
#         }

#         for raw_scene in self.dataset:
#             room_type = self._infer_room_type(raw_scene)
#             stats["room_types"][room_type] += 1

#             objects = raw_scene.get("objects", [])
#             stats["total_objects"] += len(objects)

#             for obj in objects:
#                 category = obj.get("objectType", obj.get("type", "unknown"))
#                 stats["object_categories"].add(category)

#                 # Count potential hazards
#                 category_lower = category.lower().replace(" ", "_")
#                 semantic_types = PROCTHOR_SEMANTIC_MAPPING.get(category_lower, [])
#                 for hazard in ["sharp", "hot", "fire_source", "chemical", "toxic", "electrical", "flammable"]:
#                     if hazard in semantic_types:
#                         stats["hazard_counts"][hazard] += 1

#         stats["object_categories"] = sorted(stats["object_categories"])
#         return stats


# # ─── Neuro-Symbolic Pipeline Integration ───

# def run_neuro_symbolic_pipeline(
#     scene: ProcTHORScene,
#     activation_threshold: float = 0.3,
#     verbose: bool = False,
# ) -> Tuple[Dict, Any]:
#     """
#     Run the complete neuro-symbolic safety pipeline on a ProcTHOR scene.

#     Pipeline stages:
#     1. Scene Graph Construction - Parse ProcTHOR objects into scene graph
#     2. Logical Predicate Generation - Convert to Prolog-style predicates
#     3. Differentiable Logic Reasoning - Forward chaining with neural predicates
#     4. Safety Rule Extraction - Get activated safety rules with probabilities

#     Args:
#         scene: Loaded ProcTHOR scene
#         activation_threshold: Minimum probability to activate a rule
#         verbose: Print detailed output

#     Returns:
#         Tuple of (pipeline_info dict, SafetyReasoningResult)
#     """
#     adapter = ProcTHORAdapter([])  # Dummy adapter for helper methods

#     if verbose:
#         print(f"\n  [Stage 1/4] Scene Graph Construction")
#         print(f"    - Objects: {scene.num_objects}")
#         print(f"    - Room type: {scene.room_type}")

#     # Stage 1: Convert to grounded facts through SceneGraphBuilder
#     builder = SceneGraphBuilder(require_grad_positions=False)
#     parsed = adapter._to_parsed_scene(scene)
#     grounded_facts, entity_ids = builder.build_from_parsed(parsed)

#     # Count spatial relations
#     spatial_relations = [k for k in grounded_facts.keys() if any(
#         rel in k for rel in ['near(', 'touching(', 'above(', 'below(']
#     )]

#     if verbose:
#         print(f"    ✓ ParsedScene created with {len(parsed.objects)} objects")
#         print(f"\n  [Stage 2/4] Logical Predicate Generation")
#         print(f"    - Entity IDs: {entity_ids[:5]}..." if len(entity_ids) > 5 else f"    - Entity IDs: {entity_ids}")
#         print(f"    - Grounded facts: {len(grounded_facts)} predicates")
#         print(f"    - Spatial relations: {len(spatial_relations)} (near/touching/above/below)")

#     # Stage 3: Differentiable Safety Reasoning
#     if verbose:
#         print(f"\n  [Stage 3/4] Differentiable Logic Reasoning")
#         print(f"    - Loading safety knowledge base...")

#     reasoner = DifferentiableSafetyReasoner(activation_threshold=activation_threshold)

#     if verbose:
#         print(f"    - Running forward chaining inference with semiring semantics...")
#         print(f"      AND: product, OR: noisy-OR")

#     result = reasoner.reason(grounded_facts, entity_ids)

#     if verbose:
#         print(f"    ✓ Inference complete")
#         print(f"\n  [Stage 4/4] Safety Rule Extraction")
#         print(f"    - Total activated rules: {result.total_activated}")
#         print(f"    - Critical: {result.critical_count} | High: {result.high_count} | "
#               f"Medium: {result.medium_count} | Low: {result.low_count}")

#         if result.activated_rules:
#             print(f"\n    Top activated rules:")
#             for rule in result.activated_rules[:5]:
#                 desc = rule.natural_language[:50] + "..." if len(rule.natural_language) > 50 else rule.natural_language
#                 print(f"      [{rule.severity.name}] P={rule.probability:.3f} | {rule.rule_name}")
#                 print(f"        {desc}")

#     pipeline_info = {
#         "grounded_facts_count": len(grounded_facts),
#         "spatial_relations_count": len(spatial_relations),
#         "entity_ids": entity_ids,
#         "scene": scene,
#     }

#     return pipeline_info, result


# def run_safety_pipeline_procthor(
#     dataset,
#     scene_id: str,
#     activation_threshold: float = 0.3,
#     action: Optional[str] = None,
#     actor: Optional[str] = None,
#     target: Optional[str] = None,
#     verbose: bool = True,
# ) -> dict:
#     """
#     Run safety assessment on a ProcTHOR scene with full neuro-symbolic pipeline.

#     Args:
#         dataset: ProcTHOR dataset split from prior.load_dataset()
#         scene_id: Scene identifier
#         activation_threshold: Minimum probability to activate a rule
#         action: Action to check (optional)
#         actor: Actor entity ID or category (optional)
#         target: Target entity ID or category (optional)
#         verbose: Print detailed output

#     Returns:
#         dict with safety results including:
#         - scene: ProcTHORScene
#         - result: SafetyReasoningResult
#         - violations: List of violated rules
#         - is_safe: Boolean
#         - pipeline_info: Dict with grounded facts, spatial relations, etc.
#     """
#     # Step 1: Load scene
#     if verbose:
#         print("=" * 70)
#         print("  ProcTHOR-10k Neuro-Symbolic Safety Pipeline")
#         print("=" * 70)
#         print(f"\n[1/5] Loading ProcTHOR scene: {scene_id}")

#     adapter = ProcTHORAdapter(dataset)
#     scene = adapter.load_scene(scene_id)

#     if verbose:
#         print(f"  Room type: {scene.room_type}")
#         print(f"  Objects: {scene.num_objects}")
#         print(f"  Potential hazards: {len(scene.hazard_objects)}")

#     # Steps 2-4: Run neuro-symbolic pipeline
#     pipeline_info, result = run_neuro_symbolic_pipeline(
#         scene,
#         activation_threshold=activation_threshold,
#         verbose=verbose,
#     )

#     if verbose:
#         print(f"\n[5/5] Action violation checking...")

#     # Step 5: Action violation checking
#     violations = []
#     if action:
#         # Find matching entities
#         actor_id = None
#         target_id = None

#         for obj in scene.objects:
#             if actor and (actor.lower() in obj.category.lower() or obj.obj_id == actor):
#                 actor_id = obj.obj_id
#             if target and (target.lower() in obj.category.lower() or obj.obj_id == target):
#                 target_id = obj.obj_id

#         if actor_id:
#             violations = result.check_action(action, actor_id, target_id)
#             if verbose:
#                 if violations:
#                     print(f"  BLOCKED: {len(violations)} violations found!")
#                     for v in violations:
#                         print(f"    - {v.rule_name} ({v.severity.name}): {v.natural_language}")
#                 else:
#                     print(f"  ALLOWED: No safety violations found")
#         elif verbose:
#             print(f"  Could not find actor entity: {actor}")

#     return {
#         "scene": scene,
#         "result": result,
#         "violations": violations,
#         "is_safe": len(violations) == 0 and result.is_safe,
#         "pipeline_info": pipeline_info,
#     }


"""
ProcTHOR-10k Dataset Adapter
==============================
Bridges ProcTHOR-10k ai2thor scenes to the Neuro-Symbolic Safety Guardrail pipeline.

Complete Pipeline Integration:
    ProcTHOR Scene → Scene Graph → Logical Predicates → Differentiable Logic Reasoning
    → Safety Rule Activation → Action Violation Checking

Usage:
    import prior
    dataset = prior.load_dataset("procthor-10k")

    adapter = ProcTHORAdapter(dataset["train"])
    scene = adapter.load_scene(scene_id="train_0")

    # Run full neuro-symbolic pipeline
    pipeline_info, safety_result = run_neuro_symbolic_pipeline(scene)

    # Check actions
    violations = safety_result.check_action("move_towards", "obj_0", "obj_1")
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from utils.scene_graph_builder import (
    SceneGraphBuilder,
    SpatialRelationConfig,
    ParsedScene,
    SceneObject,
    ExplicitRelation,
)
from utils.differentiable_safety_reasoner import DifferentiableSafetyReasoner


# ProcTHOR object category to safety semantic types mapping
# Includes both canonical names and ProcTHOR-10k specific type names
# ProcTHOR uses CamelCase (e.g., "StoveBurner", "CoffeeMachine")
PROCTHOR_SEMANTIC_MAPPING = {
    # Fire hazards - lowercase variants
    "stove": ["fire_source", "hot", "electrical"],
    "stoveburner": ["fire_source", "hot", "electrical"],
    "oven": ["fire_source", "hot", "electrical"],
    "toaster": ["fire_source", "hot", "electrical"],
    "toasteroven": ["fire_source", "hot", "electrical"],
    "microwave": ["fire_source", "electrical"],
    "candle": ["fire_source", "flammable"],
    "fireplace": ["fire_source", "hot"],
    "lighter": ["fire_source", "flammable"],
    "match": ["fire_source", "flammable"],
    "gas_can": ["flammable", "liquid"],
    "gasoline_can": ["flammable", "liquid"],
    "oil": ["flammable", "liquid"],

    # Fire hazards - CamelCase variants (ProcTHOR format)
    "Stove": ["fire_source", "hot", "electrical"],
    "StoveBurner": ["fire_source", "hot", "electrical"],
    "Oven": ["fire_source", "hot", "electrical"],
    "Toaster": ["fire_source", "hot", "electrical"],
    "ToasterOven": ["fire_source", "hot", "electrical"],
    "Microwave": ["fire_source", "electrical"],
    "Candle": ["fire_source", "flammable"],
    "Fireplace": ["fire_source", "hot"],
    "Lighter": ["fire_source", "flammable"],
    "Match": ["fire_source", "flammable"],
    "GasCan": ["flammable", "liquid"],
    "GasolineCan": ["flammable", "liquid"],
    "Oil": ["flammable", "liquid"],

    # ProcTHOR-specific fire hazards
    "pot": ["hot", "container", "fire_source"],
    "pan": ["hot", "container", "fire_source"],
    "kettle": ["hot", "electrical"],
    "coffee_machine": ["hot", "electrical"],
    "heater": ["hot", "fire_source", "electrical"],

    # Electrical hazards - lowercase
    "television": ["electrical", "is_electrical", "fragile"],
    "tv": ["electrical", "is_electrical", "fragile"],
    "televisionstand": ["electrical", "is_electrical", "fragile"],
    "desk_lamp": ["electrical", "is_electrical"],
    "floor_lamp": ["electrical", "is_electrical"],
    "table_lamp": ["electrical", "is_electrical"],
    "lamp": ["electrical", "is_electrical"],
    "computer": ["electrical", "is_electrical", "fragile"],
    "monitor": ["electrical", "is_electrical", "fragile"],
    "laptop": ["electrical", "is_electrical", "fragile"],
    "phone": ["electrical", "is_electrical"],
    "refrigerator": ["electrical", "is_electrical", "heavy"],
    "fridge": ["electrical", "is_electrical", "heavy"],
    "dishwasher": ["electrical", "is_electrical"],
    "washing_machine": ["electrical", "is_electrical", "is_wet"],
    "washingmachine": ["electrical", "is_electrical", "is_wet"],
    "dryer": ["electrical", "is_electrical", "hot"],
    "power_strip": ["electrical", "is_electrical"],
    "powerstrip": ["electrical", "is_electrical"],
    "outlet": ["electrical", "is_electrical"],
    "extension_cord": ["electrical", "is_electrical"],
    "extensioncord": ["electrical", "is_electrical"],
    "fan": ["electrical", "is_electrical"],
    "printer": ["electrical", "is_electrical"],

    # Electrical hazards - CamelCase
    "Television": ["electrical", "is_electrical", "fragile"],
    "TV": ["electrical", "is_electrical", "fragile"],
    "TelevisionStand": ["electrical", "is_electrical", "fragile"],
    "DeskLamp": ["electrical", "is_electrical"],
    "FloorLamp": ["electrical", "is_electrical"],
    "TableLamp": ["electrical", "is_electrical"],
    "Lamp": ["electrical", "is_electrical"],
    "Computer": ["electrical", "is_electrical", "fragile"],
    "Monitor": ["electrical", "is_electrical", "fragile"],
    "Laptop": ["electrical", "is_electrical", "fragile"],
    "Phone": ["electrical", "is_electrical"],
    "Refrigerator": ["electrical", "is_electrical", "heavy"],
    "Fridge": ["electrical", "is_electrical", "heavy"],
    "Dishwasher": ["electrical", "is_electrical"],
    "WashingMachine": ["electrical", "is_electrical", "is_wet"],
    "Dryer": ["electrical", "is_electrical", "hot"],
    "PowerStrip": ["electrical", "is_electrical"],
    "Outlet": ["electrical", "is_electrical"],
    "ExtensionCord": ["electrical", "is_electrical"],
    "Fan": ["electrical", "is_electrical"],
    "Printer": ["electrical", "is_electrical"],
    "computer": ["electrical", "is_electrical", "fragile"],
    "monitor": ["electrical", "is_electrical", "fragile"],
    "laptop": ["electrical", "is_electrical", "fragile"],
    "phone": ["electrical", "is_electrical"],
    "refrigerator": ["electrical", "is_electrical", "heavy"],
    "dishwasher": ["electrical", "is_electrical"],
    "washing_machine": ["electrical", "is_electrical", "is_wet"],
    "dryer": ["electrical", "is_electrical", "hot"],
    "power_strip": ["electrical", "is_electrical"],
    "outlet": ["electrical", "is_electrical"],
    "extension_cord": ["electrical", "is_electrical"],
    "fan": ["electrical", "is_electrical"],
    "printer": ["electrical", "is_electrical"],

    # Sharp objects - lowercase
    "knife": ["sharp", "weapon"],
    "butterknife": ["sharp", "weapon"],
    "steakknife": ["sharp", "weapon"],
    "scissors": ["sharp"],
    "razor": ["sharp"],
    "glass": ["sharp", "fragile", "glass"],
    "bottle": ["sharp", "fragile", "glass"],
    "wine_glass": ["fragile", "glass"],
    "wineglass": ["fragile", "glass"],
    "winebottle": ["fragile", "glass"],
    "mirror": ["sharp", "fragile", "glass"],
    "window": ["fragile", "glass"],
    "plate": ["fragile", "breakable"],
    "bowl": ["fragile", "breakable"],
    "cup": ["fragile", "breakable"],

    # Sharp objects - CamelCase
    "Knife": ["sharp", "weapon"],
    "ButterKnife": ["sharp", "weapon"],
    "SteakKnife": ["sharp", "weapon"],
    "Scissors": ["sharp"],
    "Razor": ["sharp"],
    "Glass": ["sharp", "fragile", "glass"],
    "Bottle": ["sharp", "fragile", "glass"],
    "WineGlass": ["fragile", "glass"],
    "WineBottle": ["fragile", "glass"],
    "Mirror": ["sharp", "fragile", "glass"],
    "Window": ["fragile", "glass"],
    "Plate": ["fragile", "breakable"],
    "Bowl": ["fragile", "breakable"],
    "Cup": ["fragile", "breakable"],

    # Chemical/toxic hazards - lowercase
    "bleach": ["chemical", "toxic", "liquid"],
    "cleaning_solution": ["chemical", "toxic", "liquid"],
    "detergent": ["chemical", "toxic", "liquid"],
    "medicine": ["chemical", "toxic"],
    "pills": ["chemical", "toxic"],
    "pesticide": ["chemical", "toxic"],
    "paint": ["chemical", "toxic", "liquid"],
    "solvent": ["chemical", "toxic", "liquid", "flammable"],
    "cleaningspray": ["chemical", "toxic", "liquid"],
    "spraybottle": ["chemical", "toxic", "liquid"],

    # Chemical/toxic hazards - CamelCase
    "Bleach": ["chemical", "toxic", "liquid"],
    "CleaningSolution": ["chemical", "toxic", "liquid"],
    "Detergent": ["chemical", "toxic", "liquid"],
    "Medicine": ["chemical", "toxic"],
    "Pills": ["chemical", "toxic"],
    "Pesticide": ["chemical", "toxic"],
    "Paint": ["chemical", "toxic", "liquid"],
    "Solvent": ["chemical", "toxic", "liquid", "flammable"],
    "CleaningSpray": ["chemical", "toxic", "liquid"],
    "SprayBottle": ["chemical", "toxic", "liquid"],

    # Heavy objects (crush hazard) - lowercase
    "book": ["heavy"],
    "books": ["heavy"],
    "bookshelf": ["heavy", "furniture"],
    "bookcase": ["heavy", "furniture"],
    "cabinet": ["heavy", "furniture"],
    "dresser": ["heavy", "furniture"],
    "wardrobe": ["heavy", "furniture"],
    "closet": ["heavy", "furniture"],
    "tv_stand": ["heavy", "furniture"],
    "tvstand": ["heavy", "furniture"],
    "desk": ["heavy", "furniture"],
    "table": ["heavy", "furniture"],
    "diningtable": ["heavy", "furniture"],
    "coffeetable": ["heavy", "furniture"],
    "sidetable": ["heavy", "furniture"],
    "endtable": ["heavy", "furniture"],
    "nightstand": ["heavy", "furniture"],
    "bathtub": ["heavy"],
    "sink": ["heavy", "is_wet"],
    "kitchenisland": ["heavy", "furniture"],
    "countertop": ["heavy", "furniture"],
    "counter": ["heavy", "furniture"],

    # Heavy objects - CamelCase
    "Book": ["heavy"],
    "Books": ["heavy"],
    "Bookshelf": ["heavy", "furniture"],
    "Bookcase": ["heavy", "furniture"],
    "Cabinet": ["heavy", "furniture"],
    "Dresser": ["heavy", "furniture"],
    "Wardrobe": ["heavy", "furniture"],
    "Closet": ["heavy", "furniture"],
    "TVStand": ["heavy", "furniture"],
    "Desk": ["heavy", "furniture"],
    "Table": ["heavy", "furniture"],
    "DiningTable": ["heavy", "furniture"],
    "CoffeeTable": ["heavy", "furniture"],
    "SideTable": ["heavy", "furniture"],
    "EndTable": ["heavy", "furniture"],
    "NightStand": ["heavy", "furniture"],
    "Bathtub": ["heavy"],
    "Sink": ["heavy", "is_wet"],
    "KitchenIsland": ["heavy", "furniture"],
    "CounterTop": ["heavy", "furniture"],
    "Counter": ["heavy", "furniture"],

    # Slip/fall hazards - lowercase
    "rug": ["slip_hazard"],
    "mat": ["slip_hazard", "on_floor"],
    "stairs": ["fall_hazard"],
    "staircase": ["fall_hazard"],
    "shower": ["slip_hazard", "is_wet"],
    "faucet": ["is_wet"],
    "water": ["liquid", "is_wet", "slip_hazard"],
    "puddle": ["liquid", "is_wet", "slip_hazard", "on_floor"],
    "showerhead": ["is_wet"],
    "soap": ["slip_hazard"],

    # Slip/fall hazards - CamelCase
    "Rug": ["slip_hazard"],
    "Mat": ["slip_hazard", "on_floor"],
    "Stairs": ["fall_hazard"],
    "Staircase": ["fall_hazard"],
    "Shower": ["slip_hazard", "is_wet"],
    "Faucet": ["is_wet"],
    "Water": ["liquid", "is_wet", "slip_hazard"],
    "Puddle": ["liquid", "is_wet", "slip_hazard", "on_floor"],
    "ShowerHead": ["is_wet"],
    "Soap": ["slip_hazard"],

    # People (child safety) - lowercase
    "person": ["is_person"],
    "child": ["is_person", "is_child"],
    "baby": ["is_person", "is_child"],
    "infant": ["is_person", "is_child"],
    "elderly": ["is_person", "is_elderly"],
    "man": ["is_person"],
    "woman": ["is_person"],
    "boy": ["is_person", "is_child"],
    "girl": ["is_person", "is_child"],

    # People - CamelCase
    "Person": ["is_person"],
    "Child": ["is_person", "is_child"],
    "Baby": ["is_person", "is_child"],
    "Infant": ["is_person", "is_child"],
    "Elderly": ["is_person", "is_elderly"],
    "Man": ["is_person"],
    "Woman": ["is_person"],
    "Boy": ["is_person", "is_child"],
    "Girl": ["is_person", "is_child"],

    # Containers - lowercase
    "cup": ["is_container"],
    "mug": ["is_container"],
    "bowl": ["is_container"],
    "pot": ["is_container", "hot"],
    "pan": ["is_container", "hot"],
    "box": ["is_container"],
    "drawer": ["is_container", "locked"],
    "safe": ["is_container", "locked"],
    "jar": ["is_container"],
    "can": ["is_container"],
    "bucket": ["is_container"],
    "trashcan": ["is_container"],
    "plate": ["is_container"],
    "coffee_maker": ["is_container", "hot", "electrical"],
    "coffeemaker": ["is_container", "hot", "electrical"],
    "kettle": ["is_container", "hot", "electrical"],

    # Containers - CamelCase
    "Cup": ["is_container"],
    "Mug": ["is_container"],
    "Bowl": ["is_container"],
    "Pot": ["is_container", "hot"],
    "Pan": ["is_container", "hot"],
    "Box": ["is_container"],
    "Drawer": ["is_container", "locked"],
    "Safe": ["is_container", "locked"],
    "Jar": ["is_container"],
    "Can": ["is_container"],
    "Bucket": ["is_container"],
    "TrashCan": ["is_container"],
    "Plate": ["is_container"],
    "CoffeeMaker": ["is_container", "hot", "electrical"],
    "Kettle": ["is_container", "hot", "electrical"],

    # Choking hazards (small objects)
    "pen": ["small"],
    "pencil": ["small"],
    "coin": ["small", "choking"],
    "button": ["small", "choking"],
    "battery": ["small", "chemical", "toxic", "choking"],
    "magnet": ["small", "choking"],
    "key": ["small"],
    "keys": ["small"],
    "remote": ["small"],
    "cellphone": ["small", "electrical", "is_electrical"],
    "creditcard": ["small"],
    "watch": ["small"],
    "spoon": ["small"],
    "fork": ["small", "sharp"],
    "spatula": ["small"],

    # General furniture (neutral)
    "chair": ["furniture"],
    "armchair": ["furniture"],
    "bed": ["furniture"],
    "couch": ["furniture"],
    "sofa": ["furniture"],
    "loveseat": ["furniture"],
    "shelf": ["furniture"],
    "coffeetable": ["furniture"],
    "ottoman": ["furniture"],
    "stool": ["furniture"],
    "bench": ["furniture"],
    "fridge": ["heavy", "electrical", "is_electrical"],
    "kitchencart": ["furniture"],
    "diningtable": ["furniture", "heavy"],
    "endtable": ["furniture"],
    "nightstand": ["furniture"],
    "sidetable": ["furniture"],
    "counter": ["furniture"],
    "countertop": ["furniture"],

    # Kitchen items
    "spatula": ["small"],
    "butterknife": ["small", "sharp"],
    "bread": ["small"],
    "apple": ["small"],
    "banana": ["small"],
    "lettuce": ["small"],
    "tomato": ["small"],
    "potato": ["small"],
    "egg": ["small", "fragile"],
    "orange": ["small"],
    "grape": ["small"],
    "sandwich": ["small"],
    "winebottle": ["sharp", "fragile", "glass"],
    "saltshaker": ["small"],
    "peppershaker": ["small"],
    "kettle": ["hot", "electrical"],
    "mug": ["is_container"],
    "cup": ["is_container"],
    "glass": ["fragile", "glass"],
    "plate": ["fragile", "breakable"],

    # Decorations
    "vase": ["fragile"],
    "statue": ["heavy"],
    "pictureframe": ["fragile"],
    "painting": ["fragile"],
    "plant": [],
    "houseplant": [],

    # Misc
    "alarmclock": ["small", "electrical", "is_electrical"],
    "tissuebox": ["small"],
    "newspaper": ["small"],
    "cd": ["small"],
    "pillow": ["small"],
    "towel": ["small"],
    "cloth": ["small", "flammable", "is_cloth"],
    "keychain": ["small"],

    # Furniture - CamelCase
    "Chair": ["furniture"],
    "ArmChair": ["furniture"],
    "Bed": ["furniture"],
    "Couch": ["furniture"],
    "Sofa": ["furniture"],
    "LoveSeat": ["furniture"],
    "Shelf": ["furniture"],
    "CoffeeTable": ["furniture"],
    "Ottoman": ["furniture"],
    "Stool": ["furniture"],
    "Bench": ["furniture"],
    "Fridge": ["heavy", "electrical", "is_electrical"],
    "KitchenCart": ["furniture"],
    "DiningTable": ["furniture", "heavy"],
    "EndTable": ["furniture"],
    "NightStand": ["furniture"],
    "SideTable": ["furniture"],
    "Counter": ["furniture"],
    "CounterTop": ["furniture"],

    # Small items - CamelCase
    "Pen": ["small"],
    "Pencil": ["small"],
    "Coin": ["small", "choking"],
    "Button": ["small", "choking"],
    "Battery": ["small", "chemical", "toxic", "choking"],
    "Magnet": ["small", "choking"],
    "Key": ["small"],
    "Keys": ["small"],
    "Remote": ["small"],
    "CellPhone": ["small", "electrical", "is_electrical"],
    "CreditCard": ["small"],
    "Watch": ["small"],
    "Spoon": ["small"],
    "Fork": ["small", "sharp"],
    "Spatula": ["small"],

    # Kitchen - CamelCase
    "ButterKnife": ["small", "sharp"],
    "Bread": ["small"],
    "Apple": ["small"],
    "Banana": ["small"],
    "Lettuce": ["small"],
    "Tomato": ["small"],
    "Potato": ["small"],
    "Egg": ["small", "fragile"],
    "Orange": ["small"],
    "Grape": ["small"],
    "Sandwich": ["small"],
    "WineBottle": ["sharp", "fragile", "glass"],
    "SaltShaker": ["small"],
    "PepperShaker": ["small"],
    "Kettle": ["hot", "electrical"],
    "Mug": ["is_container"],
    "Cup": ["is_container"],
    "Glass": ["fragile", "glass"],
    "Plate": ["fragile", "breakable"],

    # Decor - CamelCase
    "Vase": ["fragile"],
    "Statue": ["heavy"],
    "PictureFrame": ["fragile"],
    "Painting": ["fragile"],
    "Plant": [],
    "HousePlant": [],

    # Misc - CamelCase
    "AlarmClock": ["small", "electrical", "is_electrical"],
    "TissueBox": ["small"],
    "Newspaper": ["small"],
    "CD": ["small"],
    "Pillow": ["small"],
    "Towel": ["small"],
    "Cloth": ["small", "flammable", "is_cloth"],
    "KeyChain": ["small"],
}


@dataclass
class ProcTHORObject:
    """Object from ProcTHOR scene."""
    obj_id: str
    category: str
    position_3d: Tuple[float, float, float]  # (x, y, z)
    rotation: Tuple[float, float, float]  # (pitch, yaw, roll)
    bbox_3d: Optional[List[float]] = None  # [min_x, min_y, min_z, max_x, max_y, max_z]
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_types(self) -> List[str]:
        """Get safety-relevant semantic types based on category."""
        # Try exact match first (for CamelCase)
        types = PROCTHOR_SEMANTIC_MAPPING.get(self.category, [])
        if types:
            return types

        # Try lowercase variant
        category_lower = self.category.lower().replace(" ", "_")
        types = PROCTHOR_SEMANTIC_MAPPING.get(category_lower, [])
        if types:
            return types

        # Try snake_case variant
        category_snake = self._camel_to_snake(self.category)
        types = PROCTHOR_SEMANTIC_MAPPING.get(category_snake, [])

        return types

    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        return pattern.sub('_', name).lower()


@dataclass
class ProcTHORScene:
    """Complete loaded scene from ProcTHOR dataset."""
    scene_id: str
    room_type: str  # kitchen, bedroom, living room, bathroom, etc.
    description: str

    # Room structure
    rooms: List[Dict[str, Any]] = field(default_factory=list)
    walls: List[Dict[str, Any]] = field(default_factory=list)
    floors: List[Dict[str, Any]] = field(default_factory=list)

    # Objects
    objects: List[ProcTHORObject] = field(default_factory=list)

    # Agent information (if available)
    agent_position: Optional[Tuple[float, float, float]] = None
    agent_rotation: Optional[float] = None

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def object_categories(self) -> Set[str]:
        return set(obj.category for obj in self.objects)

    @property
    def hazard_objects(self) -> List[ProcTHORObject]:
        """Return objects that are potential safety hazards."""
        hazards = []
        for obj in self.objects:
            types = obj.semantic_types
            if any(t in ["sharp", "hot", "fire_source", "chemical", "toxic", "electrical", "flammable"]
                   for t in types):
                hazards.append(obj)
        return hazards

    def get_objects_by_type(self, semantic_type: str) -> List[ProcTHORObject]:
        """Find all objects with a specific semantic type."""
        return [obj for obj in self.objects if semantic_type in obj.semantic_types]

    def summary(self) -> str:
        lines = [
            f"ProcTHORScene: {self.scene_id}",
            f"  Room type: {self.room_type}",
            f"  Description: {self.description}",
            f"  Objects: {self.num_objects}",
            f"  Categories: {', '.join(sorted(self.object_categories))}",
            f"  Potential hazards: {len(self.hazard_objects)}",
        ]
        for obj in self.objects[:10]:  # Show first 10
            types_str = ", ".join(obj.semantic_types[:3]) if obj.semantic_types else "none"
            pos = obj.position_3d
            lines.append(
                f"    [{obj.obj_id}] {obj.category} "
                f"types={{{types_str}}} "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )
        if self.num_objects > 10:
            lines.append(f"    ... and {self.num_objects - 10} more objects")
        return "\n".join(lines)


class ProcTHORAdapter:
    """
    Loads ProcTHOR-10k scenes and converts them to the safety pipeline format.

    Usage:
        import prior
        dataset = prior.load_dataset("procthor-10k")

        adapter = ProcTHORAdapter(dataset["train"])
        scene = adapter.load_scene(scene_id="train_0")

        # Convert to predicates
        predicates = adapter.to_predicates(scene)

        # Or get grounded facts directly
        grounded_facts, entity_ids = adapter.to_grounded_facts(scene)
    """

    def __init__(self, dataset_split):
        """
        Args:
            dataset_split: A dataset split from prior.load_dataset() or a list of scenes
                          (e.g., dataset["train"], dataset["val"])
        """
        self.dataset = dataset_split
        # Handle various ID field names used by different ProcTHOR dataset versions
        self._scene_map = {}
        for i, scene in enumerate(dataset_split):
            scene_id = None
            if isinstance(scene, dict):
                for key in ["id", "scene_id", "name", "unique_id", "scene_name", "sceneId"]:
                    if key in scene:
                        scene_id = scene[key]
                        break
            # Fallback: use index if no ID found
            if scene_id is None:
                scene_id = f"scene_{i}"
            self._scene_map[str(scene_id)] = scene
        print(f"ProcTHORAdapter: loaded {len(self.dataset)} scenes")

    @classmethod
    def from_local_path(cls, data_path: str, split: str = "train"):
        """
        Load ProcTHOR dataset from local directory.

        Args:
            data_path: Path to the dataset directory (e.g., "/data/procthor-10k")
            split: Which split to load ("train", "val", "test")

        Returns:
            ProcTHORAdapter instance
        """
        import json
        from pathlib import Path

        data_path = Path(data_path)

        # Try different file naming conventions
        possible_files = [
            data_path / f"{split}.json",
            data_path / f"{split}.jsonl",
            data_path / split / "scenes.json",
            data_path / split / "data.json",
            data_path / "scenes" / f"{split}.json",
        ]

        scenes = []
        loaded_file = None

        for file_path in possible_files:
            if file_path.exists():
                print(f"Loading {split} split from: {file_path}")
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.jsonl':
                        # JSONL format: one JSON object per line
                        scenes = [json.loads(line) for line in f if line.strip()]
                    else:
                        # JSON format
                        data = json.load(f)
                        # Handle both list format and dict with "scenes" key
                        if isinstance(data, list):
                            scenes = data
                        elif isinstance(data, dict) and "scenes" in data:
                            scenes = data["scenes"]
                        elif isinstance(data, dict) and split in data:
                            scenes = data[split]
                        else:
                            scenes = [data]  # Single scene
                loaded_file = file_path
                break

        if not scenes:
            # Try loading from directory of individual scene files
            scene_dir = data_path / split
            if scene_dir.exists() and scene_dir.is_dir():
                print(f"Loading scenes from directory: {scene_dir}")
                scene_files = sorted(scene_dir.glob("*.json"))
                for scene_file in scene_files:
                    with open(scene_file, 'r') as f:
                        scenes.append(json.load(f))

        if not scenes:
            raise FileNotFoundError(
                f"Could not find ProcTHOR data at {data_path} for split '{split}'. "
                f"Tried: {[str(p) for p in possible_files]}"
            )

        print(f"Loaded {len(scenes)} scenes from {loaded_file or scene_dir}")
        return cls(scenes)

    def list_scenes(self) -> List[str]:
        """List all available scene IDs."""
        return list(self._scene_map.keys())

    def load_scene(self, scene_id: str) -> ProcTHORScene:
        """
        Load a ProcTHOR scene by ID.

        Args:
            scene_id: Scene identifier (e.g., "train_0", "val_42")

        Returns:
            ProcTHORScene with all objects and metadata
        """
        if scene_id not in self._scene_map:
            raise KeyError(f"Scene {scene_id} not found in dataset")

        raw_scene = self._scene_map[scene_id]

        # Determine room type from objects or rooms
        room_type = self._infer_room_type(raw_scene)

        scene = ProcTHORScene(
            scene_id=scene_id,
            room_type=room_type,
            description=raw_scene.get("description", f"ProcTHOR {room_type}"),
        )

        # Load rooms structure if available
        if "rooms" in raw_scene:
            scene.rooms = raw_scene["rooms"]
        if "walls" in raw_scene:
            scene.walls = raw_scene["walls"]
        if "floors" in raw_scene:
            scene.floors = raw_scene["floors"]

        # Load objects
        objects_data = raw_scene.get("objects", [])
        for i, obj_data in enumerate(objects_data):
            obj = self._parse_object(obj_data, obj_id=f"obj_{i}")
            scene.objects.append(obj)

        # Agent position if available
        if "agent" in raw_scene:
            agent_data = raw_scene["agent"]
            if "position" in agent_data:
                scene.agent_position = tuple(agent_data["position"])
            if "rotation" in agent_data:
                scene.agent_rotation = agent_data["rotation"]

        return scene

    def to_predicates(self, scene: ProcTHORScene) -> str:
        """
        Convert scene to Prolog-style predicates.

        Output format:
            % ProcTHOR scene: train_0
            object(obj_0, stove, [fire_source, hot, electrical]).
            position(obj_0, 2.5, 0.8, 1.2).
            ...
        """
        lines = [
            f"% ProcTHOR scene: {scene.scene_id}",
            f"% Room type: {scene.room_type}",
            f"% Objects: {scene.num_objects}",
            "",
            "% ── Objects ──",
        ]

        for obj in scene.objects:
            types = obj.semantic_types
            if len(types) == 1:
                types_str = types[0]
            elif len(types) > 1:
                types_str = "[" + ", ".join(types) + "]"
            else:
                types_str = "unknown"

            lines.append(f"object({obj.obj_id}, {obj.category}, {types_str}).")

        # Positions
        lines.append("")
        lines.append("% ── 3D Positions (meters) ──")
        for obj in scene.objects:
            x, y, z = obj.position_3d
            lines.append(f"position({obj.obj_id}, {x:.3f}, {y:.3f}, {z:.3f}).")

        # Attributes (temperature, etc.)
        lines.append("")
        lines.append("% ── Attributes ──")
        for obj in scene.objects:
            types = obj.semantic_types
            if "hot" in types or "fire_source" in types:
                lines.append(f"attribute({obj.obj_id}, temperature, 100).")
            if "is_wet" in types:
                lines.append(f"attribute({obj.obj_id}, wet, 1).")
            if "electrical" in types:
                lines.append(f"attribute({obj.obj_id}, powered, 1).")

        return "\n".join(lines)

    def to_grounded_facts(
        self,
        scene: ProcTHORScene,
        spatial_config: Optional[SpatialRelationConfig] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Full conversion: scene -> grounded facts ready for the reasoner.
        """
        builder = SceneGraphBuilder(
            spatial_config=spatial_config,
            require_grad_positions=False,  # ProcTHOR positions are fixed
        )
        parsed = self._to_parsed_scene(scene)
        return builder.build_from_parsed(parsed, device)

    def _to_parsed_scene(self, scene: ProcTHORScene) -> ParsedScene:
        """Convert to ParsedScene for SceneGraphBuilder."""
        parsed = ParsedScene()

        for obj in scene.objects:
            parsed.objects[obj.obj_id] = SceneObject(
                obj_id=obj.obj_id,
                category=obj.category,
                semantic_types=set(obj.semantic_types),
                position_3d=obj.position_3d,
                attributes=obj.attributes.copy(),
                detection_confidence=1.0,  # Ground truth from simulation
            )

        return parsed

    def _parse_object(self, obj_data: Dict, obj_id: str) -> ProcTHORObject:
        """Parse a single object from ProcTHOR format."""
        # Get position
        position = obj_data.get("position", [0.0, 0.0, 0.0])
        if isinstance(position, dict):
            position = (position.get("x", 0), position.get("y", 0), position.get("z", 0))
        else:
            position = tuple(position[:3])

        # Get rotation
        rotation = obj_data.get("rotation", [0.0, 0.0, 0.0])
        if isinstance(rotation, dict):
            rotation = (rotation.get("x", 0), rotation.get("y", 0), rotation.get("z", 0))
        else:
            rotation = tuple(rotation[:3])

        # Get bounding box if available
        bbox = obj_data.get("bbox", None)
        if bbox and isinstance(bbox, dict):
            bbox = [
                bbox.get("min", {}).get("x", 0), bbox.get("min", {}).get("y", 0), bbox.get("min", {}).get("z", 0),
                bbox.get("max", {}).get("x", 0), bbox.get("max", {}).get("y", 0), bbox.get("max", {}).get("z", 0),
            ]

        obj = ProcTHORObject(
            obj_id=obj_id,
            category=obj_data.get("objectType", obj_data.get("type", "unknown")),
            position_3d=position,
            rotation=rotation,
            bbox_3d=bbox,
            attributes=obj_data.get("attributes", {}),
        )

        return obj

    def _infer_room_type(self, raw_scene: Dict) -> str:
        """Infer the primary room type from scene metadata."""
        # Direct room type
        if "roomType" in raw_scene:
            return raw_scene["roomType"]

        # Infer from rooms list
        rooms = raw_scene.get("rooms", [])
        if rooms:
            room_types = [r.get("roomType", "").lower() for r in rooms]
            # Count occurrences and pick most common
            type_counts = {}
            for rt in room_types:
                type_counts[rt] = type_counts.get(rt, 0) + 1
            if type_counts:
                return max(type_counts, key=type_counts.get)

        # Infer from objects
        objects = raw_scene.get("objects", [])
        obj_types = set(obj.get("objectType", "").lower() for obj in objects)

        # Kitchen indicators
        kitchen_objs = {"stove", "oven", "refrigerator", "sink", "countertop", "dishwasher"}
        if kitchen_objs & obj_types:
            return "kitchen"

        # Bathroom indicators
        bathroom_objs = {"toilet", "sink", "bathtub", "shower", "mirror"}
        if bathroom_objs & obj_types:
            return "bathroom"

        # Bedroom indicators
        bedroom_objs = {"bed", "nightstand", "dresser", "wardrobe"}
        if bedroom_objs & obj_types:
            return "bedroom"

        # Living room indicators
        living_objs = {"sofa", "couch", "tv", "television", "coffee_table"}
        if living_objs & obj_types:
            return "living_room"

        return "unknown"

    def sample_scenes_by_type(self, room_type: str, n: int = 5) -> List[str]:
        """
        Sample scene IDs of a specific room type.

        Args:
            room_type: e.g., "kitchen", "bedroom", "bathroom", "living_room"
            n: Number of scenes to sample

        Returns:
            List of scene IDs
        """
        matching = []
        for scene_id, raw_scene in self._scene_map.items():
            if self._infer_room_type(raw_scene) == room_type.lower():
                matching.append(scene_id)

        import random
        random.seed(42)
        return random.sample(matching, min(n, len(matching)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_scenes": len(self.dataset),
            "room_types": defaultdict(int),
            "total_objects": 0,
            "object_categories": set(),
            "hazard_counts": defaultdict(int),
        }

        for raw_scene in self.dataset:
            room_type = self._infer_room_type(raw_scene)
            stats["room_types"][room_type] += 1

            objects = raw_scene.get("objects", [])
            stats["total_objects"] += len(objects)

            for obj in objects:
                category = obj.get("objectType", obj.get("type", "unknown"))
                stats["object_categories"].add(category)

                # Count potential hazards
                category_lower = category.lower().replace(" ", "_")
                semantic_types = PROCTHOR_SEMANTIC_MAPPING.get(category_lower, [])
                for hazard in ["sharp", "hot", "fire_source", "chemical", "toxic", "electrical", "flammable"]:
                    if hazard in semantic_types:
                        stats["hazard_counts"][hazard] += 1

        stats["object_categories"] = sorted(stats["object_categories"])
        return stats


# ─── Neuro-Symbolic Pipeline Integration ───

def run_neuro_symbolic_pipeline(
    scene: ProcTHORScene,
    activation_threshold: float = 0.3,
    verbose: bool = False,
) -> Tuple[Dict, Any]:
    """
    Run the complete neuro-symbolic safety pipeline on a ProcTHOR scene.

    Pipeline stages:
    1. Scene Graph Construction - Parse ProcTHOR objects into scene graph
    2. Logical Predicate Generation - Convert to Prolog-style predicates
    3. Differentiable Logic Reasoning - Forward chaining with neural predicates
    4. Safety Rule Extraction - Get activated safety rules with probabilities

    Args:
        scene: Loaded ProcTHOR scene
        activation_threshold: Minimum probability to activate a rule
        verbose: Print detailed output

    Returns:
        Tuple of (pipeline_info dict, SafetyReasoningResult)
    """
    adapter = ProcTHORAdapter([])  # Dummy adapter for helper methods

    if verbose:
        print(f"\n  [Stage 1/4] Scene Graph Construction")
        print(f"    - Objects: {scene.num_objects}")
        print(f"    - Room type: {scene.room_type}")

    # Stage 1: Convert to grounded facts through SceneGraphBuilder
    builder = SceneGraphBuilder(require_grad_positions=False)
    parsed = adapter._to_parsed_scene(scene)
    grounded_facts, entity_ids = builder.build_from_parsed(parsed)

    # Count spatial relations
    spatial_relations = [k for k in grounded_facts.keys() if any(
        rel in k for rel in ['near(', 'touching(', 'above(', 'below(']
    )]

    if verbose:
        print(f"    ✓ ParsedScene created with {len(parsed.objects)} objects")
        print(f"\n  [Stage 2/4] Logical Predicate Generation")
        print(f"    - Entity IDs: {entity_ids[:5]}..." if len(entity_ids) > 5 else f"    - Entity IDs: {entity_ids}")
        print(f"    - Grounded facts: {len(grounded_facts)} predicates")
        print(f"    - Spatial relations: {len(spatial_relations)} (near/touching/above/below)")

    # Stage 3: Differentiable Safety Reasoning
    if verbose:
        print(f"\n  [Stage 3/4] Differentiable Logic Reasoning")
        print(f"    - Loading safety knowledge base...")

    reasoner = DifferentiableSafetyReasoner(activation_threshold=activation_threshold)

    if verbose:
        print(f"    - Running forward chaining inference with semiring semantics...")
        print(f"      AND: product, OR: noisy-OR")

    result = reasoner.reason(grounded_facts, entity_ids)

    if verbose:
        print(f"    ✓ Inference complete")
        print(f"\n  [Stage 4/4] Safety Rule Extraction")
        print(f"    - Total activated rules: {result.total_activated}")
        print(f"    - Critical: {result.critical_count} | High: {result.high_count} | "
              f"Medium: {result.medium_count} | Low: {result.low_count}")

        if result.activated_rules:
            print(f"\n    Top activated rules:")
            for rule in result.activated_rules[:5]:
                desc = rule.natural_language[:50] + "..." if len(rule.natural_language) > 50 else rule.natural_language
                print(f"      [{rule.severity.name}] P={rule.probability:.3f} | {rule.rule_name}")
                print(f"        {desc}")

    pipeline_info = {
        "grounded_facts_count": len(grounded_facts),
        "spatial_relations_count": len(spatial_relations),
        "entity_ids": entity_ids,
        "scene": scene,
    }

    return pipeline_info, result


def run_safety_pipeline_procthor(
    dataset,
    scene_id: str,
    activation_threshold: float = 0.3,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    target: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run safety assessment on a ProcTHOR scene with full neuro-symbolic pipeline.

    Args:
        dataset: ProcTHOR dataset split from prior.load_dataset()
        scene_id: Scene identifier
        activation_threshold: Minimum probability to activate a rule
        action: Action to check (optional)
        actor: Actor entity ID or category (optional)
        target: Target entity ID or category (optional)
        verbose: Print detailed output

    Returns:
        dict with safety results including:
        - scene: ProcTHORScene
        - result: SafetyReasoningResult
        - violations: List of violated rules
        - is_safe: Boolean
        - pipeline_info: Dict with grounded facts, spatial relations, etc.
    """
    # Step 1: Load scene
    if verbose:
        print("=" * 70)
        print("  ProcTHOR-10k Neuro-Symbolic Safety Pipeline")
        print("=" * 70)
        print(f"\n[1/5] Loading ProcTHOR scene: {scene_id}")

    adapter = ProcTHORAdapter(dataset)
    scene = adapter.load_scene(scene_id)

    if verbose:
        print(f"  Room type: {scene.room_type}")
        print(f"  Objects: {scene.num_objects}")
        print(f"  Potential hazards: {len(scene.hazard_objects)}")

    # Steps 2-4: Run neuro-symbolic pipeline
    pipeline_info, result = run_neuro_symbolic_pipeline(
        scene,
        activation_threshold=activation_threshold,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[5/5] Action violation checking...")

    # Step 5: Action violation checking
    violations = []
    if action:
        # Find matching entities
        actor_id = None
        target_id = None

        for obj in scene.objects:
            if actor and (actor.lower() in obj.category.lower() or obj.obj_id == actor):
                actor_id = obj.obj_id
            if target and (target.lower() in obj.category.lower() or obj.obj_id == target):
                target_id = obj.obj_id

        if actor_id:
            violations = result.check_action(action, actor_id, target_id)
            if verbose:
                if violations:
                    print(f"  BLOCKED: {len(violations)} violations found!")
                    for v in violations:
                        print(f"    - {v.rule_name} ({v.severity.name}): {v.natural_language}")
                else:
                    print(f"  ALLOWED: No safety violations found")
        elif verbose:
            print(f"  Could not find actor entity: {actor}")

    return {
        "scene": scene,
        "result": result,
        "violations": violations,
        "is_safe": len(violations) == 0 and result.is_safe,
        "pipeline_info": pipeline_info,
    }
