"""
EmbodiedScan Dataset Adapter
==============================
Bridges EmbodiedScan/OpenScan 3D scene data to the Neuro-Symbolic
Safety Guardrail pipeline.

Data flow:
    EmbodiedScan raw data (RGB, depth, poses, mesh)
    + Scene annotations (annotations.json per scene)
            |
            v
    EmbodiedScanAdapter.load_scene("office")
            |
            v
    EmbodiedScanScene (structured metadata + objects)
            |
            v
    adapter.to_predicates(scene) -> Prolog-style text
            |
            v
    SceneGraphBuilder.build(text) -> grounded_facts, entity_ids
            |
            v
    DifferentiableSafetyReasoner.reason(...)
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from scene_graph_builder import (
    SceneGraphBuilder,
    SpatialRelationConfig,
    NeuralPredicateParser,
    ParsedScene,
    SceneObject,
    ExplicitRelation,
)


# Data Structures

@dataclass
class CameraIntrinsic:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


@dataclass
class CameraPose:
    """A single camera pose (timestamp + 4x4 transform or position + quaternion)."""
    timestamp: float
    position: np.ndarray      # (3,) xyz
    quaternion: np.ndarray     # (4,) xyzw
    transform: Optional[np.ndarray] = None  # (4, 4) if available


@dataclass
class SceneAnnotation:
    """Object annotation for a scene."""
    obj_id: str
    category: str
    semantic_types: List[str]
    position_3d: Tuple[float, float, float]
    attributes: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class EmbodiedScanScene:
    """Complete loaded scene from EmbodiedScan dataset."""
    scene_id: str
    scene_type: str
    description: str
    scene_path: str

    # Sensor data metadata
    intrinsic: Optional[CameraIntrinsic] = None
    poses: List[CameraPose] = field(default_factory=list)
    axis_align_matrix: Optional[np.ndarray] = None
    num_rgb_frames: int = 0
    num_depth_frames: int = 0
    has_mesh: bool = False

    # Annotations
    objects: List[SceneAnnotation] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def object_ids(self) -> List[str]:
        return [o.obj_id for o in self.objects]

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    def summary(self) -> str:
        lines = [
            f"EmbodiedScanScene: {self.scene_id} ({self.scene_type})",
            f"  Description: {self.description}",
            f"  Path: {self.scene_path}",
            f"  Frames: {self.num_rgb_frames} RGB, {self.num_depth_frames} depth",
            f"  Poses: {len(self.poses)}",
            f"  Mesh: {'yes' if self.has_mesh else 'no'}",
            f"  Objects: {self.num_objects}",
        ]
        for obj in self.objects:
            types_str = ", ".join(obj.semantic_types)
            pos = obj.position_3d
            lines.append(
                f"    [{obj.obj_id}] {obj.category} "
                f"types={{{types_str}}} "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) "
                f"conf={obj.confidence:.2f}"
            )
        if self.relations:
            lines.append(f"  Relations: {len(self.relations)}")
            for rel in self.relations:
                lines.append(
                    f"    {rel['subject']} --{rel['relation']}--> {rel['object']}"
                )
        return "\n".join(lines)


# Adapter

class EmbodiedScanAdapter:
    """
    Loads EmbodiedScan/OpenScan data and converts it to the format
    expected by the safety reasoning pipeline.

    Usage:
        adapter = EmbodiedScanAdapter("data/openscan")
        scenes = adapter.list_scenes()
        scene = adapter.load_scene("office")
        prolog_text = adapter.to_predicates(scene)
        grounded_facts, entity_ids = adapter.to_grounded_facts(scene)
    """

    def __init__(self, data_root: str = "data/openscan"):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"EmbodiedScan data root not found: {self.data_root}"
            )

    def list_scenes(self) -> List[str]:
        """List all available scene directories."""
        scenes = []
        for entry in sorted(self.data_root.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                scenes.append(entry.name)
        return scenes

    def load_scene(self, scene_name: str) -> EmbodiedScanScene:
        """
        Load a complete scene: metadata + annotations.

        Args:
            scene_name: Directory name under data_root (e.g., "office")

        Returns:
            EmbodiedScanScene with all available data loaded
        """
        scene_path = self.data_root / scene_name
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene not found: {scene_path}")

        scene = EmbodiedScanScene(
            scene_id=scene_name,
            scene_type=scene_name,
            description="",
            scene_path=str(scene_path),
        )

        # Load annotations (required)
        self._load_annotations(scene, scene_path)

        # Load sensor metadata (optional, for reference)
        self._load_intrinsic(scene, scene_path)
        self._load_poses(scene, scene_path)
        self._load_axis_align(scene, scene_path)
        self._count_frames(scene, scene_path)
        scene.has_mesh = (scene_path / "mesh.ply").exists()

        return scene

    def to_predicates(self, scene: EmbodiedScanScene) -> str:
        """
        Convert scene to Prolog-style predicate text that
        SceneGraphBuilder.build() can parse directly.

        Output format:
            % Scene: office
            object(monitor, electrical_device, [electrical, is_electrical, fragile]).
            attribute(monitor, temperature, 35).
            position(monitor, 1.5, 0.8, 0.5).
            ...
        """
        lines = [
            f"% EmbodiedScan scene: {scene.scene_id}",
            f"% Type: {scene.scene_type}",
            f"% Objects: {scene.num_objects}",
            "",
            "% ── Objects ──",
        ]

        for obj in scene.objects:
            # Format semantic types
            if len(obj.semantic_types) == 1:
                types_str = obj.semantic_types[0]
            elif len(obj.semantic_types) > 1:
                types_str = "[" + ", ".join(obj.semantic_types) + "]"
            else:
                types_str = "unknown"

            # Confidence annotation
            conf_str = ""
            if obj.confidence < 1.0:
                conf_str = f" [{obj.confidence:.2f}]"

            lines.append(
                f"object({obj.obj_id}, {obj.category}, {types_str}){conf_str}."
            )

        # Attributes
        has_attrs = any(obj.attributes for obj in scene.objects)
        if has_attrs:
            lines.append("")
            lines.append("% ── Attributes ──")
            for obj in scene.objects:
                for attr_name, attr_val in obj.attributes.items():
                    lines.append(
                        f"attribute({obj.obj_id}, {attr_name}, {attr_val})."
                    )

        # Positions
        lines.append("")
        lines.append("% ── 3D Positions (meters) ──")
        for obj in scene.objects:
            x, y, z = obj.position_3d
            lines.append(f"position({obj.obj_id}, {x}, {y}, {z}).")

        # Explicit relations
        if scene.relations:
            lines.append("")
            lines.append("% ── Explicit Relations ──")
            for rel in scene.relations:
                conf_str = ""
                if rel.get("confidence", 1.0) < 1.0:
                    conf_str = f" [{rel['confidence']:.2f}]"
                lines.append(
                    f"relation({rel['subject']}, {rel['relation']}, "
                    f"{rel['object']}){conf_str}."
                )

        return "\n".join(lines)

    def to_parsed_scene(self, scene: EmbodiedScanScene) -> ParsedScene:
        """
        Convert to ParsedScene (intermediate representation),
        bypassing text serialization for efficiency.
        """
        parsed = ParsedScene()

        for obj in scene.objects:
            parsed.objects[obj.obj_id] = SceneObject(
                obj_id=obj.obj_id,
                category=obj.category,
                semantic_types=set(obj.semantic_types),
                position_3d=tuple(obj.position_3d),
                attributes=dict(obj.attributes),
                detection_confidence=obj.confidence,
            )

        for rel in scene.relations:
            parsed.explicit_relations.append(ExplicitRelation(
                subject_id=rel["subject"],
                relation=rel["relation"],
                object_id=rel["object"],
                confidence=rel.get("confidence", 1.0),
            ))

        return parsed

    def to_grounded_facts(
        self,
        scene: EmbodiedScanScene,
        spatial_config: Optional[SpatialRelationConfig] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Full conversion: scene -> grounded facts ready for the reasoner.

        This is the main entry point for pipeline integration.
        """
        builder = SceneGraphBuilder(
            spatial_config=spatial_config,
            require_grad_positions=True,
        )
        parsed = self.to_parsed_scene(scene)
        return builder.build_from_parsed(parsed, device)

    def get_frame_path(
        self,
        scene: EmbodiedScanScene,
        frame_idx: int,
        modality: str = "rgb",
    ) -> Optional[str]:
        """
        Get the file path for a specific frame.

        Args:
            scene: Loaded scene
            frame_idx: Frame index (0-based)
            modality: "rgb" or "depth"

        Returns:
            Full path to the frame image, or None
        """
        scene_path = Path(scene.scene_path)
        modality_dir = scene_path / modality
        if not modality_dir.exists():
            return None

        files = sorted(modality_dir.iterdir())
        if 0 <= frame_idx < len(files):
            return str(files[frame_idx])
        return None

    def get_mesh_path(self, scene: EmbodiedScanScene) -> Optional[str]:
        """Get path to scene mesh PLY file."""
        mesh_path = Path(scene.scene_path) / "mesh.ply"
        return str(mesh_path) if mesh_path.exists() else None

    # ─── Internal loaders ───

    def _load_annotations(self, scene: EmbodiedScanScene, scene_path: Path):
        """Load annotations.json for the scene."""
        ann_path = scene_path / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"No annotations.json found at {ann_path}. "
                f"Please create annotations for scene '{scene.scene_id}'."
            )

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scene.scene_id = data.get("scene_id", scene.scene_id)
        scene.scene_type = data.get("scene_type", scene.scene_type)
        scene.description = data.get("description", "")

        for obj_data in data.get("objects", []):
            scene.objects.append(SceneAnnotation(
                obj_id=obj_data["id"],
                category=obj_data.get("category", "unknown"),
                semantic_types=obj_data.get("semantic_types", []),
                position_3d=tuple(obj_data["position_3d"]),
                attributes=obj_data.get("attributes", {}),
                confidence=obj_data.get("confidence", 1.0),
            ))

        scene.relations = data.get("relations", [])

    def _load_intrinsic(self, scene: EmbodiedScanScene, scene_path: Path):
        """Load camera intrinsic parameters."""
        intrinsic_path = scene_path / "intrinsic.txt"
        if not intrinsic_path.exists():
            return

        mat = np.loadtxt(str(intrinsic_path))
        # Standard 4x4 intrinsic matrix format
        scene.intrinsic = CameraIntrinsic(
            fx=mat[0, 0], fy=mat[1, 1],
            cx=mat[0, 2], cy=mat[1, 2],
            width=1280, height=720,  # from calibration.yaml
        )

        # Try to get resolution from calibration.yaml
        cal_path = scene_path / "calibration.yaml"
        if cal_path.exists():
            with open(cal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("image_width:"):
                        scene.intrinsic.width = int(line.split(":")[1].strip())
                    elif line.startswith("image_height:"):
                        scene.intrinsic.height = int(line.split(":")[1].strip())

    def _load_poses(self, scene: EmbodiedScanScene, scene_path: Path):
        """Load camera poses (timestamp x y z qx qy qz qw)."""
        poses_path = scene_path / "poses.txt"
        if not poses_path.exists():
            return

        with open(poses_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                scene.poses.append(CameraPose(
                    timestamp=float(parts[0]),
                    position=np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                    quaternion=np.array([float(parts[4]), float(parts[5]),
                                        float(parts[6]), float(parts[7])]),
                ))

    def _load_axis_align(self, scene: EmbodiedScanScene, scene_path: Path):
        """Load axis alignment matrix."""
        align_path = scene_path / "axis_align_matrix.txt"
        if not align_path.exists():
            return
        scene.axis_align_matrix = np.loadtxt(str(align_path))

    def _count_frames(self, scene: EmbodiedScanScene, scene_path: Path):
        """Count available RGB and depth frames."""
        rgb_dir = scene_path / "rgb"
        depth_dir = scene_path / "depth"
        if rgb_dir.exists():
            scene.num_rgb_frames = len(list(rgb_dir.iterdir()))
        if depth_dir.exists():
            scene.num_depth_frames = len(list(depth_dir.iterdir()))
