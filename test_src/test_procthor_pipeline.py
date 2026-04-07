"""
ProcTHOR-10k Safety Pipeline Test Suite
========================================
Comprehensive testing of the neuro-symbolic safety pipeline on ProcTHOR scenes.

Complete Pipeline Flow:
    ProcTHOR Scene → Scene Graph → Logical Predicates → Differentiable Logic Reasoning
    → Safety Rule Activation → Action Violation Checking

Usage:
    # Test with subset of scenes
    python test_procthor_pipeline.py --max-scenes 100

    # Test specific room type
    python test_procthor_pipeline.py --room-type kitchen --max-scenes 50

    # Full dataset test
    python test_procthor_pipeline.py --all-scenes

    # Test with action interception
    python test_procthor_pipeline.py --test-actions --max-scenes 50
"""

import argparse
import sys
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict

import torch
import numpy as np

# Import neuro-symbolic safety pipeline components
from utils.scene_graph_builder import SceneGraphBuilder
from utils.differentiable_safety_reasoner import DifferentiableSafetyReasoner
from utils.safety_knowledge_base import SafetyCategory, Severity, create_embodied_safety_kb

# Import the adapter
from procthor_adapter import (
    ProcTHORAdapter,
    ProcTHORScene,
    PROCTHOR_SEMANTIC_MAPPING,
)


@dataclass
class TestResult:
    """Record of a single test scene with full pipeline details."""
    scene_id: str
    room_type: str
    num_objects: int
    hazard_objects: int
    activated_rules: int
    critical_rules: int
    high_rules: int
    medium_rules: int
    low_rules: int
    is_safe: bool
    violations_found: int
    test_duration_ms: float
    grounded_facts_count: int = 0
    spatial_relations_count: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionTestResult:
    """Record of a single action interception test."""
    scene_id: str
    action: str
    actor: str
    target: str
    should_intercept: bool
    did_intercept: bool
    violated_rules: List[str] = field(default_factory=list)


@dataclass
class TestSuiteReport:
    """Aggregate test results."""
    total_scenes: int = 0
    successful_scenes: int = 0
    failed_scenes: int = 0
    room_type_breakdown: Dict[str, int] = field(default_factory=dict)
    avg_objects_per_scene: float = 0.0
    avg_activated_rules: float = 0.0
    avg_hazard_objects: float = 0.0
    avg_grounded_facts: float = 0.0
    avg_spatial_relations: float = 0.0

    # Action interception stats
    total_action_tests: int = 0
    successful_interceptions: int = 0
    missed_interceptions: int = 0
    false_positives: int = 0

    # Hazard type coverage
    hazard_type_counts: Dict[str, int] = field(default_factory=dict)

    # Detailed results
    results: List[TestResult] = field(default_factory=list)
    action_results: List[ActionTestResult] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)

    def print_summary(self):
        """Print comprehensive test suite summary."""
        print("\n" + "=" * 70)
        print("  ProcTHOR-10k Neuro-Symbolic Safety Pipeline Test Report")
        print("=" * 70)
        print(f"\n[Pipeline Flow]")
        print("  ProcTHOR Scene → Scene Graph → Logical Predicates → Differentiable Logic")
        print("  → Safety Rule Activation → Action Violation Checking")

        print(f"\n[Scenes Tested]")
        print(f"  Total: {self.total_scenes}")
        print(f"  Successful: {self.successful_scenes}")
        print(f"  Failed: {self.failed_scenes}")

        print(f"\n[Room Type Breakdown]")
        for room_type, count in sorted(self.room_type_breakdown.items()):
            print(f"  {room_type}: {count}")

        print(f"\n[Averages per Scene]")
        print(f"  Objects: {self.avg_objects_per_scene:.1f}")
        print(f"  Hazard objects: {self.avg_hazard_objects:.1f}")
        print(f"  Grounded facts: {self.avg_grounded_facts:.1f}")
        print(f"  Spatial relations: {self.avg_spatial_relations:.1f}")
        print(f"  Activated rules: {self.avg_activated_rules:.1f}")

        print(f"\n[Hazard Type Coverage]")
        for hazard_type, count in sorted(self.hazard_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {hazard_type}: {count}")

        if self.total_action_tests > 0:
            print(f"\n[Action Interception Performance]")
            print(f"  Total action tests: {self.total_action_tests}")
            print(f"  Successful interceptions: {self.successful_interceptions}")
            print(f"  Missed interceptions: {self.missed_interceptions}")
            print(f"  False positives: {self.false_positives}")
            if self.total_action_tests > 0:
                precision = self.successful_interceptions / max(self.successful_interceptions + self.false_positives, 1)
                recall = self.successful_interceptions / max(self.successful_interceptions + self.missed_interceptions, 1)
                print(f"  Precision: {100*precision:.1f}%")
                print(f"  Recall: {100*recall:.1f}%")

        print(f"\n[Errors ({len(self.errors)})]")
        for scene_id, error in self.errors[:5]:
            print(f"  {scene_id}: {error}")
        if len(self.errors) > 5:
            print(f"  ... and {len(self.errors) - 5} more")

    def save_json(self, path: str):
        """Save report to JSON."""
        data = {
            "summary": {
                "total_scenes": self.total_scenes,
                "successful_scenes": self.successful_scenes,
                "failed_scenes": self.failed_scenes,
                "room_type_breakdown": self.room_type_breakdown,
                "avg_objects_per_scene": self.avg_objects_per_scene,
                "avg_activated_rules": self.avg_activated_rules,
                "avg_hazard_objects": self.avg_hazard_objects,
                "avg_grounded_facts": self.avg_grounded_facts,
                "avg_spatial_relations": self.avg_spatial_relations,
                "hazard_type_counts": self.hazard_type_counts,
                "total_action_tests": self.total_action_tests,
                "successful_interceptions": self.successful_interceptions,
                "missed_interceptions": self.missed_interceptions,
                "false_positives": self.false_positives,
            },
            "results": [r.to_dict() for r in self.results],
            "action_results": [
                {
                    "scene_id": r.scene_id,
                    "action": r.action,
                    "actor": r.actor,
                    "target": r.target,
                    "should_intercept": r.should_intercept,
                    "did_intercept": r.did_intercept,
                    "violated_rules": r.violated_rules,
                }
                for r in self.action_results
            ],
            "errors": self.errors,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nReport saved to {path}")


def run_neuro_symbolic_pipeline(
    scene: ProcTHORScene,
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
        verbose: Print detailed output

    Returns:
        Tuple of (pipeline info dict, SafetyReasoningResult)
    """
    adapter = ProcTHORAdapter([])  # Dummy adapter for helper methods
    adapter._scene_map = {scene.scene_id: None}  # Minimal setup

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
        rel in k for rel in ['near(', 'touching(', 'above(', 'below('])
    ]

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

    reasoner = DifferentiableSafetyReasoner(activation_threshold=0.3)

    if verbose:
        print(f"    - Running forward chaining inference...")

    result = reasoner.reason(grounded_facts, entity_ids)

    if verbose:
        print(f"    ✓ Inference complete")
        print(f"\n  [Stage 4/4] Safety Rule Extraction")
        print(f"    - Total activated rules: {result.total_activated}")
        print(f"    - Critical: {result.critical_count} | High: {result.high_count} | "
              f"Medium: {result.medium_count} | Low: {result.low_count}")

        if result.activated_rules:
            print(f"\n    Top activated rules:")
            for rule in result.activated_rules[:3]:
                desc = rule.natural_language[:50] + "..." if len(rule.natural_language) > 50 else rule.natural_language
                print(f"      [{rule.severity.name}] P={rule.probability:.3f} | {rule.rule_name}")
                print(f"        {desc}")

    pipeline_info = {
        "grounded_facts_count": len(grounded_facts),
        "spatial_relations_count": len(spatial_relations),
        "entity_ids": entity_ids,
    }

    return pipeline_info, result


def test_basic_pipeline(dataset, scene_ids: List[str], verbose: bool = False) -> TestSuiteReport:
    """
    Test the complete neuro-symbolic pipeline on a set of scenes.

    Args:
        dataset: ProcTHOR dataset split
        scene_ids: List of scene IDs to test
        verbose: Print per-scene details

    Returns:
        TestSuiteReport with aggregated results
    """
    adapter = ProcTHORAdapter(dataset)
    report = TestSuiteReport(
        total_scenes=len(scene_ids),
    )

    # Track hazard type coverage
    hazard_counts = defaultdict(int)

    for i, scene_id in enumerate(scene_ids):
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Scene [{i+1}/{len(scene_ids)}]: {scene_id}")
            print(f"{'='*70}")

        start_time = time.time()

        try:
            # Load scene
            scene = adapter.load_scene(scene_id)

            if verbose:
                print(f"\n  Scene Summary:")
                print(f"    - Room type: {scene.room_type}")
                print(f"    - Objects: {scene.num_objects}")
                print(f"    - Hazard objects: {len(scene.hazard_objects)}")

            # Run complete neuro-symbolic pipeline
            pipeline_info, safety_result = run_neuro_symbolic_pipeline(scene, verbose=verbose)

            duration_ms = (time.time() - start_time) * 1000

            # Update hazard counts
            for obj in scene.hazard_objects:
                for hazard_type in obj.semantic_types:
                    if hazard_type in ["sharp", "hot", "fire_source", "chemical", "toxic", "electrical", "flammable"]:
                        hazard_counts[hazard_type] += 1

            # Count rules by severity
            critical_rules = sum(1 for r in safety_result.activated_rules if r.severity.name == "CRITICAL")
            high_rules = sum(1 for r in safety_result.activated_rules if r.severity.name == "HIGH")
            medium_rules = sum(1 for r in safety_result.activated_rules if r.severity.name == "MEDIUM")
            low_rules = sum(1 for r in safety_result.activated_rules if r.severity.name == "LOW")

            test_result = TestResult(
                scene_id=scene_id,
                room_type=scene.room_type,
                num_objects=scene.num_objects,
                hazard_objects=len(scene.hazard_objects),
                activated_rules=safety_result.num_activated,
                critical_rules=critical_rules,
                high_rules=high_rules,
                medium_rules=medium_rules,
                low_rules=low_rules,
                is_safe=safety_result.is_safe,
                violations_found=0,
                test_duration_ms=duration_ms,
                grounded_facts_count=pipeline_info["grounded_facts_count"],
                spatial_relations_count=pipeline_info["spatial_relations_count"],
            )

            report.results.append(test_result)
            report.successful_scenes += 1
            report.room_type_breakdown[scene.room_type] = report.room_type_breakdown.get(scene.room_type, 0) + 1

            if verbose:
                print(f"\n  Test completed in {duration_ms:.1f}ms")

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            import traceback
            error_detail = traceback.format_exc()

            report.errors.append((scene_id, f"{error_msg}\n{error_detail}"))
            report.failed_scenes += 1

            report.results.append(TestResult(
                scene_id=scene_id,
                room_type="error",
                num_objects=0,
                hazard_objects=0,
                activated_rules=0,
                critical_rules=0,
                high_rules=0,
                medium_rules=0,
                low_rules=0,
                is_safe=False,
                violations_found=0,
                test_duration_ms=duration_ms,
                error=error_msg,
            ))

            if verbose:
                print(f"\n  ERROR: {error_msg}")
                print(f"  {error_detail}")

    # Compute averages
    if report.successful_scenes > 0:
        report.avg_objects_per_scene = sum(r.num_objects for r in report.results if not r.error) / report.successful_scenes
        report.avg_activated_rules = sum(r.activated_rules for r in report.results if not r.error) / report.successful_scenes
        report.avg_hazard_objects = sum(r.hazard_objects for r in report.results if not r.error) / report.successful_scenes
        report.avg_grounded_facts = sum(r.grounded_facts_count for r in report.results if not r.error) / report.successful_scenes
        report.avg_spatial_relations = sum(r.spatial_relations_count for r in report.results if not r.error) / report.successful_scenes

    report.hazard_type_counts = dict(hazard_counts)

    return report


def test_action_interception(dataset, scene_ids: List[str], verbose: bool = False) -> TestSuiteReport:
    """
    Test action interception using the neuro-symbolic pipeline.

    Tests scenarios like:
    - Knife near person (child safety)
    - Water near electrical (electrocution)
    - Flammable near fire source
    - Heavy object near person (crush)
    - Chemical near child (poisoning)
    """
    adapter = ProcTHORAdapter(dataset)
    reasoner = DifferentiableSafetyReasoner()

    report = TestSuiteReport(
        total_scenes=len(scene_ids),
    )

    # Define hazardous action test cases: (actor_category, target_category, action, should_intercept, description)
    test_cases = [
        ("knife", "person", "move_towards", True, "Moving knife towards person"),
        ("stove", "cloth", "place_near", True, "Placing cloth near stove"),
        ("bleach", "child", "hand_to_child", True, "Handing bleach to child"),
        ("water", "tv", "pour", True, "Pouring water near TV"),
        ("water", "outlet", "pour", True, "Pouring water near outlet"),
        ("pan", "person", "move_towards", True, "Moving hot pan towards person"),
        ("gas_can", "stove", "move_towards", True, "Moving gas can towards stove"),
        ("book", "shelf", "place_on", False, "Placing book on shelf"),
        ("cup", "table", "place_on", False, "Placing cup on table"),
    ]

    for i, scene_id in enumerate(scene_ids):
        if verbose:
            print(f"\n[Scene {i+1}/{len(scene_ids)}] {scene_id}")

        try:
            # Load scene and run full pipeline
            scene = adapter.load_scene(scene_id)
            pipeline_info, safety_result = run_neuro_symbolic_pipeline(scene, verbose=False)

            # Count objects by category for test case matching
            obj_by_category = defaultdict(list)
            for obj in scene.objects:
                obj_by_category[obj.category.lower()].append(obj)

            tests_run = 0

            for actor_cat, target_cat, action, should_intercept, description in test_cases:
                # Find matching objects in scene
                actor_matches = obj_by_category.get(actor_cat, [])
                target_matches = obj_by_category.get(target_cat, [])

                if actor_matches and target_matches:
                    actor = actor_matches[0]
                    target = target_matches[0]

                    tests_run += 1
                    report.total_action_tests += 1

                    # Check action using the safety result
                    violations = reasoner.check_action(safety_result, action, actor.obj_id, target.obj_id)
                    did_intercept = len(violations) > 0

                    violated_rule_names = [v.rule_name for v in violations]

                    action_result = ActionTestResult(
                        scene_id=scene_id,
                        action=action,
                        actor=actor.category,
                        target=target.category,
                        should_intercept=should_intercept,
                        did_intercept=did_intercept,
                        violated_rules=violated_rule_names,
                    )
                    report.action_results.append(action_result)

                    if should_intercept and did_intercept:
                        report.successful_interceptions += 1
                        if verbose:
                            print(f"  ✓ CORRECTLY BLOCKED: {description}")
                            if violations:
                                print(f"    Rules: {[v.rule_name for v in violations]}")
                    elif should_intercept and not did_intercept:
                        report.missed_interceptions += 1
                        if verbose:
                            print(f"  ✗ MISSED: {description} (should have been blocked)")
                    elif not should_intercept and did_intercept:
                        report.false_positives += 1
                        if verbose:
                            print(f"  ! FALSE POSITIVE: {description} (wrongly blocked)")
                            if violations:
                                print(f"    Rules: {[v.rule_name for v in violations]}")
                    else:
                        if verbose:
                            print(f"  ✓ Correctly allowed: {description}")

            if tests_run > 0:
                report.successful_scenes += 1
            else:
                # No applicable tests in this scene, count as success
                report.successful_scenes += 1
                if verbose:
                    print(f"  (No matching hazard objects for action tests)")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            report.errors.append((scene_id, error_msg))
            report.failed_scenes += 1
            if verbose:
                print(f"  ERROR: {error_msg}")

    return report


def get_scene_ids_from_dataset(dataset_split, max_scenes: Optional[int] = None) -> List[str]:
    """
    Extract scene IDs from ProcTHOR dataset with proper handling of different formats.
    """
    scene_ids = []

    # Try to iterate through dataset
    try:
        # If it supports len() and indexing
        total = len(dataset_split)
        count = min(max_scenes, total) if max_scenes else total

        for i in range(count):
            try:
                scene = dataset_split[i]
                if isinstance(scene, dict):
                    # Try common ID field names
                    for key in ["id", "scene_id", "name", "sceneId", "scene_name"]:
                        if key in scene:
                            scene_ids.append(scene[key])
                            break
                    else:
                        # No ID found, use index
                        scene_ids.append(f"scene_{i}")
                else:
                    scene_ids.append(f"scene_{i}")
            except Exception as e:
                print(f"Warning: Could not extract scene {i}: {e}")
                continue

    except (TypeError, AttributeError):
        # Fall back to iteration
        for i, scene in enumerate(dataset_split):
            if max_scenes and len(scene_ids) >= max_scenes:
                break
            if isinstance(scene, dict):
                for key in ["id", "scene_id", "name", "sceneId", "scene_name"]:
                    if key in scene:
                        scene_ids.append(scene[key])
                        break
                else:
                    scene_ids.append(f"scene_{i}")
            else:
                scene_ids.append(f"scene_{i}")

    return scene_ids


def main():
    parser = argparse.ArgumentParser(
        description="Test ProcTHOR-10k neuro-symbolic safety pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. Scene Graph Construction - Parse ProcTHOR objects into scene graph
  2. Logical Predicate Generation - Convert to Prolog-style predicates
  3. Differentiable Logic Reasoning - Forward chaining with neural predicates
  4. Safety Rule Extraction - Get activated safety rules with probabilities
  5. Action Violation Checking - Check if actions violate safety rules

Examples:
  # Test 100 scenes with basic pipeline
  python test_procthor_pipeline.py --max-scenes 100

  # Test kitchen scenes only
  python test_procthor_pipeline.py --room-type kitchen --max-scenes 50

  # Test with action interception
  python test_procthor_pipeline.py --test-actions --max-scenes 50

  # Full dataset test
  python test_procthor_pipeline.py --all-scenes
        """
    )
    parser.add_argument("--max-scenes", type=int, default=100,
                       help="Maximum number of scenes to test (default: 100)")
    parser.add_argument("--room-type", type=str, default=None,
                       help="Filter by room type (kitchen, bedroom, bathroom, living_room)")
    parser.add_argument("--all-scenes", action="store_true",
                       help="Test all scenes in the dataset")
    parser.add_argument("--test-actions", action="store_true",
                       help="Run action interception tests using the safety reasoner")
    parser.add_argument("--output", type=str, default="procthor_test_report.json",
                       help="Output JSON file for results (default: procthor_test_report.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print per-scene details and pipeline stages")
    parser.add_argument("--no-prior", action="store_true",
                       help="Run without prior library (uses mock scenes for testing)")

    args = parser.parse_args()

    print("=" * 70)
    print("  ProcTHOR-10k Neuro-Symbolic Safety Pipeline Testing")
    print("=" * 70)
    print("\n[Pipeline Flow]")
    print("  ProcTHOR Scene → Scene Graph → Logical Predicates")
    print("  → Differentiable Logic Reasoning → Safety Rule Activation")
    print("  → Action Violation Checking")

    # Load dataset
    train_data = None
    if not args.no_prior:
        try:
            import prior
            print("\n[Dataset Loading]")
            print("  Loading ProcTHOR-10k dataset...")
            dataset = prior.load_dataset("procthor-10k")
            train_data = dataset["train"]
            print(f"  ✓ Loaded {len(train_data)} scenes")

            # Debug: check first scene structure
            if args.verbose:
                first_scene = train_data[0]
                if isinstance(first_scene, dict):
                    print(f"\n  First scene keys: {list(first_scene.keys())[:10]}")

        except ImportError:
            print("  Warning: 'prior' not installed. Using mock mode.")
            args.no_prior = True

    # Create mock data if needed
    if train_data is None:
        print("\n  Using mock ProcTHOR scenes...")
        train_data = create_mock_procthor_scenes()

    # Select scenes to test
    print("\n[Scene Selection]")
    if args.room_type:
        # For room type filtering, we need to examine scenes
        adapter = ProcTHORAdapter(train_data)
        matching_ids = []

        # Get all scene IDs
        all_ids = get_scene_ids_from_dataset(train_data)

        for scene_id in all_ids:
            try:
                scene = adapter.load_scene(scene_id)
                if scene.room_type == args.room_type.lower():
                    matching_ids.append(scene_id)
                if len(matching_ids) >= args.max_scenes:
                    break
            except Exception:
                continue

        scene_ids = matching_ids
        print(f"  Found {len(scene_ids)} {args.room_type} scenes")

    elif args.all_scenes:
        scene_ids = get_scene_ids_from_dataset(train_data)
        print(f"  Testing all {len(scene_ids)} available scenes")
    else:
        scene_ids = get_scene_ids_from_dataset(train_data, max_scenes=args.max_scenes)
        print(f"  Testing {len(scene_ids)} scenes (max {args.max_scenes})")

    if not scene_ids:
        print("\n  ERROR: No scenes found to test!")
        print("  Try running with --no-prior to use mock data,")
        print("  or check that the ProcTHOR dataset is properly installed.")
        sys.exit(1)

    # Run tests
    if args.test_actions:
        print("\n" + "=" * 70)
        print("  Running Action Interception Tests")
        print("=" * 70)
        report = test_action_interception(train_data, scene_ids, verbose=args.verbose)
    else:
        print("\n" + "=" * 70)
        print("  Running Basic Pipeline Tests")
        print("=" * 70)
        report = test_basic_pipeline(train_data, scene_ids, verbose=args.verbose)

    # Print and save results
    report.print_summary()
    report.save_json(args.output)


def create_mock_procthor_scenes() -> List[Dict]:
    """Create synthetic ProcTHOR-style scenes for testing without the dataset."""
    mock_scenes = []

    for i in range(100):
        room_type = ["kitchen", "bedroom", "bathroom", "living_room"][i % 4]
        scene_id = f"train_{i}"

        objects = []

        if room_type == "kitchen":
            objects = [
                {"objectType": "stove", "position": [2.0, 0.8, 1.0]},
                {"objectType": "sink", "position": [1.5, 0.8, 1.5]},
                {"objectType": "knife", "position": [1.8, 0.9, 1.2]},
                {"objectType": "refrigerator", "position": [0.5, 1.2, 0.5]},
                {"objectType": "counter", "position": [2.0, 0.8, 1.5]},
                {"objectType": "pan", "position": [2.1, 0.9, 1.0]},
                {"objectType": "person", "position": [3.0, 0.0, 2.0]},
            ]
        elif room_type == "bedroom":
            objects = [
                {"objectType": "bed", "position": [2.0, 0.5, 2.0]},
                {"objectType": "dresser", "position": [0.5, 0.8, 0.5]},
                {"objectType": "lamp", "position": [1.0, 0.6, 1.0]},
                {"objectType": "window", "position": [1.5, 1.0, 0.0]},
                {"objectType": "book", "position": [0.8, 0.9, 0.8]},
            ]
        elif room_type == "bathroom":
            objects = [
                {"objectType": "toilet", "position": [1.0, 0.5, 1.0]},
                {"objectType": "sink", "position": [0.5, 0.8, 0.5]},
                {"objectType": "bathtub", "position": [2.0, 0.5, 1.5]},
                {"objectType": "mirror", "position": [0.5, 1.2, 0.5]},
                {"objectType": "bleach", "position": [0.6, 0.9, 0.6]},
                {"objectType": "child", "position": [1.5, 0.0, 1.5]},
            ]
        else:  # living_room
            objects = [
                {"objectType": "sofa", "position": [1.5, 0.5, 1.5]},
                {"objectType": "tv", "position": [0.5, 0.8, 0.5]},
                {"objectType": "coffee_table", "position": [1.5, 0.4, 1.0]},
                {"objectType": "lamp", "position": [2.5, 0.8, 2.5]},
                {"objectType": "water", "position": [1.0, 0.0, 1.0]},
                {"objectType": "outlet", "position": [0.5, 0.3, 0.5]},
            ]

        mock_scenes.append({
            "id": scene_id,
            "roomType": room_type,
            "description": f"Mock {room_type}",
            "objects": objects,
        })

    return mock_scenes


if __name__ == "__main__":
    main()
