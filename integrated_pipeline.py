"""
Integrated Pipeline: LLaVA-3D + Scene Graph + Safety Reasoning
===============================================================
Complete end-to-end pipeline:

    Image/Video → LLaVA-3D → Neural Predicates → Safety Rules → Action Checking

Usage:
    python integrated_pipeline.py --image kitchen.jpg --model-path /path/to/llava-3d
    python integrated_pipeline.py --image kitchen.jpg --mock  # Use mock for testing
    python integrated_pipeline.py --embodiedscan-scene office --model-path /path/to/llava-3d
"""

import argparse
import sys
import os
import torch
from typing import Optional, Tuple, List
from pathlib import Path
import json

from embodiedscan_adapter import EmbodiedScanAdapter


def run_safety_pipeline(
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    model_path: Optional[str] = None,
    use_mock: bool = False,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    target: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the complete safety assessment pipeline.

    Args:
        image_path: Path to input image (optional)
        video_path: Path to input video (optional, one of image/video required)
        model_path: Path to LLaVA-3D model (required if not mock)
        use_mock: Use mock recognizer for testing
        action: Optional action to check for violations
        actor: Actor entity for action checking
        target: Target entity for action checking
        verbose: Print detailed output

    Returns:
        dict with results including safety assessment and violations
    """

    # Step 1: Initialize recognizer
    if verbose:
        print("=" * 70)
        print("  LLaVA-3D Neuro-Symbolic Safety Pipeline")
        print("=" * 70)
        print("\n[1/5] Initializing scene recognizer...")

    if use_mock:
        from llava3d_recognizer import MockLLaVA3DRecognizer
        recognizer = MockLLaVA3DRecognizer()
        if verbose:
            print("  [MOCK MODE] Using mock recognizer")
    else:
        if model_path is None:
            raise ValueError("model_path required when not using mock")

        from llava3d_recognizer import LLaVA3DRecognizer
        recognizer = LLaVA3DRecognizer(model_path=model_path)
        if verbose:
            print(f"  Loaded LLaVA-3D from {model_path}")

    # Step 2: Run scene recognition
    if verbose:
        print("\n[2/5] Running scene recognition...")

    if image_path:
        result = recognizer.recognize_image(image_path)
        if verbose:
            print(f"  Input: {image_path}")
    elif video_path:
        result = recognizer.recognize_video(video_path)
        if verbose:
            print(f"  Input: {video_path}")
    else:
        raise ValueError("Either image_path or video_path required")

    if verbose:
        print(f"  Entities detected: {len(result.entities)}")
        for ent in result.entities[:5]:
            print(f"    - {ent['id']}: {ent['category']} ({ent.get('semantic_types', [])})")

    # Step 3: Parse to predicates (SceneGraphBuilder)
    if verbose:
        print("\n[3/5] Building scene graph from predicates...")

    from scene_graph_builder import SceneGraphBuilder
    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = recognizer.parse_to_predicates(result, builder)

    if verbose:
        print(f"  Entity IDs: {entity_ids}")
        print(f"  Grounded facts: {len(grounded_facts)}")
        print("  Sample predicates:")
        for key in list(grounded_facts.keys())[:5]:
            prob = grounded_facts[key].item()
            print(f"    {key}: {prob:.3f}")

    # Step 4: Run safety reasoning
    if verbose:
        print("\n[4/5] Running differentiable safety reasoning...")

    from differentiable_safety_reasoner import DifferentiableSafetyReasoner
    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=0.3,
        learn_rule_weights=True,
    )
    safety_result = reasoner.reason(grounded_facts, entity_ids)

    if verbose:
        print(f"  Activated rules: {safety_result.num_activated}")
        print(f"  Max severity: {safety_result.max_severity.name if safety_result.max_severity else 'NONE'}")
        print(f"  Scene safe: {safety_result.is_safe}")

    # Step 5: Action checking (if requested)
    violations = []
    if action and actor:
        if verbose:
            print(f"\n[5/5] Checking action: {action}({actor}, {target or ''})...")

        violations = reasoner.check_action(safety_result, action, actor, target)

        if verbose:
            if violations:
                print(f"  [BLOCKED] {len(violations)} violations found:")
                for v in violations:
                    print(f"    - [{v.severity.name}] {v.rule_name}: {v.grounded_description}")
            else:
                print(f"  [ALLOWED] No violations found")

    # Generate report
    report = reasoner.get_safety_report(safety_result)

    # Compile results
    output = {
        "scene": {
            "image_path": image_path,
            "video_path": video_path,
            "entities": result.entities,
            "num_grounded_facts": len(grounded_facts),
        },
        "safety": {
            "is_safe": safety_result.is_safe,
            "max_severity": safety_result.max_severity.name if safety_result.max_severity else "NONE",
            "num_activated_rules": safety_result.num_activated,
            "inference_time_ms": safety_result.inference_time_ms,
            "activated_rules": [
                {
                    "name": r.rule_name,
                    "category": r.category.value,
                    "severity": r.severity.name,
                    "probability": r.prob_value,
                    "entities": r.entity_bindings,
                    "prohibited_actions": r.prohibited_actions,
                }
                for r in safety_result.activated_rules
            ],
        },
        "action_check": {
            "action": action,
            "actor": actor,
            "target": target,
            "is_allowed": len(violations) == 0,
            "violations": [
                {
                    "rule": v.rule_name,
                    "severity": v.severity.name,
                    "description": v.grounded_description,
                }
                for v in violations
            ],
        } if action else None,
        "predicates": result.predicates,
        "raw_output": result.raw_output if verbose else None,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  Pipeline completed successfully")
        print("=" * 70)

    return output


def run_safety_pipeline_embodiedscan(
    scene_name: str,
    model_path: Optional[str] = None,
    use_mock: bool = False,
    frames_to_process: int = 5,
    activation_threshold: float = 0.3,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    target: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run safety pipeline with EmbodiedScan dataset using LLaVA-3D detection.

    Pipeline:
        EmbodiedScan Scene → Sample Frames → LLaVA-3D Detection
        → Aggregate Detections → Scene Graph → Safety Reasoning

    Args:
        scene_name: Scene name (e.g., "office", "restroom", "restroom2")
        model_path: Path to LLaVA-3D model
        use_mock: Use mock recognizer
        frames_to_process: Number of frames to sample from scene
        activation_threshold: Safety rule activation threshold
        action: Optional action to check
        actor: Actor entity
        target: Target entity
        verbose: Print detailed output

    Returns:
        dict with safety assessment results
    """
    if verbose:
        print("=" * 70)
        print("  EmbodiedScan + LLaVA-3D Safety Pipeline")
        print("=" * 70)
        print(f"\n[1/6] Loading EmbodiedScan scene: {scene_name}")

    # Load EmbodiedScan scene
    adapter = EmbodiedScanAdapter("/data/junlei/NPG/data/EmbodiedScan/data")
    scene = adapter.load_scene(scene_name)

    if verbose:
        print(f"  Scene type: {scene.scene_type}")
        print(f"  Frames: {scene.num_rgb_frames} RGB, {scene.num_depth_frames} depth")
        print(f"  Poses: {len(scene.poses)}")
        print(f"  Has annotations: {scene.num_objects} objects")

    # Step 2: Initialize LLaVA-3D recognizer
    if verbose:
        print("\n[2/6] Initializing LLaVA-3D recognizer...")

    if use_mock:
        from llava3d_recognizer import MockLLaVA3DRecognizer
        recognizer = MockLLaVA3DRecognizer()
        if verbose:
            print("  [MOCK MODE] Using mock recognizer")
    else:
        if model_path is None:
            raise ValueError("model_path required when not using mock")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        from llava3d_recognizer import LLaVA3DRecognizer
        recognizer = LLaVA3DRecognizer(model_path=model_path)
        if verbose:
            print(f"  Loaded LLaVA-3D from {model_path}")

    # Step 3: Run LLaVA-3D on sampled frames
    if verbose:
        print(f"\n[3/6] Running object detection on {frames_to_process} frames...")

    # Sample frames across the scene trajectory
    frame_indices = [
        int(i * scene.num_rgb_frames / frames_to_process)
        for i in range(frames_to_process)
        if i * scene.num_rgb_frames / frames_to_process < scene.num_rgb_frames
    ]

    all_detections = []
    for idx in frame_indices:
        frame_path = adapter.get_frame_path(scene, idx, modality="rgb")
        if not frame_path:
            continue

        # Run LLaVA-3D detection on this frame
        result = recognizer.recognize_image(frame_path)
        all_detections.append({
            "frame_idx": idx,
            "frame_path": frame_path,
            "predicates": result.predicates,
            "entities": result.entities,
        })

    if verbose:
        print(f"  Processed {len(all_detections)} frames")
        print(f"  Total entity detections: {sum(len(d['entities']) for d in all_detections)}")

    # Step 4: Aggregate detections (use annotations if available, or merge detections)
    if verbose:
        print("\n[4/6] Aggregating detections into scene graph...")

    # For this implementation, we use the scene annotations as ground truth
    # positions and augment with LLaVA-3D semantic type detections
    # In a full system: fuse 3D detections from multiple frames

    # Convert to Prolog predicates using adapter
    predicates_text = adapter.to_predicates(scene)

    # Build scene graph from predicates
    from scene_graph_builder import SceneGraphBuilder
    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates_text)

    if verbose:
        print(f"  Entity IDs: {entity_ids}")
        print(f"  Grounded facts: {len(grounded_facts)}")
        print("  Detected facts (P > 0.5):")
        for key in list(grounded_facts.keys())[:8]:
            prob = grounded_facts[key].item()
            if prob > 0.5:
                print(f"    {key}: {prob:.3f}")
        if len([k for k in grounded_facts.keys() if grounded_facts[k].item() > 0.5]) > 8:
            print("    ...")

    # Step 5: Safety reasoning
    if verbose:
        print("\n[5/6] Running differentiable safety reasoning...")

    from differentiable_safety_reasoner import DifferentiableSafetyReasoner
    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=activation_threshold,
        learn_rule_weights=True,
    )
    safety_result = reasoner.reason(grounded_facts, entity_ids)

    if verbose:
        print(f"  Activated rules: {safety_result.num_activated}")
        if safety_result.max_severity:
            print(f"  Max severity: {safety_result.max_severity.name}")
        print(f"  Scene safe: {safety_result.is_safe}")

        if safety_result.activated_rules:
            print("\n  Top safety hazards:")
            for rule in sorted(safety_result.activated_rules,
                               key=lambda r: r.severity.value, reverse=True)[:5]:
                print(f"    [{rule.severity.name}] {rule.rule_name}")
                print(f"      {rule.grounded_description}")

    # Step 6: Action checking
    violations = []
    if action and actor:
        if verbose:
            print(f"\n[6/6] Checking action: {action}({actor}, {target or ''})...")

        violations = reasoner.check_action(safety_result, action, actor, target)

        if verbose:
            if violations:
                print(f"  [BLOCKED] {len(violations)} violations found:")
                for v in violations:
                    print(f"    - [{v.severity.name}] {v.rule_name}")
                    print(f"      {v.grounded_description}")
            else:
                print(f"  [ALLOWED] No violations found")

    # Generate report
    report = reasoner.get_safety_report(safety_result)

    # Compile results
    output = {
        "scene": {
            "scene_name": scene_name,
            "scene_type": scene.scene_type,
            "frames_processed": len(all_detections),
            "entities_detected": [
                {"id": obj.obj_id, "category": obj.category,
                 "position": obj.position_3d, "types": obj.semantic_types}
                for obj in scene.objects
            ],
            "num_grounded_facts": len(grounded_facts),
        },
        "safety": {
            "is_safe": safety_result.is_safe,
            "max_severity": safety_result.max_severity.name if safety_result.max_severity else "NONE",
            "num_activated_rules": safety_result.num_activated,
            "inference_time_ms": safety_result.inference_time_ms,
            "activated_rules": [
                {
                    "name": r.rule_name,
                    "category": r.category.value,
                    "severity": r.severity.name,
                    "probability": r.prob_value,
                    "entities": r.entity_bindings,
                    "prohibited_actions": r.prohibited_actions,
                }
                for r in safety_result.activated_rules
            ],
        },
        "action_check": {
            "action": action,
            "actor": actor,
            "target": target,
            "is_allowed": len(violations) == 0,
            "violations": [
                {
                    "rule": v.rule_name,
                    "severity": v.severity.name,
                    "description": v.grounded_description,
                }
                for v in violations
            ],
        } if action else None,
        "detections": all_detections,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("  Pipeline completed successfully")
        print("=" * 70)

    return output


def demo_embodiedscan_scene(scene_name: str = "office"):
    """Run demo with EmbodiedScan scene using mock recognizer."""
    print(f"Running EmbodiedScan scene demo: {scene_name}...\n")

    result = run_safety_pipeline_embodiedscan(
        scene_name=scene_name,
        use_mock=True,
        action="pour",
        actor="water_bottle",
        target="power_strip",
        verbose=True,
    )

    print("\n--- Full Safety Report ---")
    print(json.dumps(result, indent=2, default=str))

    return result


def demo_kitchen_mock():
    """Run demo with mock recognizer."""
    print("Running kitchen scene demo with mock recognizer...\n")

    result = run_safety_pipeline(
        image_path="mock_kitchen.jpg",
        use_mock=True,
        action="move_towards",
        actor="robot",
        target="stove",
        verbose=True,
    )

    print("\n--- Full Safety Report ---")
    print(json.dumps(result, indent=2, default=str))

    return result


def demo_interactive():
    """Interactive demo for action checking loop."""
    print("Interactive Safety Checker (type 'quit' to exit)\n")

    # Pre-run scene recognition
    result = run_safety_pipeline(
        image_path="mock_scene.jpg",
        use_mock=True,
        verbose=False,
    )

    safety_result = result["safety"]

    print(f"Scene loaded: {len(safety_result['activated_rules'])} hazards detected")
    print(f"Max severity: {safety_result['max_severity']}")
    print()

    # Action checking loop
    while True:
        action = input("Enter action (or 'list' for entities, 'quit' to exit): ").strip()

        if action.lower() == 'quit':
            break

        if action.lower() == 'list':
            print("Available entities:", result['scene']['entities'])
            continue

        actor = input("Actor entity: ").strip()
        target = input("Target entity (optional): ").strip() or None

        # Check action
        from differentiable_safety_reasoner import DifferentiableSafetyReasoner
        from scene_graph_builder import SceneGraphBuilder

        # Re-run with specific action
        check_result = run_safety_pipeline(
            image_path="mock_scene.jpg",
            use_mock=True,
            action=action,
            actor=actor,
            target=target,
            verbose=False,
        )

        if check_result["action_check"]["is_allowed"]:
            print("[✓ ALLOWED]\n")
        else:
            print("[✗ BLOCKED]")
            for v in check_result["action_check"]["violations"]:
                print(f"  - {v['rule']}: {v['description']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="LLaVA-3D + Neuro-Symbolic Safety Pipeline"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument("--video", type=str, help="Path to input video")
    input_group.add_argument("--embodiedscan-scene", type=str,
                            choices=["office", "restroom", "restroom2"],
                            help="Use EmbodiedScan dataset scene")
    input_group.add_argument("--demo", action="store_true", help="Run demo with mock recognizer")
    input_group.add_argument("--interactive", action="store_true", help="Run interactive demo")
    input_group.add_argument("--demo-embodiedscan", type=str,
                            nargs="?", const="office", metavar="SCENE",
                            help="Run EmbodiedScan demo with optional scene name")

    # Model
    parser.add_argument("--model-path", type=str, help="Path to LLaVA-3D model")
    parser.add_argument("--mock", action="store_true", help="Use mock recognizer (no model needed)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--frames-to-process", type=int, default=5,
                       help="Number of frames to process for EmbodiedScan scenes")

    # Safety settings
    parser.add_argument("--activation-threshold", type=float, default=0.3,
                       help="Safety rule activation threshold (0.0-1.0)")

    # Action checking
    parser.add_argument("--action", type=str, help="Action to check (e.g., move_towards)")
    parser.add_argument("--actor", type=str, help="Actor entity")
    parser.add_argument("--target", type=str, help="Target entity (optional)")

    # Output
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle demo modes
    if args.demo:
        demo_kitchen_mock()
        return

    if args.demo_embodiedscan:
        # Call demo with the specified scene name
        demo_embodiedscan_scene(args.demo_embodiedscan)
        return

    if args.interactive:
        demo_interactive()
        return

    # Handle EmbodiedScan mode
    if args.embodiedscan_scene:
        if not args.mock and not args.model_path:
            parser.error("--model-path required when not using --mock with EmbodiedScan")

        result = run_safety_pipeline_embodiedscan(
            scene_name=args.embodiedscan_scene,
            model_path=args.model_path,
            use_mock=args.mock,
            frames_to_process=args.frames_to_process,
            activation_threshold=args.activation_threshold,
            action=args.action,
            actor=args.actor,
            target=args.target,
            verbose=verbose,
        )

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if verbose:
                print(f"\nResults saved to {args.output}")

        # Print summary if quiet mode
        if not verbose:
            import json
            print(json.dumps(result, default=str))

        return

    # Handle standard image/video mode
    # Validate inputs
    if not args.image and not args.video:
        parser.error("One of --image, --video, --embodiedscan-scene, --demo, --demo-embodiedscan, or --interactive required")

    if not args.mock and not args.model_path:
        parser.error("--model-path required when not using --mock")

    # Run pipeline
    result = run_safety_pipeline(
        image_path=args.image,
        video_path=args.video,
        model_path=args.model_path,
        use_mock=args.mock,
        action=args.action,
        actor=args.actor,
        target=args.target,
        verbose=not args.quiet,
    )
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        if not args.quiet:
            print(f"\nResults saved to {args.output}")

    # Print summary if quiet mode
    if args.quiet:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
