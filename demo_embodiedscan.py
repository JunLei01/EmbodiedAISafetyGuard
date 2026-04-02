"""
EmbodiedScan Safety Reasoning Demo
====================================
Demonstrates the full pipeline using EmbodiedScan/OpenScan scene data:

    EmbodiedScan Data (RGB + Depth + Poses + Mesh)
    + Scene Annotations (annotations.json)
            |
            v
    EmbodiedScanAdapter  ->  Prolog predicates
            |
            v
    SceneGraphBuilder    ->  Grounded facts
            |
            v
    DifferentiableSafetyReasoner  ->  Safety rules + Action checking

Usage:
    python demo_embodiedscan.py
"""

import torch
from scene_graph_builder import SceneGraphBuilder, SceneGraphVisualizer
from differentiable_safety_reasoner import DifferentiableSafetyReasoner
from safety_knowledge_base import SafetyCategory
from embodiedscan_adapter import EmbodiedScanAdapter


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_list_scenes(adapter: EmbodiedScanAdapter):
    """Show all available scenes."""
    print_header("EmbodiedScan Dataset Overview")
    scenes = adapter.list_scenes()
    print(f"\n  Data root: {adapter.data_root}")
    print(f"  Available scenes: {len(scenes)}")
    for name in scenes:
        scene = adapter.load_scene(name)
        print(f"\n  [{name}]")
        print(f"    Type: {scene.scene_type}")
        print(f"    Description: {scene.description}")
        print(f"    Objects: {scene.num_objects}")
        print(f"    RGB frames: {scene.num_rgb_frames}")
        print(f"    Depth frames: {scene.num_depth_frames}")
        print(f"    Camera poses: {len(scene.poses)}")
        print(f"    Has mesh: {scene.has_mesh}")
        if scene.intrinsic:
            intr = scene.intrinsic
            print(f"    Intrinsic: {intr.width}x{intr.height}, "
                  f"fx={intr.fx:.1f}, fy={intr.fy:.1f}")
    return scenes


def demo_scene_loading(adapter: EmbodiedScanAdapter, scene_name: str):
    """Load and inspect a single scene."""
    print_header(f"Scene Loading: {scene_name}")

    scene = adapter.load_scene(scene_name)
    print(f"\n{scene.summary()}")

    # Show Prolog predicate conversion
    predicates = adapter.to_predicates(scene)
    print(f"\n  Prolog predicate output:")
    print(f"  {'─' * 50}")
    for line in predicates.split("\n"):
        print(f"  {line}")

    return scene


def demo_safety_reasoning(adapter: EmbodiedScanAdapter, scene_name: str):
    """Run full safety reasoning on a scene."""
    print_header(f"Safety Reasoning: {scene_name}")

    scene = adapter.load_scene(scene_name)

    # Convert to grounded facts
    grounded_facts, entity_ids = adapter.to_grounded_facts(scene)
    print(f"\n  Entities: {entity_ids}")
    print(f"  Grounded facts: {len(grounded_facts)}")

    # Show high-probability facts
    print(f"\n  High-probability facts (P > 0.5):")
    for key in sorted(grounded_facts.keys()):
        prob = grounded_facts[key].item()
        if prob > 0.5:
            print(f"    {key:45s} = {prob:.4f}")

    # Run safety reasoner
    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=0.3,
        learn_rule_weights=True,
    )
    result = reasoner.reason(grounded_facts, entity_ids)

    print(f"\n{result.summary()}")

    # Show proof trees for top activated rules
    if result.activated_rules:
        print(f"\n  Proof Trees:")
        print(f"  {'─' * 50}")
        for rule in result.activated_rules[:5]:
            if rule.proof_tree:
                for line in rule.proof_tree.to_lines():
                    print(f"    {line}")
                print()

    return result, reasoner


def demo_action_checking(
    adapter: EmbodiedScanAdapter,
    scene_name: str,
    result,
    reasoner: DifferentiableSafetyReasoner,
):
    """Test action safety for a scene."""
    print_header(f"Action Checking: {scene_name}")

    scene = adapter.load_scene(scene_name)

    # Define test actions based on scene type
    if scene_name == "office":
        test_actions = [
            ("pour", "water_bottle", "power_strip",
             "Pour water near power strip"),
            ("place_near", "paper_stack", "desk_lamp",
             "Place paper stack near hot desk lamp"),
            ("move_towards", "coffee_mug", "monitor",
             "Move coffee mug towards monitor"),
            ("move_away", "water_bottle", "power_strip",
             "Move water bottle away from power strip"),
        ]
    elif scene_name == "restroom":
        test_actions = [
            ("walk_through", "person_washing", "water_puddle",
             "Person walks through water puddle"),
            ("touch", "person_washing", "floor_outlet",
             "Wet person touches floor outlet"),
            ("move_away", "cleaning_bottle", "person_washing",
             "Move cleaning bottle away from person"),
        ]
    elif scene_name == "restroom2":
        test_actions = [
            ("hand_to_child", "bleach_bottle", "child",
             "Hand bleach bottle to child"),
            ("place_within_reach", "razor", "child",
             "Place razor within child's reach"),
            ("move_away", "bleach_bottle", "child",
             "Move bleach bottle away from child"),
            ("move_away", "razor", "child",
             "Move razor away from child"),
        ]
    else:
        test_actions = []

    print(f"\n  Testing proposed actions:\n")
    for action, actor, target, description in test_actions:
        violations = reasoner.check_action(result, action, actor, target)

        if violations:
            print(f"  [BLOCKED] {description}")
            print(f"     Action: {action}({actor}, {target})")
            for v in violations:
                print(f"     Violates: [{v.severity.name}] {v.rule_name}")
                print(f"       Reason: {v.grounded_description}")
        else:
            print(f"  [ALLOWED] {description}")
        print()


def demo_scene_comparison(adapter: EmbodiedScanAdapter):
    """Compare safety analysis across all scenes."""
    print_header("Cross-Scene Safety Comparison")

    scenes = adapter.list_scenes()
    results = {}

    for scene_name in scenes:
        scene = adapter.load_scene(scene_name)
        grounded_facts, entity_ids = adapter.to_grounded_facts(scene)

        reasoner = DifferentiableSafetyReasoner(activation_threshold=0.3)
        result = reasoner.reason(grounded_facts, entity_ids)
        results[scene_name] = result

    # Summary table
    print(f"\n  {'Scene':<15} {'Safe?':<8} {'Rules':<8} "
          f"{'Max Severity':<15} {'Categories'}")
    print(f"  {'─' * 70}")

    for scene_name, result in results.items():
        categories = set()
        for rule in result.activated_rules:
            categories.add(rule.category.value)

        cat_str = ", ".join(sorted(categories)) if categories else "none"
        sev_str = result.max_severity.name if result.max_severity else "NONE"

        print(f"  {scene_name:<15} "
              f"{'YES' if result.is_safe else 'NO':<8} "
              f"{result.num_activated:<8} "
              f"{sev_str:<15} "
              f"{cat_str}")

    return results


def demo_gradient_flow(adapter: EmbodiedScanAdapter):
    """Verify gradient flow through the EmbodiedScan pipeline."""
    print_header("Gradient Flow Verification (EmbodiedScan)")

    scene = adapter.load_scene("office")
    grounded_facts, entity_ids = adapter.to_grounded_facts(scene)

    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=0.1,
        learn_rule_weights=True,
    )
    result = reasoner.reason(grounded_facts, entity_ids)

    if not result.activated_rules:
        print("\n  No activated rules to test gradient flow.")
        return

    target_rule = result.activated_rules[0]
    prob = target_rule.probability

    print(f"\n  Target rule: {target_rule.rule_name}")
    print(f"  Hazard: {target_rule.hazard_predicate}")
    print(f"  P = {prob.item():.4f}")
    print(f"  requires_grad = {prob.requires_grad}")

    if prob.requires_grad:
        loss = (1.0 - prob) ** 2
        loss.backward()

        print(f"\n  Gradient flow check:")
        has_grad = False
        for name, param in reasoner.engine.named_parameters():
            if param.grad is not None:
                has_grad = True
                print(f"    dL/d{name} = {param.grad.item():.6f}")

        if has_grad:
            print(f"\n  [OK] Gradients flow end-to-end through EmbodiedScan pipeline!")
        else:
            print(f"\n  [WARN] No gradients reached engine parameters")
    else:
        print(f"\n  [INFO] Probability has no grad — spatial relations may be fixed")


if __name__ == "__main__":
    print("=" * 70)
    print("  EmbodiedScan + Neuro-Symbolic Safety Guardrail Pipeline")
    print("=" * 70)

    # Initialize adapter
    adapter = EmbodiedScanAdapter("data/EmbodiedScan/data")

    # 1. Dataset overview
    scenes = demo_list_scenes(adapter)

    # 2. Load and inspect each scene
    for scene_name in scenes:
        demo_scene_loading(adapter, scene_name)

    # 3. Safety reasoning on each scene
    for scene_name in scenes:
        result, reasoner = demo_safety_reasoning(adapter, scene_name)
        demo_action_checking(adapter, scene_name, result, reasoner)

    # 4. Cross-scene comparison
    demo_scene_comparison(adapter)

    # 5. Gradient flow verification
    demo_gradient_flow(adapter)

    print("\n" + "=" * 70)
    print("  ALL EMBODIEDSCAN DEMOS COMPLETED")
    print("=" * 70)
