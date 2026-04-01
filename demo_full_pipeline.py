"""
Full Pipeline Demo: Differentiable Safety Logic Reasoner
==========================================================
Demonstrates the complete pipeline:

    LLaVa-3D Predicates → Scene Graph → Safety Reasoning → Action Checking
"""

import torch
from scene_graph_builder import SceneGraphBuilder
from safety_knowledge_base import (
    SafetyKnowledgeBase,
    SafetyRuleTemplate,
    SafetyCategory,
    Severity,
    create_embodied_safety_kb,
)
from differentiable_safety_reasoner import (
    DifferentiableSafetyReasoner,
    SafetyReasoningResult,
)
from knowledge_base import Rule, Literal


def demo_kitchen_fire():
    """Demo: Kitchen fire hazard detection."""
    print("=" * 70)
    print("  DEMO: Kitchen Fire Hazard Detection")
    print("=" * 70)

    predicates = """
    % ── Objects in the kitchen ──
    object(gasoline_can, gasoline_can, flammable).
    object(stove, stove, fire_source).
    object(lighter, lighter, fire_source).
    object(cloth, cloth, [flammable, is_cloth]).
    object(pan, pan, container).
    object(water, water, liquid).
    object(power_strip, power_strip, electrical).

    % ── Attributes ──
    attribute(stove, temperature, 200).
    attribute(gasoline_can, temperature, 25).

    % ── 3D Positions ──
    position(gasoline_can, 1.2, 0.5, 0.8).
    position(stove, 1.0, 0.5, 0.8).
    position(lighter, 1.1, 0.6, 0.75).
    position(cloth, 1.15, 0.55, 0.78).
    position(pan, 1.0, 0.6, 0.8).
    position(water, 0.8, 0.5, 1.2).
    position(power_strip, 0.85, 0.3, 1.15).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    print(f"\n  Entities: {entity_ids}")
    print(f"  Grounded facts: {len(grounded_facts)}")

    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=0.3,
        learn_rule_weights=True,
    )
    result = reasoner.reason(grounded_facts, entity_ids)

    print(result.summary())

    if result.activated_rules:
        print(f"\n  Proof Trees:")
        print(f"  {'─' * 50}")
        for rule in result.activated_rules[:3]:
            if rule.proof_tree:
                for line in rule.proof_tree.to_lines():
                    print(f"    {line}")
                print()

    return result


def demo_child_safety():
    """Demo: Child safety in kitchen."""
    print("\n" + "=" * 70)
    print("  DEMO: Child Safety Hazard Detection")
    print("=" * 70)

    predicates = """
    object(child, child, [child, is_child]).
    object(knife, knife, sharp).
    object(stove, stove, [fire_source, is_hot]).
    object(glass, glass, [fragile, sharp]).
    object(medicine_bottle, medicine_bottle, [is_medication, toxic]).

    attribute(knife, sharpness, 0.95).
    attribute(stove, temperature, 180).

    position(child, 2.0, 0.0, 1.0).
    position(knife, 2.1, 0.4, 0.9).
    position(stove, 1.5, 0.5, 1.0).
    position(glass, 2.05, 0.3, 1.05).
    position(medicine_bottle, 2.15, 0.3, 0.95).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    reasoner = DifferentiableSafetyReasoner(activation_threshold=0.2)
    result = reasoner.reason(grounded_facts, entity_ids)

    print(result.summary())

    # Show rules specifically about the child
    child_rules = result.get_rules_involving_entity("child")
    if child_rules:
        print(f"\n  Rules involving 'child': {len(child_rules)}")
        for r in child_rules:
            print(f"    {r.severity.name:8s} | {r.rule_name}: "
                  f"{r.grounded_description}")

    return result


def demo_chemical_electrical():
    """Demo: Chemical and electrical hazards."""
    print("\n" + "=" * 70)
    print("  DEMO: Chemical & Electrical Hazard Detection")
    print("=" * 70)

    predicates = """
    object(worker, worker, [is_person]).
    object(acid_bottle, acid_bottle, [toxic, chemical]).
    object(water_spill, water_spill, [liquid, on_floor]).
    object(power_cable, power_cable, [electrical, is_electrical]).
    object(server_rack, server_rack, [electrical, is_electrical, heavy]).

    attribute(acid_bottle, toxicity, 0.95).

    position(worker, 3.0, 0.0, 2.0).
    position(acid_bottle, 3.1, 0.4, 2.1).
    position(water_spill, 3.05, 0.0, 1.95).
    position(power_cable, 2.9, 0.0, 2.05).
    position(server_rack, 2.8, 0.0, 2.0).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    reasoner = DifferentiableSafetyReasoner(activation_threshold=0.2)
    result = reasoner.reason(grounded_facts, entity_ids)

    print(result.summary())
    return result


def demo_safe_office():
    """Demo: Safe office scenario (should detect no hazards)."""
    print("\n" + "=" * 70)
    print("  DEMO: Safe Office (No Hazards Expected)")
    print("=" * 70)

    predicates = """
    object(person, person, is_person).
    object(laptop, laptop, [electrical, is_electrical]).
    object(mug, mug, container).
    object(notebook, notebook, []).

    position(person, 1.0, 0.0, 1.0).
    position(laptop, 1.0, 0.7, 1.0).
    position(mug, 1.2, 0.7, 1.0).
    position(notebook, 0.8, 0.7, 1.0).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    reasoner = DifferentiableSafetyReasoner(activation_threshold=0.5)
    result = reasoner.reason(grounded_facts, entity_ids)

    print(result.summary())
    print(f"\n  Scene is safe: {result.is_safe}")
    return result


def demo_action_checking(kitchen_result: SafetyReasoningResult):
    """Demo: Check proposed actions against safety rules."""
    print("\n" + "=" * 70)
    print("  DEMO: Action Violation Checking")
    print("=" * 70)

    reasoner = DifferentiableSafetyReasoner()

    print("\n  Testing proposed embodied actions against kitchen safety rules:\n")

    test_actions = [
        ("move_towards", "gasoline_can", "stove",
         "Robot moves gasoline can towards stove"),
        ("place_near", "cloth", "stove",
         "Robot places cloth near stove"),
        ("pour", "water", "power_strip",
         "Robot pours water near power strip"),
        ("move_away", "gasoline_can", "stove",
         "Robot moves gasoline can away from stove"),
        ("grab", "pan", None,
         "Robot grabs pan (safe action)"),
    ]

    for action, actor, target, description in test_actions:
        violations = reasoner.check_action(
            kitchen_result, action, actor, target
        )

        if violations:
            print(f"  [BLOCKED]: {description}")
            print(f"     Action: {action}({actor}"
                  f"{', ' + target if target else ''})")
            for v in violations:
                print(f"     Violates: [{v.severity.name}] {v.rule_name}")
                print(f"       Reason: {v.grounded_description}")
        else:
            print(f"  [ALLOWED]: {description}")

    # Suggest safe alternatives
    print("  Safe action suggestions for 'gasoline_can':")
    suggestions = reasoner.suggest_safe_actions(
        kitchen_result, "gasoline_can"
    )
    for rule_name, actions in suggestions.items():
        print(f"    {rule_name}: {actions}")


def demo_gradient_flow():
    """Demo: Verify gradient flow through the reasoner."""
    print("\n" + "=" * 70)
    print("  DEMO: Gradient Flow Verification")
    print("=" * 70)

    predicates = """
    object(gasoline_can, gasoline_can, flammable).
    object(stove, stove, fire_source).

    position(gasoline_can, 1.2, 0.5, 0.8).
    position(stove, 1.0, 0.5, 0.8).
    """

    builder = SceneGraphBuilder(require_grad_positions=True)
    grounded_facts, entity_ids = builder.build(predicates)

    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=0.1,
        learn_rule_weights=True,
    )
    result = reasoner.reason(grounded_facts, entity_ids)

    print(f"\n  Activated rules: {result.num_activated}")

    # Find a fire hazard rule and backpropagate
    fire_rules = result.get_rules_by_category(SafetyCategory.FIRE)
    if fire_rules:
        target_rule = fire_rules[0]
        prob = target_rule.probability

        print(f"\n  Target: {target_rule.hazard_predicate}")
        print(f"  P = {prob.item():.4f}")
        print(f"  requires_grad = {prob.requires_grad}")

        if prob.requires_grad:
            # Simulate loss: maximize this hazard detection probability
            loss = (1.0 - prob) ** 2
            loss.backward()

            print(f"\n  Gradient flow check:")
            has_grad = False
            for name, param in reasoner.engine.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    print(f"    dL/d{name} = {param.grad.item():.6f}")

            if has_grad:
                print(f"\n  [OK] Gradients flow through the reasoner!")
                print(f"    -> Rule weights are learnable end-to-end")
                print(f"    -> Can train with weak supervision on scene labels")
            else:
                print(f"\n  [WARN] No gradients reached engine parameters")
        else:
            print(f"\n  [WARN] Probability has no grad. "
                  f"Check requires_grad settings.")
    else:
        print("  No fire rules activated for gradient test.")

def demo_custom_rules():
    """Demo: Adding custom domain-specific rules."""
    print("\n" + "=" * 70)
    print("  DEMO: Custom Safety Rules")
    print("=" * 70)

    # Start with default KB
    skb = create_embodied_safety_kb()

    # Add a custom domain-specific rule
    skb.add_template(SafetyRuleTemplate(
        name="robot_arm_near_person",
        category=SafetyCategory.COLLISION,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("robot_collision_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_robot", ["X"]),
                Literal("is_person", ["Y"]),
            ],
            name="robot_arm_near_person",
        ),
        natural_language="Robot X is operating near person Y.",
        prohibited_actions=["extend_arm", "swing", "rotate_fast"],
        mitigating_actions=["slow_down", "stop", "reduce_force"],
    ))

    print(f"\n  Added custom rule: 'robot_arm_near_person'")
    print(f"  Total templates: {len(skb.templates)}")

    # Test with a scene containing a robot
    predicates = """
    object(robot_arm, robot_arm, is_robot).
    object(person1, person, is_person).
    object(knife, knife, sharp).

    position(robot_arm, 1.0, 0.5, 1.0).
    position(person1, 1.2, 0.0, 1.0).
    position(knife, 1.1, 0.4, 1.0).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    reasoner = DifferentiableSafetyReasoner(
        safety_kb=skb,
        activation_threshold=0.2,
    )
    result = reasoner.reason(grounded_facts, entity_ids)

    print(result.summary())

    # Check robot actions
    print("\n  Robot action safety check:")
    for action in ["extend_arm", "move_away", "rotate_fast", "stop"]:
        violations = reasoner.check_action(
            result, action, "robot_arm", "person1"
        )
        status = "[BLOCKED]" if violations else "[ALLOWED]"
        print(f"    {status}: {action}(robot_arm, person1)")
        for v in violations:
            print(f"      → Violates [{v.severity.name}] {v.rule_name}")

def demo_safety_report(kitchen_result: SafetyReasoningResult):
    """Demo: Generate structured safety report."""
    print("\n" + "=" * 70)
    print("  DEMO: Structured Safety Report (JSON)")
    print("=" * 70)

    reasoner = DifferentiableSafetyReasoner()
    report = reasoner.get_safety_report(kitchen_result)

    import json
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    # Run all demos
    kitchen_result = demo_kitchen_fire()
    # child_result = demo_child_safety()
    # chemical_result = demo_chemical_electrical()
    # safe_result = demo_safe_office()

    demo_action_checking(kitchen_result)
    demo_gradient_flow()
    demo_custom_rules()
    demo_safety_report(kitchen_result)

    print("\n" + "=" * 70)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
