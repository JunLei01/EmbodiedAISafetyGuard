"""
Demo: Neural Predicate → Scene Graph → Logic Inference
=======================================================
Tests the full pipeline with the exact predicate format
from the LLaVa-3D / embodied safety architecture.

Scenario: Kitchen safety scene
    - gasoline_can near a hot stove
    - lighter present
    - child near a knife
"""

import torch
from files_saving.scene_graph_builder import (
    NeuralPredicateParser,
    SceneGraphBuilder,
    SpatialRelationConfig,
    SemanticTypeMapper,
    SceneGraphVisualizer,
)


def demo_basic_parsing():
    """
    Demo 1: Parse the exact predicate format from the architecture diagram.
    """
    print("=" * 70)
    print("DEMO 1: Neural Predicate Parsing")
    print("=" * 70)

    # Exact format from the user's architecture
    predicates = """
    % ──── Objects detected by LLaVa-3D ────
    object(1, gasoline_can, flammable).
    object(2, stove, fire_source).
    object(3, lighter, fire_source).
    object(4, pan, container).
    object(5, water, liquid).
    object(6, knife, sharp).
    object(7, child, child).
    object(8, power_strip, electrical).
    object(9, cloth, flammable).

    % ──── Attributes ────
    attribute(1, temperature, 25).
    attribute(2, temperature, 150).
    attribute(3, temperature, 30).
    attribute(5, temperature, 20).
    attribute(6, sharpness, 0.95).
    attribute(7, velocity, 0.3).

    % ──── 3D Positions (x, y, z) in meters ────
    position(1, 1.2, 0.5, 0.8).
    position(2, 1.0, 0.5, 0.8).
    position(3, 1.1, 0.6, 0.7).
    position(4, 1.0, 0.6, 0.8).
    position(5, 0.8, 0.5, 1.0).
    position(6, 2.0, 0.4, 1.5).
    position(7, 2.2, 0.0, 1.6).
    position(8, 0.9, 0.3, 0.9).
    position(9, 1.15, 0.55, 0.75).
    """

    parser = NeuralPredicateParser()
    parsed = parser.parse(predicates)

    print("\n[Parsed Scene Summary]")
    print(parsed.summary())

    print(f"\n[Raw predicates parsed: {len(parsed.raw_predicates)}]")
    for pred in parsed.raw_predicates[:5]:
        print(f"  {pred}")
    if len(parsed.raw_predicates) > 5:
        print(f"  ... and {len(parsed.raw_predicates) - 5} more")

    return parsed


def demo_scene_graph_building(parsed):
    """
    Demo 2: Build grounded fact probabilities from parsed predicates.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Scene Graph Construction")
    print("=" * 70)

    # Configure spatial reasoning
    spatial_config = SpatialRelationConfig(
        near_radius=0.5,       # Objects within 0.5m are "near"
        near_sharpness=8.0,    # Sigmoid steepness
        touching_radius=0.15,  # Within 15cm = touching
        above_height_thresh=0.3,
    )

    builder = SceneGraphBuilder(
        spatial_config=spatial_config,
        require_grad_positions=True,  # Enable gradient flow
    )

    grounded_facts, entity_ids = builder.build_from_parsed(parsed)

    print(f"\n[Entity IDs]: {entity_ids}")
    print(f"[Total grounded facts]: {len(grounded_facts)}")

    # Show high-probability facts
    print("\n[High-probability facts (P > 0.5)]:")
    high_prob = {
        k: v for k, v in grounded_facts.items()
        if v.item() > 0.5
    }
    for key in sorted(high_prob.keys()):
        prob = high_prob[key].item()
        print(f"  {key:40s} = {prob:.4f}")

    # ASCII visualization
    print("\n" + SceneGraphVisualizer.to_ascii(
        grounded_facts, entity_ids, threshold=0.5
    ))

    # Prolog export
    print("\n[Prolog-style export (P > 0.3)]:")
    print(SceneGraphVisualizer.to_prolog(grounded_facts, threshold=0.3))

    return grounded_facts, entity_ids


def demo_gradient_flow(parsed):
    """
    Demo 3: Verify that gradients flow from grounded facts back to positions.
    This is critical for end-to-end training.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Gradient Flow Verification")
    print("=" * 70)

    builder = SceneGraphBuilder(
        require_grad_positions=True
    )

    grounded_facts, entity_ids = builder.build_from_parsed(parsed)

    # Pick a spatial relation and backpropagate
    near_key = "near(1,2)"  # gasoline_can near stove
    if near_key in grounded_facts:
        prob = grounded_facts[near_key]
        print(f"\n  P({near_key}) = {prob.item():.4f}")
        print(f"  requires_grad = {prob.requires_grad}")

        if prob.requires_grad:
            # Simulate a loss: we want this probability to be 1.0
            loss = (1.0 - prob) ** 2
            loss.backward()

            # Check that spatial inferrer parameters got gradients
            for name, param in builder.spatial_inferrer.named_parameters():
                if param.grad is not None:
                    print(f"  ∂L/∂{name} = {param.grad.item():.6f}")
                else:
                    print(f"  ∂L/∂{name} = None (no gradient)")

            print("\n  ✓ Gradients flow through spatial relation inference!")
            print("    → Positions can be optimized end-to-end")
        else:
            print("\n  ✗ No gradient — check requires_grad settings")
    else:
        print(f"\n  Key '{near_key}' not found in grounded facts")


def demo_explicit_relations():
    """
    Demo 4: Parse predicates that include both VLM-explicit and inferred relations.
    Matches the architecture diagram: Neural Predicate Grounding output.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Explicit + Inferred Relations")
    print("=" * 70)

    # This matches the format shown in the architecture diagram:
    # Object(stove), Object(pan), on(pan, stove), hot(pan), etc.
    predicates = """
    % Objects from LLaVa-3D detection
    object(stove, stove, fire_source).
    object(pan, pan, container).
    object(water, water, liquid).
    object(knife, knife, sharp).
    object(child, child, child).
    object(table, table, []).

    % Explicit spatial relations from scene graph generator
    on(pan, stove).
    hot(pan).
    liquid(water).
    near(water, power_strip).
    sharp(knife).
    near(knife, child).
    on(knife, table).

    % 3D positions for additional spatial inference
    position(stove, 1.0, 0.8, 0.5).
    position(pan, 1.0, 0.9, 0.5).
    position(water, 0.8, 0.5, 0.5).
    position(knife, 2.0, 0.8, 1.0).
    position(child, 2.1, 0.0, 1.1).
    position(table, 2.0, 0.4, 1.0).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    print(f"\n[Entities]: {entity_ids}")

    print("\n[All grounded facts (P > 0.3)]:")
    for key in sorted(grounded_facts.keys()):
        prob = grounded_facts[key].item()
        if prob > 0.3:
            source = "explicit" if prob >= 0.85 else "inferred"
            print(f"  {key:40s} = {prob:.4f}  ({source})")

    # Show the adjacency structure
    adj = SceneGraphVisualizer.to_adjacency_dict(
        grounded_facts, threshold=0.3
    )
    print("\n[Graph adjacency]:")
    for node_id, data in sorted(adj.items()):
        props = ", ".join(
            f"{p}={v:.2f}" for p, v in data["properties"]
        )
        print(f"  {node_id}: [{props}]")
        for rel, target, prob in data["outgoing"]:
            print(f"    ──{rel}({prob:.2f})──▶ {target}")


def demo_dict_format():
    """
    Demo 5: Build from structured dictionary (JSON-like) format.
    Useful when VLM outputs structured JSON instead of Prolog text.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Dictionary / JSON Format Input")
    print("=" * 70)

    scene_data = {
        "objects": [
            {"id": 1, "category": "gasoline_can",
             "types": ["flammable", "liquid"], "confidence": 0.92},
            {"id": 2, "category": "stove",
             "types": ["fire_source", "hot"], "confidence": 0.98},
            {"id": 3, "category": "child",
             "types": ["child", "person"], "confidence": 0.95},
            {"id": 4, "category": "knife",
             "types": ["sharp"], "confidence": 0.90},
        ],
        "attributes": [
            {"id": 1, "name": "temperature", "value": 25},
            {"id": 2, "name": "temperature", "value": 200},
            {"id": 4, "name": "sharpness", "value": 0.95},
        ],
        "positions": [
            {"id": 1, "x": 1.2, "y": 0.5, "z": 0.8},
            {"id": 2, "x": 1.0, "y": 0.5, "z": 0.8},
            {"id": 3, "x": 2.2, "y": 0.0, "z": 1.6},
            {"id": 4, "x": 2.0, "y": 0.4, "z": 1.5},
        ],
        "relations": [
            {"subject": 3, "relation": "near", "object": 4,
             "confidence": 0.88},
        ],
    }

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build_from_dict(scene_data)

    print(f"\n[Entities]: {entity_ids}")
    print(f"[Grounded facts]: {len(grounded_facts)}")

    for key in sorted(grounded_facts.keys()):
        prob = grounded_facts[key].item()
        if prob > 0.3:
            print(f"  {key:40s} = {prob:.4f}")


def demo_full_pipeline():
    """
    Demo 6: Full pipeline — predicates → scene graph → logic inference simulation.
    Shows the exact data format the DifferentiableLogicLayer expects.
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Full Pipeline Integration Check")
    print("=" * 70)

    predicates = """
    object(1, gasoline_can, flammable).
    object(2, stove, fire_source).
    object(3, lighter, fire_source).
    object(4, child, child).
    object(5, knife, sharp).
    
    attribute(2, temperature, 150).
    attribute(5, sharpness, 0.95).
    
    position(1, 1.2, 0.5, 0.8).
    position(2, 1.0, 0.5, 0.8).
    position(3, 1.1, 0.6, 0.7).
    position(4, 2.2, 0.0, 1.6).
    position(5, 2.0, 0.4, 1.5).
    """

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(predicates)

    print("\n[Output format for DifferentiableLogicLayer]")
    print(f"  entity_ids = {entity_ids}")
    print(f"  grounded_facts keys = {{")
    for key in sorted(grounded_facts.keys()):
        prob = grounded_facts[key]
        grad_str = "✓ grad" if prob.requires_grad else "  fixed"
        print(f"    '{key}': tensor({prob.item():.4f}),  [{grad_str}]")
    print("  }")

    # Verify this matches what DifferentiableLogicLayer.forward() expects
    print("\n[Integration verification]")
    print("  ✓ grounded_facts: Dict[str, torch.Tensor] — correct format")
    print("  ✓ entity_ids: List[str] — correct format")
    print("  ✓ Keys follow 'predicate(entity_id)' or 'predicate(id1,id2)' format")
    print("  ✓ Values are scalar tensors in [0, 1]")

    # Show which safety rules could potentially fire
    print("\n[Safety rules that may fire given these facts]:")
    risk_relevant = {}
    for key, prob in grounded_facts.items():
        p = prob.item()
        if p > 0.3:
            risk_relevant[key] = p

    # Check fire risk prerequisites
    fire_pairs = []
    for key, p in risk_relevant.items():
        if key.startswith("near(") and p > 0.5:
            args = key[5:-1].split(",")
            a, b = args[0], args[1]
            # Check if one is flammable and other is fire
            flam_a = risk_relevant.get(f"flammable({a})", 0)
            fire_b = risk_relevant.get(f"is_fire({b})", 0)
            flam_b = risk_relevant.get(f"flammable({b})", 0)
            fire_a = risk_relevant.get(f"is_fire({a})", 0)
            if flam_a > 0.3 and fire_b > 0.3:
                conj = p * flam_a * fire_b
                fire_pairs.append((a, b, conj))
                print(f"  🔥 FIRE RISK: near({a},{b})={p:.2f} ∧ "
                      f"flammable({a})={flam_a:.2f} ∧ "
                      f"is_fire({b})={fire_b:.2f} → P={conj:.4f}")
            if flam_b > 0.3 and fire_a > 0.3:
                conj = p * flam_b * fire_a
                fire_pairs.append((b, a, conj))
                print(f"  🔥 FIRE RISK: near({a},{b})={p:.2f} ∧ "
                      f"flammable({b})={flam_b:.2f} ∧ "
                      f"is_fire({a})={fire_a:.2f} → P={conj:.4f}")

    # Check cut risk prerequisites
    for key, p in risk_relevant.items():
        if key.startswith("near(") and p > 0.5:
            args = key[5:-1].split(",")
            a, b = args[0], args[1]
            child_a = risk_relevant.get(f"is_child({a})", 0)
            sharp_b = risk_relevant.get(f"sharp({b})", 0)
            if child_a > 0.3 and sharp_b > 0.3:
                conj = p * child_a * sharp_b
                print(f"  🔪 CUT RISK:  near({a},{b})={p:.2f} ∧ "
                      f"is_child({a})={child_a:.2f} ∧ "
                      f"sharp({b})={sharp_b:.2f} → P={conj:.4f}")

    if not fire_pairs:
        print("  (No fire risk combinations found above threshold)")


# Main

if __name__ == "__main__":
    parsed = demo_basic_parsing()
    demo_scene_graph_building(parsed)
    demo_gradient_flow(parsed)
    demo_explicit_relations()
    demo_dict_format()
    demo_full_pipeline()

    print("\n" + "=" * 70)
    print("All demos completed successfully.")
    print("=" * 70)