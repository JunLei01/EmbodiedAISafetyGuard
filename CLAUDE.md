# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Neuro-Symbolic Embodied Safety Guardrail System** that uses differentiable logic to detect safety hazards in physical environments and prevent unsafe robotic actions.

The pipeline takes VLM/LLaVA-3D entity detection outputs, converts them to neural predicates, performs forward-chaining logical inference with learnable rule weights, and outputs activated safety rules that can block dangerous robot actions.

## Architecture

The system follows a 4-stage pipeline:

```
Input: LLaVA-3D neural predicates (Prolog-style text)
  object(1, gasoline_can, flammable). position(1, 1.2, 0.5, 0.8).
                |
                v
Stage 2: scene_graph_builder.SceneGraphBuilder
  - Parses Prolog-style predicates into structured objects
  - Infers spatial relations (near, touching, above) using sigmoid kernels
  - Maps semantic types to neural predicate probabilities
  Output: { "near(1,2)": tensor(0.92), "flammable(1)": tensor(0.95), ... }
                |
                v
Stage 3: differentiable_safety_reasoner.DifferentiableSafetyReasoner
  - Forward chaining with DeepProbLog-style semiring inference
  - AND = product, OR = noisy-OR
  - Supports learnable rule weights via nn.ParameterDict
  Output: SafetyReasoningResult with activated safety rules
                |
                v
Stage 4: Action Violation Checking
  - reasoner.check_action(result, "move_towards", "gas", "stove")
  - Returns violated rules or empty list if action is safe
```

## Key Files and Responsibilities

| File | Purpose |
|------|---------|
| `scene_graph_builder.py` | Neural predicate parser, spatial relation inference (differentiable), semantic type mapper |
| `differentiable_safety_reasoner.py` | Main reasoner API, forward chaining engine, activated rule extraction |
| `safety_knowledge_base.py` | SafetyRuleTemplate definitions (18 rules, 10 hazard categories), severity levels, prohibited actions |
| `knowledge_base.py` | Core logic programming primitives (Rule, Literal, KnowledgeBase), topological ordering |
| `neural_predicates.py` | Unary and binary neural predicate networks (MLPs outputting probabilities) |
| `logic_layer.py` | DifferentiableLogicLayer with _noisy_or aggregation, ProofTracer for explanations |
| `loss.py` | NeuroSymbolicLoss combining entity risk, scene risk, predicate consistency, and rule weight regularization |

## Running and Testing

```bash
# Run full pipeline demos (kitchen fire, child safety, chemical hazards, action checking)
python demo_full_pipeline.py

# Run scene graph builder demos only
python scene_builder_demo.py

# Run test suite
python test_suite.py
```

## Usage Pattern

```python
from scene_graph_builder import SceneGraphBuilder
from differentiable_safety_reasoner import DifferentiableSafetyReasoner

# Parse LLaVA-3D output and build grounded facts
builder = SceneGraphBuilder()
grounded_facts, entity_ids = builder.build("""
    object(gas, gasoline_can, flammable).
    object(stove, stove, fire_source).
    position(gas, 1.2, 0.5, 0.8).
    position(stove, 1.0, 0.5, 0.8).
""")

# Run safety reasoning
reasoner = DifferentiableSafetyReasoner(activation_threshold=0.3)
result = reasoner.reason(grounded_facts, entity_ids)

# Check if proposed action is safe
violations = reasoner.check_action(result, "move_towards", "gas", "stove")
if violations:
    print(f"Action blocked: violates {violations[0].rule_name}")
```

## Important Technical Details

**Differentiable Spatial Relations**: `SpatialRelationInferrer` uses sigmoid kernels instead of hard thresholds:
```python
P(near(A,B)) = sigmoid(sharpness * (radius - distance(A,B)))
```
This allows gradients to flow from rule outputs back to position estimates.

**Semiring Semantics**:
- Conjunction (AND): `P(A ∧ B) = P(A) × P(B)`
- Disjunction (OR): `P(A ∨ B) = 1 - ∏(1 - P_i)` (noisy-OR)

**Rule Weights**: When `learn_rule_weights=True`, each rule has a learnable weight applied via sigmoid to its output probability. Initialized at 2.0 (sigmoid ≈ 0.88).

**Safety Categories**: FIRE, ELECTRICAL, CUT, BURN, CHEMICAL, FALL, CRUSH, CHILD_SAFETY, SLIP, POISON, COLLISION

**Severity Levels**: LOW (1), MEDIUM (2), HIGH (3), CRITICAL (4)

## Extending the System

To add a new safety rule:
```python
from safety_knowledge_base import SafetyRuleTemplate, SafetyCategory, Severity
from knowledge_base import Rule, Literal

skb.add_template(SafetyRuleTemplate(
    name="my_new_hazard",
    category=SafetyCategory.FIRE,
    severity=Severity.HIGH,
    rule=Rule(
        head=Literal("my_hazard", ["X", "Y"]),
        body=[
            Literal("near", ["X", "Y"]),
            Literal("flammable", ["X"]),
            Literal("is_fire", ["Y"]),
        ],
        name="my_new_hazard",
    ),
    natural_language="Description of the hazard",
    prohibited_actions=["move_towards", "place_near"],
))
```

## Dependencies

Core: `torch` (2.0+)
Optional: `deepproblog/` (submodule, not strictly required for basic operation)

## Output Format

`SafetyReasoningResult` contains:
- `activated_rules`: List of `ActivatedSafetyRule` with probabilities, proof trees, prohibited actions
- `is_safe`: True if no HIGH or CRITICAL rules activated
- `max_severity`: Highest severity level found
- `get_rules_involving_entity(entity_id)`: Filter rules by entity
- `get_prohibited_actions_for_entity(entity_id)`: Map of rule names to prohibited actions
