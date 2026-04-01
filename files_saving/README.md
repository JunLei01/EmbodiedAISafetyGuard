# 基于神经符号的具身智能安全护栏系统
# Neuro-Symbolic Embodied Safety Guardrail System

## Architecture

```
LLaVa-3D Neural Predicates
    object(1, gasoline_can, flammable).  position(1, 1.2, 0.5, 0.8).
                       │
    ┌──────────────────▼──────────────────────────────────┐
    │  scene_graph_builder.py                              │
    │  Parse + spatial inference (differentiable σ kernels)│
    │  Output: {"near(1,2)": tensor(0.92), ...}           │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────────────┐
    │  safety_knowledge_base.py  (18 rules, 10 hazards)   │
    │  fire_hazard, electrical_hazard, cut_hazard, ...     │
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────────────┐
    │  differentiable_safety_reasoner.py  (可微逻辑推理器) │
    │  DeepProbLog forward chaining → ActivatedSafetyRules│
    └──────────────────┬───────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────────────────────┐
    │  Action Violation Checking                           │
    │  check_action("move_towards", "gas_can", "stove")   │
    │  → ⛔ BLOCKED: violates [CRITICAL] fire_hazard      │
    └──────────────────────────────────────────────────────┘
```

## Quick Start

```python
from scene_graph_builder import SceneGraphBuilder
from differentiable_safety_reasoner import DifferentiableSafetyReasoner

builder = SceneGraphBuilder()
facts, ids = builder.build("object(gas, gas, flammable). object(stove, stove, fire_source). position(gas,1.2,0.5,0.8). position(stove,1.0,0.5,0.8).")

reasoner = DifferentiableSafetyReasoner()
result = reasoner.reason(facts, ids)
print(result.summary())

violations = reasoner.check_action(result, "move_towards", "gas", "stove")
```

## Run Demo
```bash
pip install torch && python demo_full_pipeline.py
```
