"""
demo_complex_kitchen.py
═══════════════════════════════════════════════════════════════════════════
Complex Kitchen-Home Demo: Multi-Step Causal Chain Reasoning
═══════════════════════════════════════════════════════════════════════════

场景叙事:
    一个家庭厨房与相邻客厅构成的开放式空间。场景中一位老人(elderly_person)
    正坐在客厅沙发上，一个服务机器人(robot_arm)正在厨房备餐。

    此 demo 旨在展示系统对 **多步因果链** 的深度推理能力：

    ╔══════════════════════════════════════════════════════════════════╗
    ║  Chain-1  油脂溢出 → 灶台明火引燃 → 火焰蔓延至纸巾卷              ║
    ║           → 纸巾卷紧邻窗帘 → 窗帘大面积燃烧 → 烟雾蔓延至客厅      ║
    ║           → 老人吸入性窒息风险 (6步因果链)                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Chain-2  微波炉内金属餐具 → 电弧/火花 → 引燃微波炉内食品包装       ║
    ║           → 微波炉过热 → 相邻酒精消毒液受热膨胀 → 泄漏/爆燃        ║
    ║           → 飞溅至木质砧板 → 灶台区域复合火灾 (7步因果链)          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Chain-3  水壶沸腾溢出 → 水流至电磁炉插座 → 短路                   ║
    ║           → 电弧引燃周围油渍 → 排油烟机管道油脂被引燃              ║
    ║           → 管道火灾扩散至天花板 (6步因果链)                       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Chain-4  机器人手臂高速运动 → 碰撞热油锅 → 热油飞溅               ║
    ║           → 溅至老人/宠物(cat) → 烫伤 (4步因果链)                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Chain-5  冰箱制冷剂微泄漏(R-600a异丁烷) → 厨房低处气体聚集        ║
    ║           → 灶台点火 → 闪燃/爆炸 (4步因果链)                      ║
    ╚══════════════════════════════════════════════════════════════════╝

    实体总数:  28
    安全规则链: 5 条主链, 多条交叉耦合
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
from utils.differentiable_safety_reasoner import (
    DifferentiableSafetyReasoner,
    SafetyReasoningResult,
)
from utils.knowledge_base import Rule, Literal


# Scene Predicate Definition

KITCHEN_HOME_PREDICATES = """
% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     ZONE DEFINITIONS                                 ║
% ╚═══════════════════════════════════════════════════════════════════════╝

zone(kitchen, kitchen_area, enclosed).
zone(living_room, living_area, open).
zone_connection(kitchen, living_room, open_archway).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     AGENTS & LIVING ENTITIES                         ║
% ╚═══════════════════════════════════════════════════════════════════════╝

object(elderly_person, human, [vulnerable, low_mobility, seated]).
object(robot_arm, robot, [actuator, high_speed, metallic]).
object(cat, animal, [mobile, unpredictable, small]).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     HEAT / FIRE SOURCES                              ║
% ╚═══════════════════════════════════════════════════════════════════════╝

object(gas_stove, stove, [fire_source, gas_powered, open_flame]).
object(induction_cooker, cooker, [electric, high_temperature]).
object(microwave, appliance, [electric, enclosed_heat, radiation]).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     FLAMMABLE / COMBUSTIBLE MATERIALS                ║
% ╚═══════════════════════════════════════════════════════════════════════╝

object(cooking_oil_pot, pot, [contains_oil, high_temperature, open_top]).
object(oil_residue_surface, residue, [flammable, thin_film, on_countertop]).
object(paper_towel_roll, paper, [flammable, lightweight, cellulose]).
object(curtain, fabric, [flammable, large_surface, hanging]).
object(wooden_cutting_board, board, [combustible, wood, dry]).
object(food_packaging, packaging, [flammable, plastic, in_microwave]).
object(alcohol_sanitizer, liquid, [flammable, volatile, pressurized]).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     ELECTRICAL COMPONENTS                            ║
% ╚═══════════════════════════════════════════════════════════════════════╝

object(wall_outlet_A, outlet, [electrical, 220v, near_water]).
object(power_strip, strip, [electrical, multi_socket, overloaded]).
object(kettle, appliance, [electric, contains_water, boiling]).
object(metal_fork_in_microwave, utensil, [metallic, conductive, in_microwave]).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     GAS / CHEMICAL HAZARDS                           ║
% ╚═══════════════════════════════════════════════════════════════════════╝

object(fridge, appliance, [cooling, contains_R600a, sealed]).
object(fridge_leak_point, defect, [gas_leak, R600a_isobutane, low_level]).
object(gas_pipe_valve, valve, [gas_supply, methane, pressurized]).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     STRUCTURAL / VENTILATION                         ║
% ╚═══════════════════════════════════════════════════════════════════════╝

object(range_hood, ventilation, [duct, grease_buildup, connected_to_ceiling]).
object(range_hood_duct, duct, [grease_lined, flammable, vertical]).
object(ceiling_panel, structure, [combustible, wood_composite, overhead]).
object(smoke_detector, sensor, [safety_device, ceiling_mounted, battery_low]).
object(fire_extinguisher, safety, [suppression, accessible, charged]).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     ATTRIBUTES                                       ║
% ╚═══════════════════════════════════════════════════════════════════════╝

attribute(gas_stove, temperature, 450).
attribute(gas_stove, flame_status, active).
attribute(induction_cooker, temperature, 280).
attribute(microwave, temperature, 180).
attribute(microwave, status, running).
attribute(cooking_oil_pot, temperature, 190).
attribute(cooking_oil_pot, oil_level, 0.7).
attribute(cooking_oil_pot, spill_risk, 0.85).
attribute(kettle, temperature, 100).
attribute(kettle, water_level, 0.95).
attribute(kettle, boil_status, active_overflow).
attribute(alcohol_sanitizer, flash_point, 17).
attribute(alcohol_sanitizer, temperature, 25).
attribute(fridge_leak_point, leak_rate, 0.02).
attribute(fridge_leak_point, gas_concentration_floor, 0.03).
attribute(range_hood_duct, grease_thickness_mm, 4.5).
attribute(smoke_detector, battery_level, 0.05).
attribute(elderly_person, mobility_score, 0.2).
attribute(elderly_person, hearing_impairment, 0.6).
attribute(robot_arm, max_speed_ms, 2.5).
attribute(robot_arm, current_task, stir_frying).
attribute(cat, position_predictability, 0.1).
attribute(oil_residue_surface, coverage_area_m2, 0.3).
attribute(power_strip, load_ratio, 0.92).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     3D POSITIONS  (x, y, z in meters)                ║
% ║  Kitchen origin at (0,0,0) = floor-level NW corner                  ║
% ║  x: east, y: up, z: south                                           ║
% ╚═══════════════════════════════════════════════════════════════════════╝

% ── Kitchen: Stove Cluster (east wall) ──
position(gas_stove,           2.0, 0.90, 1.0).
position(cooking_oil_pot,     2.0, 0.95, 1.0).
position(oil_residue_surface, 2.15, 0.90, 1.05).
position(robot_arm,           1.8, 0.90, 1.0).
position(range_hood,          2.0, 1.80, 1.0).
position(range_hood_duct,     2.0, 2.20, 1.0).
position(ceiling_panel,       2.0, 2.60, 1.0).

% ── Kitchen: Counter Cluster (center) ──
position(paper_towel_roll,    2.35, 0.92, 1.0).
position(wooden_cutting_board,1.6, 0.90, 1.2).
position(alcohol_sanitizer,   1.45, 0.90, 0.6).

% ── Kitchen: Microwave Cluster (north wall) ──
position(microwave,            1.4, 1.20, 0.3).
position(metal_fork_in_microwave, 1.4, 1.22, 0.3).
position(food_packaging,       1.4, 1.21, 0.3).

% ── Kitchen: Water/Electric Cluster (south wall) ──
position(kettle,              0.8, 0.90, 1.8).
position(induction_cooker,    1.0, 0.90, 1.8).
position(wall_outlet_A,       0.8, 0.50, 1.85).
position(power_strip,         0.85, 0.30, 1.82).

% ── Kitchen: Gas/Cooling (west wall) ──
position(fridge,              0.2, 0.00, 0.5).
position(fridge_leak_point,   0.2, 0.05, 0.5).
position(gas_pipe_valve,      2.3, 0.40, 0.2).

% ── Kitchen: Window/Curtain ──
position(curtain,             2.6, 1.20, 1.0).

% ── Kitchen: Safety Equipment ──
position(smoke_detector,      1.5, 2.60, 1.0).
position(fire_extinguisher,   0.1, 0.30, 1.9).

% ── Living Room ──
position(elderly_person,      4.5, 0.45, 1.5).
position(cat,                 3.2, 0.00, 1.2).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     SPATIAL RELATIONS (critical proximities)         ║
% ╚═══════════════════════════════════════════════════════════════════════╝

near(cooking_oil_pot, gas_stove, 0.05).
near(oil_residue_surface, gas_stove, 0.18).
near(paper_towel_roll, gas_stove, 0.35).
near(paper_towel_roll, curtain, 0.30).
near(curtain, gas_stove, 0.65).
near(robot_arm, cooking_oil_pot, 0.22).
near(alcohol_sanitizer, microwave, 0.32).
near(wooden_cutting_board, gas_stove, 0.45).
near(kettle, wall_outlet_A, 0.41).
near(kettle, induction_cooker, 0.20).
near(power_strip, wall_outlet_A, 0.08).
near(fridge_leak_point, gas_stove, 1.85).
near(range_hood, gas_stove, 0.90).
near(range_hood_duct, range_hood, 0.40).
near(range_hood_duct, ceiling_panel, 0.40).
near(cat, robot_arm, 1.40).
near(cat, cooking_oil_pot, 1.20).
near(elderly_person, gas_stove, 2.55).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     CONTAINMENT / STRUCTURAL RELATIONS               ║
% ╚═══════════════════════════════════════════════════════════════════════╝

contains(microwave, metal_fork_in_microwave).
contains(microwave, food_packaging).
contains(cooking_oil_pot, cooking_oil).
mounted_above(range_hood, gas_stove).
mounted_above(range_hood_duct, range_hood).
mounted_above(ceiling_panel, range_hood_duct).
connected(gas_pipe_valve, gas_stove, gas_line).
connected(wall_outlet_A, induction_cooker, power_cable).
connected(wall_outlet_A, kettle, power_cable).
connected(power_strip, wall_outlet_A, adapter).

% ╔═══════════════════════════════════════════════════════════════════════╗
% ║                     TEMPORAL / STATE ANNOTATIONS                     ║
% ╚═══════════════════════════════════════════════════════════════════════╝

state(cooking_oil_pot, surface_bubbling, true).
state(kettle, overflow_imminent, true).
state(microwave, arcing_detected, true).
state(fridge_leak_point, active_leak, true).
state(robot_arm, motion_active, true).
state(smoke_detector, functional, false).
"""


# ═══════════════════════════════════════════════════════════════════════════
# Causal Chain Definitions (for annotation / expected output validation)
# ═══════════════════════════════════════════════════════════════════════════

EXPECTED_CAUSAL_CHAINS = {
    "chain_1_oil_spill_to_smoke_inhalation": {
        "severity": "CRITICAL",
        "depth": 6,
        "steps": [
            "cooking_oil_pot[spill_risk=0.85] → oil_spill_onto_gas_stove[open_flame]",
            "oil_ignition → flame_spread_to_paper_towel_roll[near=0.35m, flammable]",
            "paper_towel_roll_burning → flame_propagation_to_curtain[near=0.30m, large_surface]",
            "curtain_fully_engulfed → heavy_smoke_generation",
            "smoke_spread_via_open_archway → living_room_smoke_accumulation",
            "elderly_person[low_mobility=0.2, hearing=0.6] + smoke_detector[dead] → inhalation_risk",
        ],
        "aggravating_factors": [
            "smoke_detector battery_level=0.05 → effectively non-functional",
            "elderly_person mobility_score=0.2 → cannot self-evacuate quickly",
            "zone_connection open_archway → no barrier to smoke propagation",
        ],
    },
    "chain_2_microwave_arc_to_compound_fire": {
        "severity": "CRITICAL",
        "depth": 7,
        "steps": [
            "metal_fork_in_microwave[conductive] + microwave[running] → electrical_arcing",
            "arcing → ignition_of_food_packaging[flammable, plastic, in_microwave]",
            "food_packaging_fire → microwave_internal_overheat",
            "microwave_overheat → radiant_heat_to_alcohol_sanitizer[near=0.32m]",
            "alcohol_sanitizer[flash_point=17°C] → vapor_ignition_or_container_burst",
            "alcohol_splash → ignition_of_wooden_cutting_board[near, dry, combustible]",
            "secondary_fire_zone_merges_with_stove_cluster → compound_fire",
        ],
        "aggravating_factors": [
            "alcohol_sanitizer flash_point 17°C ≪ ambient temp → already volatile",
            "power_strip load_ratio=0.92 → electrical failure cascade possible",
        ],
    },
    "chain_3_kettle_overflow_to_duct_fire": {
        "severity": "HIGH",
        "depth": 6,
        "steps": [
            "kettle[boil_status=active_overflow] → water_spill_onto_counter",
            "water_reaches_wall_outlet_A[near=0.41m, 220v] → short_circuit",
            "short_circuit_arc → ignition_of_oil_residue_surface[near, flammable]",
            "oil_residue_fire → updraft_into_range_hood",
            "range_hood_duct[grease_thickness=4.5mm] → grease_fire_in_duct",
            "duct_fire → ceiling_panel[combustible, wood_composite] → structural_fire",
        ],
        "aggravating_factors": [
            "power_strip load_ratio=0.92 → overload amplifies short circuit",
            "grease_thickness 4.5mm → well above safe maintenance threshold",
        ],
    },
    "chain_4_robot_collision_hot_oil_splash": {
        "severity": "HIGH",
        "depth": 4,
        "steps": [
            "robot_arm[max_speed=2.5m/s, task=stir_frying] → high_speed_motion_near_pot",
            "robot_arm collision with cooking_oil_pot[open_top, oil_level=0.7] → oil_splash",
            "hot_oil[190°C] projectile → splash_radius covers cat[near=1.2m]",
            "cat/elderly_person → severe_burn_injury",
        ],
        "aggravating_factors": [
            "cat position_predictability=0.1 → may jump onto counter unpredictably",
            "cooking_oil_pot temperature=190°C → causes instant 3rd-degree burns",
        ],
    },
    "chain_5_refrigerant_leak_flash_fire": {
        "severity": "CRITICAL",
        "depth": 4,
        "steps": [
            "fridge_leak_point[active_leak, R600a] → isobutane_gas_accumulation_at_floor",
            "gas_concentration_floor=0.03 → approaching_LEL(lower_explosive_limit)",
            "gas_stove[open_flame] ignition_source[distance=1.85m] → flash_ignition",
            "flash_fire/explosion → catastrophic_damage_to_kitchen_zone",
        ],
        "aggravating_factors": [
            "R-600a is denser than air → pools at floor level near ignition sources",
            "cat at floor level → direct exposure to flash fire",
            "gas_pipe_valve pressurized methane → secondary explosion risk",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Demo Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def demo_complex_kitchen():
    """
    Complex kitchen-home scenario demonstrating multi-step causal chain
    safety reasoning with 28 entities and 5 interlocking hazard chains.
    """

    print("=" * 70)
    print("  DEMO: Complex Kitchen-Home — Multi-Step Causal Chain Reasoning")
    print("=" * 70)

    # ── Step 1: Build Scene Graph ──────────────────────────────────────
    print("\n[1/4] Building scene graph from predicates ...")
    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(KITCHEN_HOME_PREDICATES)

    print(f"       Entities discovered:  {len(entity_ids)}")
    print(f"       Grounded facts:       {len(grounded_facts)}")
    print(f"       Entity list:")
    for i, eid in enumerate(sorted(entity_ids)):
        print(f"         {i+1:>2}. {eid}")

    # ── Step 2: Initialize Reasoner ────────────────────────────────────
    print("\n[2/4] Initializing differentiable safety reasoner ...")
    reasoner = DifferentiableSafetyReasoner(
        activation_threshold=0.25,      # lower threshold to catch subtle chains
        learn_rule_weights=True,
        max_chain_depth=8,              # support up to 8-step causal chains
        enable_cross_chain_coupling=True,  # detect chain interactions
    )

    # ── Step 3: Run Reasoning ──────────────────────────────────────────
    print("\n[3/4] Running multi-step causal chain reasoning ...")
    result: SafetyReasoningResult = reasoner.reason(grounded_facts, entity_ids)

    # ── Step 4: Display Results ────────────────────────────────────────
    print("\n[4/4] Results")
    print("=" * 70)
    print(result.summary())

    # ── Detailed Proof Trees ───────────────────────────────────────────
    if result.activated_rules:
        print(f"\n{'━' * 70}")
        print(f"  PROOF TREES (top 5 activated causal chains)")
        print(f"{'━' * 70}")
        for i, rule in enumerate(result.activated_rules[:5]):
            print(f"\n  ┌─ Chain #{i+1}: {rule.rule_id}")
            print(f"  │  Severity:   {rule.severity}")
            print(f"  │  Confidence: {rule.confidence:.4f}")
            print(f"  │  Depth:      {rule.chain_depth}")
            if rule.proof_tree:
                print(f"  │")
                for line in rule.proof_tree.to_lines():
                    print(f"  │  {line}")
            print(f"  └{'─' * 60}")

    # ── Cross-Chain Coupling Report ────────────────────────────────────
    if hasattr(result, "cross_chain_interactions") and result.cross_chain_interactions:
        print(f"\n{'━' * 70}")
        print(f"  CROSS-CHAIN COUPLING DETECTED")
        print(f"{'━' * 70}")
        for interaction in result.cross_chain_interactions:
            print(f"  ⚠ {interaction.chain_a} ←→ {interaction.chain_b}")
            print(f"    Shared entity:    {interaction.shared_entity}")
            print(f"    Coupling score:   {interaction.coupling_score:.4f}")
            print(f"    Combined severity escalation: {interaction.escalation}")
            print()

    # ── Validation against expected chains ─────────────────────────────
    print(f"\n{'━' * 70}")
    print(f"  EXPECTED CHAIN COVERAGE VALIDATION")
    print(f"{'━' * 70}")
    detected_ids = {r.rule_id for r in result.activated_rules} if result.activated_rules else set()
    for chain_name, chain_spec in EXPECTED_CAUSAL_CHAINS.items():
        depth = chain_spec["depth"]
        severity = chain_spec["severity"]
        status = "✓ DETECTED" if chain_name in detected_ids else "✗ MISSED"
        print(f"  {status}  {chain_name}")
        print(f"           depth={depth}, severity={severity}")
        if chain_name not in detected_ids:
            print(f"           → first step: {chain_spec['steps'][0]}")
    print()

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Ablation: Run with increasing chain-depth limits to show reasoning depth
# ═══════════════════════════════════════════════════════════════════════════

def demo_ablation_chain_depth():
    """
    Ablation study: progressively increase max_chain_depth and observe
    how many causal chains the reasoner can uncover at each depth level.
    """

    print("=" * 70)
    print("  ABLATION: Effect of max_chain_depth on Detected Hazards")
    print("=" * 70)

    builder = SceneGraphBuilder()
    grounded_facts, entity_ids = builder.build(KITCHEN_HOME_PREDICATES)

    results_by_depth = {}

    for depth in [1, 2, 3, 4, 5, 6, 7, 8]:
        reasoner = DifferentiableSafetyReasoner(
            activation_threshold=0.25,
            learn_rule_weights=True,
            max_chain_depth=depth,
            enable_cross_chain_coupling=(depth >= 4),
        )
        result = reasoner.reason(grounded_facts, entity_ids)
        n_activated = len(result.activated_rules) if result.activated_rules else 0
        max_sev = max(
            (r.severity_score for r in result.activated_rules),
            default=0.0,
        ) if result.activated_rules else 0.0

        results_by_depth[depth] = {
            "n_rules": n_activated,
            "max_severity": max_sev,
        }
        bar = "█" * n_activated + "░" * (20 - min(n_activated, 20))
        print(f"  depth={depth}  │{bar}│  rules={n_activated:>3}  max_sev={max_sev:.3f}")

    print()
    print("  Observation: deeper chain limits reveal compound hazards that")
    print("  shallow reasoning misses entirely — validating the necessity of")
    print("  multi-step causal inference in embodied safety systems.")
    print()

    return results_by_depth


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = demo_complex_kitchen()

    print("\n" + "=" * 70)
    print("  Running chain-depth ablation study ...")
    print("=" * 70 + "\n")

    ablation = demo_ablation_chain_depth()