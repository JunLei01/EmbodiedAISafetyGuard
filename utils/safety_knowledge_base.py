"""
Embodied Safety Knowledge Base
================================
Hierarchical safety rules for the differentiable logic reasoner.

Architecture position:
    高层安全规则 (High-Level Safety Knowledge)
         ↓
    ┌─────────────────────────────────────────┐
    │  >>> THIS MODULE <<<                    │
    │  SafetyKnowledgeBase                    │
    │  SafetyRuleTemplate                     │
    │  SafetyRuleParser                       │
    └─────────────────────────────────────────┘
         ↓
    可微逻辑推理器 (Differentiable Logic Reasoner)

Rule hierarchy (3 layers):
    Layer 1 — Base predicates (from scene graph / neural predicates):
        near(X, Y), flammable(X), sharp(X), is_fire(Y), ...
    
    Layer 2 — Hazard condition rules (intermediate derived predicates):
        fire_hazard(X, Y) :- near(X, Y), flammable(X), is_fire(Y).
        cut_hazard(X, Y)  :- near(X, Y), sharp(Y), is_child(X).
        
    Layer 3 — Scene safety rules (activated rules for specific entity pairs):
        scene_rule(fire_safety, X, Y) :- fire_hazard(X, Y).
        scene_rule(child_safety, X, Y) :- cut_hazard(X, Y).
    
    Layer 4 — Action-violation rules (for downstream action checking):
        violates(Action, Rule) :- action_moves(Action, X, Y), 
                                  scene_rule(fire_safety, X, Y).

This design ensures:
    1. Safety rules are DISCOVERED (not just risks quantified)
    2. Each rule carries its provenance (which objects, which conditions)
    3. The output format directly supports action-violation checking
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum, auto

# Import from existing knowledge_base module
from utils.knowledge_base import KnowledgeBase, Rule, Literal


# Part 1: Safety Rule Data Structures

class SafetyCategory(Enum):
    """Categories of safety hazards."""
    FIRE           = "fire"
    ELECTRICAL     = "electrical"
    CUT            = "cut"
    BURN           = "burn"
    CHEMICAL       = "chemical"
    FALL           = "fall"
    CRUSH          = "crush"
    CHILD_SAFETY   = "child_safety"
    SLIP           = "slip"
    POISON         = "poison"
    COLLISION      = "collision"


class Severity(Enum):
    """Severity level of safety hazards."""
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class SafetyRuleTemplate:
    """
    A high-level safety rule template.
    
    Template rules are parameterized; the reasoner grounds them
    against specific scene entities to produce activated rules.
    
    Example:
        name:     "fire_near_flammable"
        category: FIRE
        severity: CRITICAL
        template: fire_hazard(X, Y) :- near(X, Y), flammable(X), is_fire(Y).
        natural_language: "Keep flammable objects away from fire sources."
        prohibited_actions: ["move_towards", "place_near"]
    """
    name: str
    category: SafetyCategory
    severity: Severity
    rule: Rule                              # The logic rule
    natural_language: str                   # Human-readable description
    prohibited_actions: List[str] = field(default_factory=list)
    # Actions that would violate this rule if performed on involved entities
    mitigating_actions: List[str] = field(default_factory=list)
    # Actions that would reduce the hazard

    def __repr__(self):
        return (
            f"SafetyRule[{self.name}] "
            f"({self.category.value}, {self.severity.name}): "
            f"{self.rule}"
        )


# Part 2: Safety Knowledge Base

class SafetyKnowledgeBase:
    """
    Complete safety knowledge base for embodied intelligence.
    
    Manages:
    - SafetyRuleTemplates (high-level rules with metadata)
    - The underlying KnowledgeBase (for the logic layer)
    - Rule-to-template mapping (for traceability)
    - Action-violation rule generation
    
    The KB is organized into hazard detection rules and aggregation rules.
    Hazard rules are the "interesting" ones — they tell us WHAT specific
    safety condition exists. Aggregation rules just roll them up.
    """

    def __init__(self):
        self.templates: Dict[str, SafetyRuleTemplate] = {}
        self.kb = KnowledgeBase()
        self._rule_name_to_template: Dict[str, str] = {}

    def add_template(self, template: SafetyRuleTemplate):
        """Register a safety rule template."""
        self.templates[template.name] = template
        self.kb.add_rule(template.rule)
        rule_name = template.rule.name or template.name
        self._rule_name_to_template[rule_name] = template.name

    def get_template(self, name: str) -> Optional[SafetyRuleTemplate]:
        return self.templates.get(name)

    def get_template_for_rule(self, rule_name: str) -> Optional[SafetyRuleTemplate]:
        template_name = self._rule_name_to_template.get(rule_name)
        if template_name:
            return self.templates.get(template_name)
        return None

    def get_templates_by_category(
        self, category: SafetyCategory
    ) -> List[SafetyRuleTemplate]:
        return [
            t for t in self.templates.values()
            if t.category == category
        ]

    @property
    def hazard_predicates(self) -> Set[str]:
        """All hazard predicates defined by templates."""
        return {t.rule.head.predicate for t in self.templates.values()}

    @property
    def underlying_kb(self) -> KnowledgeBase:
        """The raw KnowledgeBase for the logic layer."""
        return self.kb

    def summary(self) -> str:
        lines = ["Safety Knowledge Base", "=" * 50]
        by_cat = {}
        for t in self.templates.values():
            by_cat.setdefault(t.category.value, []).append(t)
        for cat, templates in sorted(by_cat.items()):
            lines.append(f"\n  [{cat.upper()}]")
            for t in templates:
                sev = t.severity.name
                lines.append(f"    [{sev:8s}] {t.name}: {t.natural_language}")
                lines.append(f"             Rule: {t.rule}")
                if t.prohibited_actions:
                    lines.append(
                        f"             Prohibited: {t.prohibited_actions}"
                    )
        return "\n".join(lines)


# Part 3: Pre-built Safety Knowledge Bases

def create_embodied_safety_kb() -> SafetyKnowledgeBase:
    """
    Complete safety knowledge base for embodied AI in home/kitchen environments.
    
    Covers the hazard types from the architecture diagram:
        - Fire Risk
        - Electric Shock Risk
        - Cut / Sharp Object Risk
        - Burn Risk
        - Chemical / Poison Risk
        - Fall / Crush Risk
        - Child Safety
        - Slip Risk
        - Collision Risk
    
    Each rule is a SafetyRuleTemplate carrying:
        - The Prolog-style logic rule
        - Category and severity metadata
        - Natural language description
        - Prohibited/mitigating actions for downstream action checking
    """
    skb = SafetyKnowledgeBase()

    # FIRE HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="fire_near_flammable",
        category=SafetyCategory.FIRE,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("fire_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("flammable", ["X"]),
                Literal("is_fire", ["Y"]),
            ],
            name="fire_near_flammable",
        ),
        natural_language="Flammable object X is near fire source Y.",
        prohibited_actions=["move_towards", "place_near", "push_towards"],
        mitigating_actions=["move_away", "remove", "extinguish"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="fire_cloth_near_flame",
        category=SafetyCategory.FIRE,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("fire_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_cloth", ["X"]),
                Literal("is_fire", ["Y"]),
            ],
            name="fire_cloth_near_flame",
        ),
        natural_language="Cloth/fabric X is near fire source Y.",
        prohibited_actions=["drape_over", "place_near"],
        mitigating_actions=["move_away", "remove"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="fire_oil_spill_near_flame",
        category=SafetyCategory.FIRE,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("fire_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_oil", ["X"]),
                Literal("is_fire", ["Y"]),
            ],
            name="fire_oil_spill_near_flame",
        ),
        natural_language="Oil/grease X is near fire source Y.",
        prohibited_actions=["pour", "splash", "move_towards"],
        mitigating_actions=["cover", "remove", "smother"],
    ))

    # ELECTRICAL HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="electric_water_near_power",
        category=SafetyCategory.ELECTRICAL,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("electrical_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("liquid", ["X"]),
                Literal("is_electrical", ["Y"]),
            ],
            name="electric_water_near_power",
        ),
        natural_language="Liquid X is near electrical device Y.",
        prohibited_actions=["pour", "splash", "place_near", "move_towards"],
        mitigating_actions=["move_away", "dry", "unplug", "cover"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="electric_wet_hands_touch",
        category=SafetyCategory.ELECTRICAL,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("electrical_hazard", ["X", "Y"]),
            body=[
                Literal("touching", ["X", "Y"]),
                Literal("is_wet", ["X"]),
                Literal("is_electrical", ["Y"]),
            ],
            name="electric_wet_hands_touch",
        ),
        natural_language="Wet entity X is touching electrical device Y.",
        prohibited_actions=["touch", "grab", "operate"],
        mitigating_actions=["dry_hands", "unplug", "step_back"],
    ))

    # BURN HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="burn_touch_hot_surface",
        category=SafetyCategory.BURN,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("burn_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_person", ["X"]),
                Literal("is_hot", ["Y"]),
            ],
            name="burn_touch_hot_surface",
        ),
        natural_language="Person X is near hot surface Y.",
        prohibited_actions=["touch", "grab", "lean_on"],
        mitigating_actions=["use_protection", "wait_cool", "move_away"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="burn_hot_liquid_near_person",
        category=SafetyCategory.BURN,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("burn_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_person", ["X"]),
                Literal("is_hot_liquid", ["Y"]),
            ],
            name="burn_hot_liquid_near_person",
        ),
        natural_language="Person X is near hot liquid Y.",
        prohibited_actions=["reach_over", "bump", "move_fast_near"],
        mitigating_actions=["move_away", "handle_carefully"],
    ))

    # CUT / SHARP OBJECT HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="cut_child_near_sharp",
        category=SafetyCategory.CUT,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("cut_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_child", ["X"]),
                Literal("sharp", ["Y"]),
            ],
            name="cut_child_near_sharp",
        ),
        natural_language="Child X is near sharp object Y.",
        prohibited_actions=[
            "hand_to_child", "place_within_reach", "leave_unattended"
        ],
        mitigating_actions=["move_away", "store_safely", "cover_blade"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="cut_running_with_sharp",
        category=SafetyCategory.CUT,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("cut_hazard", ["X", "Y"]),
            body=[
                Literal("holding", ["X", "Y"]),
                Literal("sharp", ["Y"]),
                Literal("running", ["X"]),
            ],
            name="cut_running_with_sharp",
        ),
        natural_language="Person X is running while holding sharp object Y.",
        prohibited_actions=["run", "move_fast"],
        mitigating_actions=["slow_down", "put_down", "sheath"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="cut_sharp_edge_exposed",
        category=SafetyCategory.CUT,
        severity=Severity.MEDIUM,
        rule=Rule(
            head=Literal("cut_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("sharp", ["Y"]),
                Literal("is_exposed", ["Y"]),
            ],
            name="cut_sharp_edge_exposed",
        ),
        natural_language="Exposed sharp object Y is near entity X.",
        prohibited_actions=["reach_near", "grab_blindly"],
        mitigating_actions=["cover_blade", "store_safely"],
    ))

    # CHEMICAL / POISON HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="chemical_toxic_not_contained",
        category=SafetyCategory.CHEMICAL,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("chemical_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_person", ["X"]),
                Literal("toxic", ["Y"]),
                Literal("is_container", ["Y"], negated=True),
            ],
            name="chemical_toxic_not_contained",
        ),
        natural_language="Person X is near uncontained toxic substance Y.",
        prohibited_actions=["touch", "inhale_near", "open"],
        mitigating_actions=["contain", "ventilate", "evacuate", "move_away"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="poison_child_near_medication",
        category=SafetyCategory.POISON,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("poison_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_child", ["X"]),
                Literal("is_medication", ["Y"]),
                Literal("is_locked", ["Y"], negated=True),
            ],
            name="poison_child_near_medication",
        ),
        natural_language="Child X is near unsecured medication Y.",
        prohibited_actions=["leave_unattended", "open", "place_within_reach"],
        mitigating_actions=["lock_away", "move_out_of_reach", "store_high"],
    ))

    # FALL / CRUSH HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="fall_heavy_above",
        category=SafetyCategory.FALL,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("fall_hazard", ["X", "Y"]),
            body=[
                Literal("above", ["X", "Y"]),
                Literal("heavy", ["X"]),
                Literal("is_person", ["Y"]),
            ],
            name="fall_heavy_above",
        ),
        natural_language="Heavy object X is above person Y.",
        prohibited_actions=["push", "shake", "pull_from_below"],
        mitigating_actions=["secure", "move_person", "lower_carefully"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="fall_unstable_stack",
        category=SafetyCategory.FALL,
        severity=Severity.MEDIUM,
        rule=Rule(
            head=Literal("fall_hazard", ["X", "Y"]),
            body=[
                Literal("above", ["X", "Y"]),
                Literal("heavy", ["X"]),
                Literal("is_unstable", ["Y"]),
            ],
            name="fall_unstable_stack",
        ),
        natural_language="Heavy object X is on unstable support Y.",
        prohibited_actions=["stack_more", "bump", "pull"],
        mitigating_actions=["redistribute", "secure", "move_to_stable"],
    ))

    # SLIP HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="slip_liquid_on_floor",
        category=SafetyCategory.SLIP,
        severity=Severity.MEDIUM,
        rule=Rule(
            head=Literal("slip_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_person", ["X"]),
                Literal("liquid", ["Y"]),
                Literal("on_floor", ["Y"]),
            ],
            name="slip_liquid_on_floor",
        ),
        natural_language="Person X is near liquid Y spilled on floor.",
        prohibited_actions=["walk_through", "run_near"],
        mitigating_actions=["clean_up", "mark_wet", "walk_around"],
    ))

    # COLLISION HAZARDS

    skb.add_template(SafetyRuleTemplate(
        name="collision_running_near_person",
        category=SafetyCategory.COLLISION,
        severity=Severity.MEDIUM,
        rule=Rule(
            head=Literal("collision_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("running", ["X"]),
                Literal("is_person", ["Y"]),
            ],
            name="collision_running_near_person",
        ),
        natural_language="Running entity X is near person Y.",
        prohibited_actions=["run", "move_fast"],
        mitigating_actions=["slow_down", "change_path", "alert"],
    ))

    # CHILD-SPECIFIC SAFETY

    skb.add_template(SafetyRuleTemplate(
        name="child_near_hot",
        category=SafetyCategory.CHILD_SAFETY,
        severity=Severity.CRITICAL,
        rule=Rule(
            head=Literal("child_safety_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_child", ["X"]),
                Literal("is_hot", ["Y"]),
            ],
            name="child_near_hot",
        ),
        natural_language="Child X is near hot object Y. "
                         "Keep hot objects away from children.",
        prohibited_actions=[
            "leave_unattended", "hand_to_child", "place_within_reach"
        ],
        mitigating_actions=["move_child", "barrier", "cool_down"],
    ))

    skb.add_template(SafetyRuleTemplate(
        name="child_near_electrical",
        category=SafetyCategory.CHILD_SAFETY,
        severity=Severity.HIGH,
        rule=Rule(
            head=Literal("child_safety_hazard", ["X", "Y"]),
            body=[
                Literal("near", ["X", "Y"]),
                Literal("is_child", ["X"]),
                Literal("is_electrical", ["Y"]),
            ],
            name="child_near_electrical",
        ),
        natural_language="Child X is near electrical device Y.",
        prohibited_actions=["leave_unattended", "let_touch"],
        mitigating_actions=["childproof", "unplug", "move_child"],
    ))

    return skb


# Part 4: Safety Rule Parser (from text)

class SafetyRuleParser:
    """
    Parses safety rules from Prolog-style text format.
    
    Supports:
        % [FIRE, CRITICAL] Keep flammable objects away from fire
        fire_hazard(X, Y) :- near(X, Y), flammable(X), is_fire(Y).
        prohibited: move_towards, place_near.
        mitigating: move_away, remove.
    """

    _RULE_PATTERN = re.compile(
        r"(\w+)\(([^)]+)\)\s*:-\s*(.+)\."
    )
    _META_PATTERN = re.compile(
        r"%\s*\[(\w+),\s*(\w+)\]\s*(.*)"
    )

    def parse(self, text: str) -> SafetyKnowledgeBase:
        """Parse multi-line safety rule definitions."""
        skb = SafetyKnowledgeBase()
        lines = text.strip().split("\n")

        current_meta = {
            "category": SafetyCategory.FIRE,
            "severity": Severity.MEDIUM,
            "description": "",
        }
        current_prohibited = []
        current_mitigating = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Metadata comment
            meta_match = self._META_PATTERN.match(line)
            if meta_match:
                cat_str = meta_match.group(1).upper()
                sev_str = meta_match.group(2).upper()
                desc = meta_match.group(3).strip()
                try:
                    current_meta["category"] = SafetyCategory(cat_str.lower())
                except ValueError:
                    current_meta["category"] = SafetyCategory.FIRE
                try:
                    current_meta["severity"] = Severity[sev_str]
                except KeyError:
                    current_meta["severity"] = Severity.MEDIUM
                current_meta["description"] = desc
                continue

            # Prohibited actions
            if line.lower().startswith("prohibited:"):
                actions = line.split(":", 1)[1].strip().rstrip(".")
                current_prohibited = [a.strip() for a in actions.split(",")]
                continue

            # Mitigating actions
            if line.lower().startswith("mitigating:"):
                actions = line.split(":", 1)[1].strip().rstrip(".")
                current_mitigating = [a.strip() for a in actions.split(",")]
                continue

            # Skip other comments
            if line.startswith("%") or line.startswith("//"):
                continue

            # Rule
            rule_match = self._RULE_PATTERN.match(line)
            if rule_match:
                head_pred = rule_match.group(1)
                head_args = [
                    a.strip() for a in rule_match.group(2).split(",")
                ]
                body_str = rule_match.group(3)
                body_lits = self._parse_body(body_str)

                rule = Rule(
                    head=Literal(head_pred, head_args),
                    body=body_lits,
                    name=head_pred + "_" + "_".join(
                        l.predicate for l in body_lits
                    ),
                )

                template = SafetyRuleTemplate(
                    name=rule.name,
                    category=current_meta["category"],
                    severity=current_meta["severity"],
                    rule=rule,
                    natural_language=current_meta.get("description", ""),
                    prohibited_actions=list(current_prohibited),
                    mitigating_actions=list(current_mitigating),
                )
                skb.add_template(template)

                # Reset per-rule metadata
                current_prohibited = []
                current_mitigating = []

        return skb

    def _parse_body(self, body_str: str) -> List[Literal]:
        """Parse comma-separated body literals."""
        lits = []
        # Split by comma but respect parentheses
        depth = 0
        current = ""
        for ch in body_str:
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                lits.append(self._parse_literal(current.strip()))
                current = ""
            else:
                current += ch
        if current.strip():
            lits.append(self._parse_literal(current.strip()))
        return lits

    def _parse_literal(self, lit_str: str) -> Literal:
        """Parse a single literal like 'near(X, Y)' or '\\+is_container(Y)'."""
        negated = False
        s = lit_str.strip()
        if s.startswith("\\+") or s.startswith("not "):
            negated = True
            s = s.lstrip("\\+").lstrip("not ").strip()

        match = re.match(r"(\w+)\(([^)]+)\)", s)
        if match:
            pred = match.group(1)
            args = [a.strip() for a in match.group(2).split(",")]
            return Literal(pred, args, negated)

        # Fallback: treat as nullary predicate
        return Literal(s, [], negated)
