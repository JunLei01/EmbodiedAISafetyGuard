"""
Differentiable Safety Logic Reasoner
======================================
Core module of the embodied safety guardrail system.

This is the 可微逻辑推理器 referenced in the architecture:

    Scene Graph (neural predicates)    High-Level Safety Rules
            ↓                                   ↓
    ┌───────────────────────────────────────────────────────┐
    │           DifferentiableSafetyReasoner                │
    │                                                       │
    │   ┌─────────────────────────────────────────────┐     │
    │   │  1. Forward Chaining Engine                 │     │
    │   │     Semiring: AND=product, OR=noisy-OR      │     │
    │   │     Topological order over rule dependency   │     │
    │   └──────────────┬──────────────────────────────┘     │
    │                  ↓                                    │
    │   ┌─────────────────────────────────────────────┐     │
    │   │  2. Safety Rule Activation                  │     │
    │   │     Ground rules against scene entities     │     │
    │   │     Probability threshold filtering          │     │
    │   └──────────────┬──────────────────────────────┘     │
    │                  ↓                                    │
    │   ┌─────────────────────────────────────────────┐     │
    │   │  3. Proof Trace & Explanation               │     │
    │   │     Why is this rule activated?              │     │
    │   │     Which entities? Which conditions?        │     │
    │   └──────────────┬──────────────────────────────┘     │
    │                  ↓                                    │
    │   ┌─────────────────────────────────────────────┐     │
    │   │  4. Action Violation Interface              │     │
    │   │     Prepare output for action checking      │     │
    │   └─────────────────────────────────────────────┘     │
    └───────────────────────────────────────────────────────┘
            ↓
    Activated Scene Safety Rules
    [
        ActivatedSafetyRule(
            rule_name = "fire_near_flammable",
            hazard    = "fire_hazard(gasoline_can, stove)",
            P         = 0.87,
            entities  = {X: "gasoline_can", Y: "stove"},
            evidence  = {near(gc,stv)=0.92, flammable(gc)=0.95, ...},
            prohibited_actions = ["move_towards", "place_near"],
        ),
        ...
    ]
            ↓
    Action Violation Checker (downstream)

Key properties:
    1. DIFFERENTIABLE: All probabilities are torch tensors with grad
    2. EXPLAINABLE:    Every activated rule carries its full proof trace
    3. COMPOSABLE:     Output format directly feeds action-violation checking
    4. SCALABLE:       Pruning strategies avoid combinatorial explosion
"""

import os

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from enum import Enum
import time

from utils.knowledge_base import KnowledgeBase, Rule, Literal
from utils.safety_knowledge_base import (
    SafetyKnowledgeBase,
    SafetyRuleTemplate,
    SafetyCategory,
    Severity,
    create_embodied_safety_kb,
)
from utils.llm_safety_adapter import LLMSafetyAdapter, MiniMaxLLMClient


# Part 1: Activated Safety Rule (Output Data Structure)

@dataclass
class EvidenceItem:
    """A single piece of evidence supporting a safety rule activation."""
    predicate_key: str          # e.g., "near(gasoline_can,stove)"
    probability: float          # 0.92
    source: str = "neural"      # "neural" | "inferred" | "derived"

    def __repr__(self):
        return f"{self.predicate_key}={self.probability:.3f}"


@dataclass
class ProofNode:
    """
    A node in the proof tree.
    
    For the rule:
        fire_hazard(X, Y) :- near(X, Y), flammable(X), is_fire(Y).
    with X=gasoline_can, Y=stove, the proof tree is:
    
        fire_hazard(gasoline_can, stove) = 0.87
        ├── near(gasoline_can, stove) = 0.92  [spatial inference]
        ├── flammable(gasoline_can) = 0.95     [semantic type]
        └── is_fire(stove) = 1.00              [semantic type]
    """
    predicate_key: str
    probability: float
    children: List["ProofNode"] = field(default_factory=list)
    rule_name: str = ""
    is_leaf: bool = False

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON export."""
        d = {
            "predicate": self.predicate_key,
            "probability": round(self.probability, 4),
        }
        if self.rule_name:
            d["rule"] = self.rule_name
        if self.children:
            d["evidence"] = [c.to_dict() for c in self.children]
        return d

    def to_lines(self, indent: int = 0) -> List[str]:
        """Pretty-print the proof tree."""
        prefix = "  " * indent
        connector = "├── " if indent > 0 else ""
        lines = [
            f"{prefix}{connector}{self.predicate_key} = {self.probability:.4f}"
            + (f"  [via {self.rule_name}]" if self.rule_name else "")
        ]
        for child in self.children:
            lines.extend(child.to_lines(indent + 1))
        return lines


@dataclass
class ActivatedSafetyRule:
    """
    A safety rule that has been activated (grounded and evaluated)
    in the current scene.
    
    This is the PRIMARY OUTPUT of the differentiable reasoner.
    It carries everything needed for downstream action-violation checking.
    """
    # ── Identity ──
    rule_name: str                # "fire_near_flammable"
    template_name: str            # Reference to SafetyRuleTemplate
    category: SafetyCategory
    severity: Severity

    # ── Grounding ──
    hazard_predicate: str         # "fire_hazard(gasoline_can,stove)"
    probability: torch.Tensor     # tensor(0.87) — DIFFERENTIABLE
    entity_bindings: Dict[str, str]  # {"X": "gasoline_can", "Y": "stove"}

    # ── Evidence / Explainability ──
    evidence: List[EvidenceItem] = field(default_factory=list)
    proof_tree: Optional[ProofNode] = None

    # ── Action Interface ──
    prohibited_actions: List[str] = field(default_factory=list)
    involved_entities: Set[str] = field(default_factory=set)
    mitigating_actions: List[str] = field(default_factory=list)

    # ── Natural Language ──
    natural_language: str = ""
    grounded_description: str = ""  # With entity names filled in

    @property
    def prob_value(self) -> float:
        """Get probability as float."""
        if isinstance(self.probability, torch.Tensor):
            return self.probability.item()
        return float(self.probability)

    def __repr__(self):
        return (
            f"ActivatedRule[{self.rule_name}] "
            f"P={self.prob_value:.4f} "
            f"({self.category.value}, {self.severity.name}) "
            f"entities={dict(self.entity_bindings)}"
        )

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "rule_name": self.rule_name,
            "category": self.category.value,
            "severity": self.severity.name,
            "hazard": self.hazard_predicate,
            "probability": round(self.prob_value, 4),
            "entities": dict(self.entity_bindings),
            "evidence": [
                {"predicate": e.predicate_key, "prob": round(e.probability, 4)}
                for e in self.evidence
            ],
            "prohibited_actions": self.prohibited_actions,
            "mitigating_actions": self.mitigating_actions,
            "description": self.grounded_description or self.natural_language,
            "proof": self.proof_tree.to_dict() if self.proof_tree else None,
        }


@dataclass
class SafetyReasoningResult:
    """
    Complete output of the differentiable safety reasoner.
    Contains all activated safety rules for the current scene.
    """
    activated_rules: List[ActivatedSafetyRule]
    all_derived_probs: Dict[str, torch.Tensor]
    all_grounded_probs: Dict[str, torch.Tensor]
    entity_ids: List[str]
    inference_time_ms: float = 0.0

    @property
    def num_activated(self) -> int:
        return len(self.activated_rules)

    @property
    def max_severity(self) -> Optional[Severity]:
        if not self.activated_rules:
            return None
        return max(
            (r.severity for r in self.activated_rules),
            key=lambda s: s.value,
        )

    @property
    def is_safe(self) -> bool:
        """Scene is safe if no HIGH or CRITICAL rules are activated."""
        return all(
            r.severity.value < Severity.HIGH.value
            for r in self.activated_rules
        )

    def get_rules_by_category(
        self, category: SafetyCategory
    ) -> List[ActivatedSafetyRule]:
        return [r for r in self.activated_rules if r.category == category]

    def get_rules_by_severity(
        self, min_severity: Severity
    ) -> List[ActivatedSafetyRule]:
        return [
            r for r in self.activated_rules
            if r.severity.value >= min_severity.value
        ]

    def get_rules_involving_entity(
        self, entity_id: str
    ) -> List[ActivatedSafetyRule]:
        return [
            r for r in self.activated_rules
            if entity_id in r.involved_entities
        ]

    def get_prohibited_actions_for_entity(
        self, entity_id: str
    ) -> Dict[str, List[str]]:
        """
        Get all prohibited actions for a specific entity,
        grouped by the safety rule that prohibits them.
        
        This is the key interface for downstream action checking:
            actions = result.get_prohibited_actions_for_entity("robot_arm")
            if proposed_action in actions: BLOCK
        """
        actions = {}
        for rule in self.get_rules_involving_entity(entity_id):
            actions[rule.rule_name] = rule.prohibited_actions
        return actions

    def summary(self) -> str:
        lines = [
            f"Safety Reasoning Result",
            f"=" * 60,
            f"  Entities:       {self.entity_ids}",
            f"  Activated rules: {self.num_activated}",
            f"  Max severity:   {self.max_severity.name if self.max_severity else 'NONE'}",
            f"  Scene safe:     {'YES' if self.is_safe else 'NO'}",
            f"  Inference time: {self.inference_time_ms:.1f} ms",
        ]
        if self.activated_rules:
            lines.append(f"\n  Activated Safety Rules:")
            lines.append(f"  {'─' * 54}")
            for rule in sorted(
                self.activated_rules,
                key=lambda r: (-r.severity.value, -r.prob_value),
            ):
                sev_icon = {
                    Severity.CRITICAL: "[CRIT]",
                    Severity.HIGH:     "[HIGH]",
                    Severity.MEDIUM:   "[MED ]",
                    Severity.LOW:      "[LOW ]",
                }[rule.severity]
                lines.append(
                    f"  {sev_icon} [{rule.severity.name:8s}] "
                    f"{rule.rule_name}"
                )
                lines.append(
                    f"     P={rule.prob_value:.4f}  "
                    f"entities={dict(rule.entity_bindings)}"
                )
                lines.append(
                    f"     {rule.grounded_description or rule.natural_language}"
                )
                if rule.prohibited_actions:
                    lines.append(
                        f"    Prohibited: {rule.prohibited_actions}"
                    )
        return "\n".join(lines)


# Part 2: Forward Chaining Inference Engine

class ForwardChainingEngine(nn.Module):
    """
    DeepProbLog-style forward chaining inference engine.
    
    Semiring-based computation:
        ⊗ (times) = product        → conjunction (AND)
        ⊕ (plus)  = noisy-OR       → disjunction (OR)
    
    Process:
        1. Start with grounded neural predicate probabilities
        2. Process rules in topological (dependency) order
        3. For each rule, enumerate all valid groundings
        4. Compute conjunction probability for each grounding
        5. Aggregate via noisy-OR across groundings with same head
    
    Optimizations:
        - Pruning: skip groundings where any body literal P < threshold
        - Caching: memo grounded facts to avoid recomputation
        - Relevance filtering: only ground rules involving scene entities
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        learn_rule_weights: bool = True,
        pruning_threshold: float = 0.01,
    ):
        super().__init__()
        self.kb = kb
        self.pruning_threshold = pruning_threshold

        # Learnable rule weights: w_i ∈ (0, 1) via sigmoid
        # Controls how much each rule contributes to the final probability
        self.learn_weights = learn_rule_weights
        if learn_rule_weights:
            self.rule_weight_logits = nn.ParameterDict()
            for i, rule in enumerate(self.kb.rules):
                key = self._safe_key(rule.name or f"rule_{i}")
                # Init at 2.0 → sigmoid ≈ 0.88, so rules are mostly "on"
                self.rule_weight_logits[key] = nn.Parameter(
                    torch.tensor(2.0)
                )

    @staticmethod
    def _safe_key(name: str) -> str:
        """Make a name safe for nn.ParameterDict keys."""
        return name.replace(".", "_").replace("-", "_")

    def forward(
        self,
        grounded_facts: Dict[str, torch.Tensor],
        entities: List[str],
        query_predicates: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward chaining inference.
        
        Args:
            grounded_facts: Neural predicate probabilities
                {"near(1,2)": tensor(0.92), "flammable(1)": tensor(0.95), ...}
            entities: Entity IDs in the scene ["1", "2", "3", ...]
            query_predicates: If set, only compute these predicates
            
        Returns:
            All derived fact probabilities including intermediate predicates
        """
        # Working copy of all known facts
        fact_probs: Dict[str, torch.Tensor] = dict(grounded_facts)
        entity_set = set(entities)

        # Get topological processing order
        pred_order = self.kb.dependency_order()
        target = query_predicates or self.kb.defined_predicates

        for pred_name in pred_order:
            # Skip base predicates (already in grounded_facts)
            if pred_name in self.kb.base_predicates:
                continue

            rules = self.kb.get_rules_for(pred_name)
            if not rules:
                continue

            # Determine head arity from first rule
            head_arity = len(rules[0].head.arguments)
            head_vars = rules[0].head.arguments

            # Enumerate all possible head groundings
            for entity_combo in cartesian_product(entities, repeat=head_arity):
                head_binding = dict(zip(head_vars, entity_combo))
                ground_head = rules[0].head.substitute(head_binding)
                head_key = ground_head.ground_key()

                # Collect rule contributions (noisy-OR)
                rule_contributions = []

                for rule_idx, rule in enumerate(rules):
                    rule_prob = self._evaluate_rule_grounding(
                        rule, head_binding, fact_probs, entities, rule_idx
                    )
                    if rule_prob is not None:
                        rule_contributions.append(rule_prob)

                if rule_contributions:
                    fact_probs[head_key] = self._noisy_or(rule_contributions)

        return fact_probs

    def _evaluate_rule_grounding(
        self,
        rule: Rule,
        head_binding: Dict[str, str],
        fact_probs: Dict[str, torch.Tensor],
        entities: List[str],
        rule_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Evaluate a rule with given head variable binding.
        
        Handles existential body variables by enumerating over entities
        and aggregating via noisy-OR.
        """
        head_vars = set(head_binding.keys())
        body_vars = rule.get_variables() - head_vars

        if not body_vars:
            # Fully ground → directly evaluate conjunction
            result = self._evaluate_conjunction(rule.body, head_binding, fact_probs)
        else:
            # Existential variables → enumerate and noisy-OR
            body_var_list = sorted(body_vars)
            grounding_probs = []

            for var_combo in cartesian_product(
                entities, repeat=len(body_var_list)
            ):
                full_binding = dict(head_binding)
                full_binding.update(dict(zip(body_var_list, var_combo)))

                conj_prob = self._evaluate_conjunction(
                    rule.body, full_binding, fact_probs
                )
                if conj_prob is not None:
                    grounding_probs.append(conj_prob)

            if not grounding_probs:
                return None
            result = self._noisy_or(grounding_probs)

        # Guard: conjunction may return None if facts are missing
        if result is None:
            return None

        # Apply learnable rule weight
        if self.learn_weights:
            weight_key = self._safe_key(rule.name or f"rule_{rule_idx}")
            if weight_key in self.rule_weight_logits:
                w = torch.sigmoid(self.rule_weight_logits[weight_key])
                result = result * w

        return result

    def _evaluate_conjunction(
        self,
        body: List[Literal],
        bindings: Dict[str, str],
        fact_probs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Evaluate conjunction (AND) of ground body literals.
        
        P(body) = ∏ P(lit_i)        for positive literals
                * ∏ (1 - P(lit_i))   for negated literals
        
        Returns None if any required fact is missing (open-world assumption).
        Uses pruning: early exit if running product drops below threshold.
        """
        prob = None

        for literal in body:
            ground_lit = literal.substitute(bindings)
            key = ground_lit.ground_key()

            if key not in fact_probs:
                return None  # Missing fact → cannot satisfy this grounding

            lit_prob = fact_probs[key]

            if literal.negated:
                lit_prob = 1.0 - lit_prob

            if prob is None:
                prob = lit_prob
            else:
                prob = prob * lit_prob

            # Pruning: if probability already very low, skip remaining
            if isinstance(prob, torch.Tensor):
                if prob.item() < self.pruning_threshold:
                    return None
            elif prob < self.pruning_threshold:
                return None

        return prob

    @staticmethod
    def _noisy_or(probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Noisy-OR (differentiable disjunction):
            P(A ∨ B ∨ ...) = 1 - ∏(1 - P_i)
            
        Computed in log-space for numerical stability:
            log(1-P) = Σ log(1 - P_i)
            P = 1 - exp(Σ log(1 - P_i))
        """
        if len(probs) == 1:
            return probs[0]

        log_complement = sum(
            torch.log1p(-p.clamp(0.0, 1.0 - 1e-7))
            for p in probs
        )
        return 1.0 - torch.exp(log_complement)


# Part 3: Safety Rule Activator

class SafetyRuleActivator:
    """
    Extracts activated safety rules from the inference results.
    
    Given derived probabilities from the forward chaining engine,
    identifies which safety rules are activated (above threshold)
    and builds the full ActivatedSafetyRule objects with proof traces.
    """

    def __init__(
        self,
        safety_kb: SafetyKnowledgeBase,
        activation_threshold: float = 0.3,
    ):
        self.safety_kb = safety_kb
        self.activation_threshold = activation_threshold

    def extract_activated_rules(
        self,
        all_fact_probs: Dict[str, torch.Tensor],
        entity_ids: List[str],
    ) -> List[ActivatedSafetyRule]:
        """
        Scan inference results for activated safety rules.
        
        For each safety rule template, check all groundings.
        If P(grounding) > threshold, create an ActivatedSafetyRule.
        """
        activated = []

        for template_name, template in self.safety_kb.templates.items():
            rule = template.rule
            head_pred = rule.head.predicate
            head_args = rule.head.arguments

            # Find all groundings of this hazard predicate
            for entity_combo in cartesian_product(
                entity_ids, repeat=len(head_args)
            ):
                bindings = dict(zip(head_args, entity_combo))
                ground_head = rule.head.substitute(bindings)
                head_key = ground_head.ground_key()

                if head_key not in all_fact_probs:
                    continue

                prob = all_fact_probs[head_key]
                prob_val = prob.item() if isinstance(prob, torch.Tensor) else prob

                if prob_val < self.activation_threshold:
                    continue

                # Build evidence list
                evidence = self._collect_evidence(
                    rule, bindings, all_fact_probs, entity_ids
                )

                # Build proof tree
                proof = self._build_proof_tree(
                    head_key, prob_val, rule, bindings,
                    all_fact_probs, entity_ids
                )

                # Generate grounded natural language description
                grounded_desc = self._ground_description(
                    template.natural_language, bindings
                )

                involved = set(entity_combo)

                activated.append(ActivatedSafetyRule(
                    rule_name=template.name,
                    template_name=template_name,
                    category=template.category,
                    severity=template.severity,
                    hazard_predicate=head_key,
                    probability=prob,
                    entity_bindings=bindings,
                    evidence=evidence,
                    proof_tree=proof,
                    prohibited_actions=list(template.prohibited_actions),
                    involved_entities=involved,
                    mitigating_actions=list(template.mitigating_actions),
                    natural_language=template.natural_language,
                    grounded_description=grounded_desc,
                ))

        # Sort by severity (descending) then probability (descending)
        activated.sort(
            key=lambda r: (-r.severity.value, -r.prob_value)
        )

        return activated

    def _collect_evidence(
        self,
        rule: Rule,
        head_binding: Dict[str, str],
        all_facts: Dict[str, torch.Tensor],
        entities: List[str],
    ) -> List[EvidenceItem]:
        """Collect all evidence supporting a rule activation."""
        evidence = []

        # Find the best body grounding (highest conjunction probability)
        body_vars = rule.get_variables() - set(head_binding.keys())

        if not body_vars:
            # Fully ground
            for lit in rule.body:
                ground = lit.substitute(head_binding)
                key = ground.ground_key()
                if key in all_facts:
                    prob_val = all_facts[key]
                    p = prob_val.item() if isinstance(prob_val, torch.Tensor) else prob_val
                    source = "derived" if any(
                        key.startswith(hp + "(")
                        for hp in self.safety_kb.hazard_predicates
                    ) else "neural"
                    evidence.append(EvidenceItem(key, p, source))
        else:
            # Find best grounding for existential variables
            body_var_list = sorted(body_vars)
            best_prob = -1.0
            best_evidence = []

            for var_combo in cartesian_product(
                entities, repeat=len(body_var_list)
            ):
                full_binding = dict(head_binding)
                full_binding.update(dict(zip(body_var_list, var_combo)))

                current_evidence = []
                conj_prob = 1.0
                valid = True

                for lit in rule.body:
                    ground = lit.substitute(full_binding)
                    key = ground.ground_key()
                    if key not in all_facts:
                        valid = False
                        break
                    pv = all_facts[key]
                    p = pv.item() if isinstance(pv, torch.Tensor) else pv
                    if lit.negated:
                        conj_prob *= (1.0 - p)
                    else:
                        conj_prob *= p
                    current_evidence.append(
                        EvidenceItem(key, p, "neural")
                    )

                if valid and conj_prob > best_prob:
                    best_prob = conj_prob
                    best_evidence = current_evidence

            evidence = best_evidence

        return evidence

    def _build_proof_tree(
        self,
        head_key: str,
        head_prob: float,
        rule: Rule,
        bindings: Dict[str, str],
        all_facts: Dict[str, torch.Tensor],
        entities: List[str],
    ) -> ProofNode:
        """Build a proof tree for explainability."""
        children = []
        for ev_item in self._collect_evidence(
            rule, bindings, all_facts, entities
        ):
            children.append(ProofNode(
                predicate_key=ev_item.predicate_key,
                probability=ev_item.probability,
                is_leaf=True,
            ))

        return ProofNode(
            predicate_key=head_key,
            probability=head_prob,
            children=children,
            rule_name=rule.name,
        )

    @staticmethod
    def _ground_description(
        template: str, bindings: Dict[str, str]
    ) -> str:
        """Replace variables in natural language with entity names."""
        result = template
        for var, entity in bindings.items():
            result = result.replace(var, entity)
        return result


# Part 4: Main Reasoner (Public API)


class DifferentiableSafetyReasoner(nn.Module):
    """
    Main entry point: the differentiable safety logic reasoner.
    
    Usage:
        # Setup
        reasoner = DifferentiableSafetyReasoner()
        
        # Option A: from raw predicates text
        result = reasoner.reason_from_predicates(predicate_text)
        
        # Option B: from pre-built grounded facts
        result = reasoner.reason(grounded_facts, entity_ids)
        
        # Inspect results
        print(result.summary())
        for rule in result.activated_rules:
            print(rule.proof_tree.to_lines())
        
        # Check if a proposed action violates any rule
        violations = reasoner.check_action(result, "move_towards", "gasoline_can", "stove")
    """

    def __init__(
        self,
        safety_kb: Optional[SafetyKnowledgeBase] = None,
        activation_threshold: float = 0.3,
        learn_rule_weights: bool = True,
        pruning_threshold: float = 0.01,

        enable_llm: bool = True,
        llm_api_key: Optional[str] = None,
        llm_threshold: float = 0.5,
        llm_min_rules: int = 3,  # 当规则数低于此值时触发 LLM
    ):
        super().__init__()

        # Safety knowledge base
        self.safety_kb = safety_kb or create_embodied_safety_kb()

        # Forward chaining engine
        self.engine = ForwardChainingEngine(
            kb=self.safety_kb.underlying_kb,
            learn_rule_weights=learn_rule_weights,
            pruning_threshold=pruning_threshold,
        )

        # Rule activator
        self.activator = SafetyRuleActivator(
            safety_kb=self.safety_kb,
            activation_threshold=activation_threshold,
        )

        # Store threshold for external access
        self.activation_threshold = activation_threshold

        # LLM Safety Adapter (可选)
        self.llm_adapter: Optional[LLMSafetyAdapter] = None
        self.llm_min_rules = llm_min_rules

        if enable_llm:
            try:
                # 检查是否有本地模型
                local_model_path = "/data/junlei/NPG/models/gpt-oss-20b"
                use_local = os.path.exists(local_model_path)

                if use_local:
                    print(f"  Using local model: {local_model_path}")
                    llm_client = MiniMaxLLMClient(
                        use_local=True,
                        local_model_path=local_model_path,
                    )
                else:
                    print("  Using LLM API")
                    llm_client = MiniMaxLLMClient(api_key=llm_api_key)

                self.llm_adapter = LLMSafetyAdapter(
                    llm_client=llm_client,
                    enable_llm=True,
                    llm_threshold=llm_threshold,
                )
                print("  LLM Enhancement mode enabled   ")
            except Exception as e:
                print(f"  Warning: Failed to initialize LLM adapter: {e}")
                print("  Will continue using traditional reasoning mode")

    def forward(
        self,
        grounded_facts: Dict[str, torch.Tensor],
        entity_ids: List[str],
        scene_name: Optional[str] = None,  
    ) -> "SafetyReasoningResult":
        """
        Core forward pass: grounded scene facts → activated safety rules.

        支持 LLM 增强：当传统规则激活不足时，使用 LLM 补充生成规则。

        Args:
            grounded_facts: Output of SceneGraphBuilder.build()
                {"near(1,2)": tensor(0.92), "flammable(1)": tensor(0.95), ...}
            entity_ids: ["1", "2", "3", ...]
            scene_name: 场景名称（用于 LLM 分析）

        Returns:
            SafetyReasoningResult with all activated rules (symbolic + LLM)
        """
        t0 = time.time()

        # Step 1: Forward chaining inference
        all_derived = self.engine(
            grounded_facts=grounded_facts,
            entities=entity_ids,
        )

        # Step 2: Extract activated safety rules (symbolic reasoning)
        activated = self.activator.extract_activated_rules(
            all_fact_probs=all_derived,
            entity_ids=entity_ids,
        )

        # Step 3: LLM 增强（如果启用且规则不足）
        if self.llm_adapter and len(activated) < self.llm_min_rules:
            llm_rules = self.llm_adapter.generate_rules(
                entity_ids=entity_ids,
                grounded_facts=grounded_facts,
                activated_rules=activated,
                scene_name=scene_name,
            )
            # 合并 LLM 生成的规则
            activated.extend(llm_rules)
            # 按严重程度和概率重新排序
            activated.sort(key=lambda r: (-r.severity.value, -r.prob_value))

        elapsed_ms = (time.time() - t0) * 1000

        return SafetyReasoningResult(
            activated_rules=activated,
            all_derived_probs=all_derived,
            all_grounded_probs=grounded_facts,
            entity_ids=entity_ids,
            inference_time_ms=elapsed_ms,
        )

    def reason(
        self,
        grounded_facts: Dict[str, torch.Tensor],
        entity_ids: List[str],
        scene_name: Optional[str] = None,
    ) -> SafetyReasoningResult:
        """Alias for forward() for clearer API."""
        return self.forward(grounded_facts, entity_ids, scene_name)

    def reason_from_predicates(
        self,
        predicate_text: str,
        scene_name: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
    ) -> SafetyReasoningResult:
        """
        Full pipeline: raw predicate text → activated safety rules.

        Convenience method that combines SceneGraphBuilder + reasoning.
        """
        from utils.scene_graph_builder import SceneGraphBuilder
        builder = SceneGraphBuilder()
        grounded_facts, entity_ids = builder.build(predicate_text, device)
        return self.forward(grounded_facts, entity_ids, scene_name)

    # ─── Action Violation Checking Interface ───

    def check_action(
        self,
        result: SafetyReasoningResult,
        action_name: str,
        actor_entity: str,
        target_entity: Optional[str] = None,
    ) -> List[ActivatedSafetyRule]:
        """
        Check if a proposed action violates any activated safety rule.
        
        This is the key interface for the downstream embodied AI module:
            - The agent proposes: move_towards(robot_arm, stove)
            - We check: does this violate any activated rule?
            - If yes: BLOCK the action and return the violations
        
        Args:
            result: Output of reason()
            action_name: Proposed action (e.g., "move_towards")
            actor_entity: Entity performing the action
            target_entity: Target entity (if applicable)
            
        Returns:
            List of violated safety rules (empty = action is safe)
        """
        violations = []

        for rule in result.activated_rules:
            # Check 1: action is in the prohibited list
            if action_name not in rule.prohibited_actions:
                continue

            # Check 2: the involved entities overlap with actor/target
            involved = rule.involved_entities
            entities_match = (
                actor_entity in involved
                or (target_entity is not None and target_entity in involved)
            )

            if entities_match:
                violations.append(rule)

        return violations

    def check_action_batch(
        self,
        result: SafetyReasoningResult,
        actions: List[Tuple[str, str, Optional[str]]],
    ) -> Dict[int, List[ActivatedSafetyRule]]:
        """
        Check multiple proposed actions at once.
        
        Args:
            actions: [(action_name, actor_entity, target_entity), ...]
            
        Returns:
            {action_index: [violated_rules]} for each violating action
        """
        violations = {}
        for idx, (action, actor, target) in enumerate(actions):
            v = self.check_action(result, action, actor, target)
            if v:
                violations[idx] = v
        return violations

    def suggest_safe_actions(
        self,
        result: SafetyReasoningResult,
        entity_id: str,
    ) -> Dict[str, List[str]]:
        """
        Suggest mitigating actions for each active hazard involving an entity.
        
        Returns:
            {"fire_near_flammable": ["move_away", "remove", "extinguish"], ...}
        """
        suggestions = {}
        for rule in result.get_rules_involving_entity(entity_id):
            if rule.mitigating_actions:
                suggestions[rule.rule_name] = rule.mitigating_actions
        return suggestions

    def get_safety_report(
        self,
        result: SafetyReasoningResult,
    ) -> Dict[str, Any]:
        """
        Generate a complete safety report for the scene.
        Structured for JSON serialization / API response.
        """
        return {
            "scene_safe": result.is_safe,
            "max_severity": (
                result.max_severity.name if result.max_severity else "NONE"
            ),
            "num_hazards": result.num_activated,
            "entities": result.entity_ids,
            "hazards": [r.to_dict() for r in result.activated_rules],
            "hazards_by_category": {
                cat.value: [
                    r.to_dict()
                    for r in result.get_rules_by_category(cat)
                ]
                for cat in SafetyCategory
                if result.get_rules_by_category(cat)
            },
            "inference_time_ms": round(result.inference_time_ms, 2),
        }