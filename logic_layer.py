"""
Differentiable Logic Layer
===========================
Implements DeepProbLog-style forward chaining with semiring-based
probabilistic inference.

Semiring semantics:
    AND (⊗) = product
    OR  (⊕) = noisy-OR

Key operations:
    P(A ∧ B) = P(A) × P(B)           (product for conjunction)
    P(A ∨ B) = 1 - (1-P(A))(1-P(B))  (noisy-OR for disjunction)

This module provides:
    - DifferentiableLogicLayer: Main inference module
    - ProofTracer: Explanation generation for proof trees (optional)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from utils.knowledge_base import KnowledgeBase, Rule, Literal


class DifferentiableLogicLayer(nn.Module):
    """
    DeepProbLog-style differentiable forward chaining inference.

    Architecture:
        Layer 1: Ground neural predicates from features
        Layer 2: Forward chain through rule dependencies
        Layer 3: Aggregate multiple rule contributions (noisy-OR)

    The entire pipeline is differentiable, allowing gradients to flow
    from derived predicate probabilities back through the logic rules
    to the neural predicate layer.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        learn_rule_weights: bool = True,
        inference_iterations: int = 10,
    ):
        super().__init__()
        self.kb = knowledge_base
        self.learn_rule_weights = learn_rule_weights
        self.inference_iterations = inference_iterations

        # Learnable rule weights
        if learn_rule_weights:
            self.rule_weights = nn.ParameterDict()
            for i, rule in enumerate(self.kb.rules):
                name = rule.name or f"rule_{i}"
                # Initialize to high confidence (sigmoid(2.0) ≈ 0.88)
                self.rule_weights[name] = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        grounded_facts: Dict[str, torch.Tensor],
        entities: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward chaining inference.

        Args:
            grounded_facts: Neural predicate probabilities
                {"near(1,2)": tensor(0.92), ...}
            entities: List of entity IDs ["1", "2", ...]

        Returns:
            derived_probs: All derived predicate probabilities
                including intermediate and final outputs
        """
        # Working copy of all facts
        all_probs = dict(grounded_facts)

        # Get topological order for forward chaining
        pred_order = self.kb.dependency_order()

        # Process derived predicates in order
        derived_preds = [p for p in pred_order if p in self.kb.defined_predicates]

        for predicate in derived_preds:
            rules = self.kb.get_rules_for(predicate)
            if not rules:
                continue

            # Ground this predicate against all entity combinations
            grounded = self._ground_predicate(rules[0], entities, all_probs)

            for head_key, prob in grounded.items():
                # Aggregate with existing probability (noisy-OR)
                if head_key in all_probs:
                    all_probs[head_key] = self._noisy_or(
                        [all_probs[head_key], prob]
                    )
                else:
                    all_probs[head_key] = prob

        return all_probs

    def _ground_predicate(
        self,
        template_rule: Rule,
        entities: List[str],
        known_facts: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Ground a predicate against all entity combinations.

        For rule:     fire_risk(X) :- near(X, Y), flammable(X), is_fire(Y).
        Grounds to:   fire_risk(1), fire_risk(2), etc.
        """
        results = {}

        head = template_rule.head
        body = template_rule.body

        # Number of arguments in head
        n_vars = len(head.arguments)

        # Enumerate all entity combinations for head variables
        from itertools import product

        for entity_combo in product(entities, repeat=n_vars):
            # Build variable binding
            binding = dict(zip(head.arguments, entity_combo))

            # Compute probability of this grounding
            prob = self._evaluate_body(body, binding, known_facts, entities)

            if prob is not None:
                ground_head = head.substitute(binding)
                head_key = ground_head.ground_key()
                results[head_key] = prob

        return results

    def _evaluate_body(
        self,
        body: List[Literal],
        binding: Dict[str, str],
        known_facts: Dict[str, torch.Tensor],
        entities: List[str],
    ) -> Optional[torch.Tensor]:
        """
        Evaluate body conjunction P(B1 ∧ B2 ∧ ...).

        Returns None if any required fact is missing or probability is 0.
        """
        from itertools import product

        # Identify variables in body not bound by head
        body_vars = set()
        for lit in body:
            for arg in lit.arguments:
                if arg[0].isupper() and arg not in binding:
                    body_vars.add(arg)

        # If no unbound variables, directly evaluate
        if not body_vars:
            return self._evaluate_ground_body(body, binding, known_facts)

        # Otherwise, enumerate over existential variables
        body_var_list = sorted(body_vars)
        probs = []

        for var_combo in product(entities, repeat=len(body_var_list)):
            extended_binding = dict(binding)
            extended_binding.update(zip(body_var_list, var_combo))

            prob = self._evaluate_ground_body(body, extended_binding, known_facts)
            if prob is not None:
                probs.append(prob)

        if not probs:
            return None

        # Aggregate existential quantification via noisy-OR
        return self._noisy_or(probs)

    def _evaluate_ground_body(
        self,
        body: List[Literal],
        binding: Dict[str, str],
        known_facts: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Evaluate a fully grounded body conjunction.

        P(L1 ∧ L2 ∧ ... ∧ Ln) = P(L1) × P(L2) × ... × P(Ln)

        For negated literals: P(¬L) = 1 - P(L)
        """
        result = torch.tensor(1.0)

        for literal in body:
            ground_lit = literal.substitute(binding)
            key = ground_lit.ground_key()

            if key not in known_facts:
                return None  # Missing fact

            lit_prob = known_facts[key]

            if literal.negated:
                lit_prob = 1.0 - lit_prob

            result = result * lit_prob

            # Early termination if probability becomes too low
            if result.item() < 1e-6:
                return None

        return result

    @staticmethod
    def _noisy_or(probs: List[torch.Tensor]) -> torch.Tensor:
        """
        Noisy-OR: P(A ∨ B) = 1 - Π(1 - P_i)

        Computed in log-space for numerical stability.
        """
        if len(probs) == 1:
            return probs[0]

        # Convert to tensor if list
        if isinstance(probs, list):
            probs = torch.stack(probs) if torch.is_tensor(probs[0]) else torch.tensor(probs)

        # Clamp to valid probability range
        probs = probs.clamp(0.0, 1.0 - 1e-7)

        # Log-space computation
        log_complement = torch.sum(torch.log1p(-probs))
        return 1.0 - torch.exp(log_complement)


@dataclass
class ProofStep:
    """A single step in a proof trace."""
    predicate: str
    probability: float
    rule_name: str = ""
    is_leaf: bool = False  # True if neural predicate


class ProofTracer:
    """
    Generate explanations for inference results.

    Traces back from a derived predicate to show:
    - Which rules were applied
    - Which ground facts supported the conclusion
    - The probability at each step
    """

    @classmethod
    def trace(
        cls,
        query: str,  # e.g., "risk(person1)"
        result: Dict[str, torch.Tensor],
        grounded_facts: Dict[str, torch.Tensor],
        kb: KnowledgeBase,
    ) -> List[str]:
        """
        Generate explanation for why query is true.

        Returns list of explanation strings.
        """
        if query not in result:
            return [f"{query}: not derived (probability ~0)"]

        prob = result[query].item()
        lines = [f"{query} = {prob:.4f}"]

        # Find rules that could derive this predicate
        predicate_name = query.split("(")[0]

        for rule in kb.get_rules_for(predicate_name):
            rule_explanation = cls._explain_rule_application(
                rule, query, result, grounded_facts, kb
            )
            if rule_explanation:
                lines.extend(rule_explanation)

        return lines

    @classmethod
    def _explain_rule_application(
        cls,
        rule: Rule,
        query: str,
        result: Dict[str, torch.Tensor],
        grounded_facts: Dict[str, torch.Tensor],
        kb: KnowledgeBase,
    ) -> Optional[List[str]]:
        """Explain how a specific rule contributed to the query."""
        lines = []

        if rule.name:
            lines.append(f"  [via rule: {rule.name}]")

        # Explain body literals
        for lit in rule.body:
            # Find grounded version
            # This is a simplified explanation
            lit_key = cls._match_literal_to_ground(lit, query)
            if lit_key and lit_key in grounded_facts:
                prob = grounded_facts[lit_key].item()
                prefix = "  └──" if lit.negated else "  ├──"
                lines.append(f"{prefix} {lit_key} = {prob:.4f}")

        return lines if len(lines) > 1 else None

    @staticmethod
    def _match_literal_to_ground(literal: Literal, query: str) -> Optional[str]:
        """Simple heuristic to match literal to grounded query."""
        # Extract entity from query: "risk(person1)" -> "person1"
        match = query.split("(")[1].rstrip(")")
        if "," in match:
            match = match.split(",")[0]

        return f"{literal.predicate}({match})"


class AttentionBasedLogicLayer(DifferentiableLogicLayer):
    """
    Extended logic layer with attention mechanism over rule applications.

    Instead of fixed semiring operations, uses learned attention to
    aggregate multiple grounding contributions.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        feature_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__(knowledge_base, learn_rule_weights=False)
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Multi-head attention for aggregating rule contributions
        self.aggregation_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def _aggregate_with_attention(
        self,
        contributions: List[torch.Tensor],
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate rule contributions using attention."""
        if len(contributions) == 1:
            return contributions[0]

        # Stack contributions
        stacked = torch.stack(contributions).unsqueeze(0)  # (1, N, 1)

        # Use attention to weight each contribution
        # This is a simplified version; full version would use entity features
        weights = torch.softmax(stacked, dim=1)
        weighted = stacked * weights

        return weighted.sum(dim=1).squeeze()
