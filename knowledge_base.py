"""
Knowledge Base — Symbolic Rule Definitions
===========================================
Defines the logic program (Prolog-style rules) that the
differentiable logic layer will execute.

Each rule is:
    head :- body_1, body_2, ..., body_n.

Where body literals can be:
    - Neural predicates (grounded by neural nets)
    - Derived predicates (defined by other rules)

Semantics:
    P(head) = 1 - ∏(1 - P(body_1) * P(body_2) * ... * P(body_n))
              (noisy-OR over all rules with the same head)

This is the standard DeepProbLog semiring-based inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional


@dataclass
class Literal:
    """
    A logical literal, e.g., near(X, Fire) or risk(X).
    
    predicate: predicate name (e.g., "near")
    arguments: list of variable/constant names (e.g., ["X", "Fire"])
    negated:   whether this is \+ (negation as failure)
    """
    predicate: str
    arguments: List[str]
    negated: bool = False

    def __str__(self):
        args = ", ".join(self.arguments)
        prefix = "\\+" if self.negated else ""
        return f"{prefix}{self.predicate}({args})"

    def substitute(self, bindings: Dict[str, str]) -> "Literal":
        """Apply variable bindings to create a ground literal."""
        new_args = [bindings.get(a, a) for a in self.arguments]
        return Literal(self.predicate, new_args, self.negated)

    def ground_key(self) -> str:
        """String key for looking up in grounded fact base."""
        args = ",".join(self.arguments)
        return f"{self.predicate}({args})"

    def is_ground(self, constants: Set[str]) -> bool:
        """Check if all arguments are constants (fully ground)."""
        return all(a in constants for a in self.arguments)


@dataclass
class Rule:
    """
    A logic rule: head :- body.
    
    Example:
        risk(X) :- near(X, Y), flammable(X), is_fire(Y).
    """
    head: Literal
    body: List[Literal]
    weight: float = 1.0  # Optional rule weight (learnable)
    name: str = ""

    def __str__(self):
        body_str = ", ".join(str(b) for b in self.body)
        w = f"[w={self.weight:.2f}] " if self.weight != 1.0 else ""
        return f"{w}{self.head} :- {body_str}."

    def get_variables(self) -> Set[str]:
        """Get all variables in the rule (uppercase identifiers)."""
        variables = set()
        for lit in [self.head] + self.body:
            for arg in lit.arguments:
                if arg[0].isupper():
                    variables.add(arg)
        return variables


@dataclass
class KnowledgeBase:
    """
    A collection of logic rules forming the reasoning program.
    
    Supports:
    - Adding rules
    - Querying which rules define a given predicate
    - Topological ordering of predicates for forward chaining
    - Listing all neural (base) predicates needed
    """
    rules: List[Rule] = field(default_factory=list)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def add_rule_from_str(
        self, head_pred: str, head_args: List[str],
        body: List[Tuple[str, List[str], bool]], weight: float = 1.0,
        name: str = ""
    ):
        """
        Convenience method to add a rule from components.

        body: list of (predicate_name, [args], negated)
        """
        head = Literal(head_pred, head_args)
        body_lits = [Literal(p, a, n) for p, a, n in body]
        self.add_rule(Rule(head, body_lits, weight, name))

    def get_rules_for(self, predicate: str) -> List[Rule]:
        """Get all rules whose head matches the given predicate."""
        return [r for r in self.rules if r.head.predicate == predicate]

    @property
    def defined_predicates(self) -> Set[str]:
        """Predicates that appear as rule heads (derived predicates)."""
        return {r.head.predicate for r in self.rules}

    @property
    def base_predicates(self) -> Set[str]:
        """Predicates that appear only in bodies (neural predicates)."""
        body_preds = set()
        for r in self.rules:
            for lit in r.body:
                body_preds.add(lit.predicate)
        return body_preds - self.defined_predicates

    @property
    def all_predicates(self) -> Set[str]:
        return self.defined_predicates | self.base_predicates

    def dependency_order(self) -> List[str]:
        """
        Topological sort of derived predicates.
        Base predicates come first, then derived in dependency order.
        """
        # Build dependency graph
        deps: Dict[str, Set[str]] = {}
        for pred in self.defined_predicates:
            deps[pred] = set()
            for rule in self.get_rules_for(pred):
                for lit in rule.body:
                    if lit.predicate in self.defined_predicates:
                        deps[pred].add(lit.predicate)

        # Kahn's algorithm
        in_degree = {p: len(d) for p, d in deps.items()}
        queue = [p for p, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for p, d in deps.items():
                if node in d:
                    in_degree[p] -= 1
                    if in_degree[p] == 0:
                        queue.append(p)

        if len(order) != len(deps):
            raise ValueError(
                "Cyclic dependencies detected in knowledge base! "
                "DeepProbLog requires acyclic programs for exact inference."
            )

        return list(self.base_predicates) + order

    def __str__(self):
        lines = ["% Knowledge Base", "% " + "=" * 40]
        for r in self.rules:
            lines.append(str(r))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Pre-built Knowledge Bases
# ─────────────────────────────────────────────────────────

def create_risk_kb() -> KnowledgeBase:
    """
    Standard risk assessment knowledge base.
    
    Encodes common-sense safety rules:
    
    % Fire risks
    fire_risk(X) :- near(X, Y), flammable(X), is_fire(Y).
    fire_risk(X) :- near(X, Y), is_electrical(X), liquid(Y).
    
    % Sharp object risks  
    cut_risk(X) :- holding(X, Y), sharp(Y), running(X).
    cut_risk(X) :- near(X, Y), sharp(Y), is_child(X).
    
    % Fall risks
    fall_risk(X) :- above(X, Y), heavy(X).
    fall_risk(X) :- running(X), near(X, Y), fragile(Y).
    
    % Chemical risks
    chemical_risk(X) :- near(X, Y), toxic(Y), \\+is_container(Y).
    
    % Aggregate risk
    risk(X) :- fire_risk(X).
    risk(X) :- cut_risk(X).
    risk(X) :- fall_risk(X).
    risk(X) :- chemical_risk(X).
    """
    kb = KnowledgeBase()

    # ── Fire risks ──
    kb.add_rule_from_str(
        "fire_risk", ["X"],
        [("near", ["X", "Y"], False),
         ("flammable", ["X"], False),
         ("is_fire", ["Y"], False)],
        name="fire_near_flammable"
    )
    kb.add_rule_from_str(
        "fire_risk", ["X"],
        [("near", ["X", "Y"], False),
         ("is_electrical", ["X"], False),
         ("liquid", ["Y"], False)],
        name="electrical_near_liquid"
    )

    # ── Sharp object / cut risks ──
    kb.add_rule_from_str(
        "cut_risk", ["X"],
        [("holding", ["X", "Y"], False),
         ("sharp", ["Y"], False),
         ("running", ["X"], False)],
        name="running_with_sharp"
    )
    kb.add_rule_from_str(
        "cut_risk", ["X"],
        [("near", ["X", "Y"], False),
         ("sharp", ["Y"], False),
         ("is_child", ["X"], False)],
        name="child_near_sharp"
    )

    # ── Fall risks ──
    kb.add_rule_from_str(
        "fall_risk", ["X"],
        [("above", ["X", "Y"], False),
         ("heavy", ["X"], False)],
        name="heavy_above"
    )
    kb.add_rule_from_str(
        "fall_risk", ["X"],
        [("running", ["X"], False),
         ("near", ["X", "Y"], False),
         ("fragile", ["Y"], False)],
        name="running_near_fragile"
    )

    # ── Chemical risks ──
    kb.add_rule_from_str(
        "chemical_risk", ["X"],
        [("near", ["X", "Y"], False),
         ("toxic", ["Y"], False),
         ("is_container", ["Y"], True)],  # negation: not in container
        name="toxic_not_contained"
    )

    # ── Aggregate risk (noisy-OR over sub-risks) ──
    kb.add_rule_from_str(
        "risk", ["X"],
        [("fire_risk", ["X"], False)],
        name="risk_from_fire"
    )
    kb.add_rule_from_str(
        "risk", ["X"],
        [("cut_risk", ["X"], False)],
        name="risk_from_cut"
    )
    kb.add_rule_from_str(
        "risk", ["X"],
        [("fall_risk", ["X"], False)],
        name="risk_from_fall"
    )
    kb.add_rule_from_str(
        "risk", ["X"],
        [("chemical_risk", ["X"], False)],
        name="risk_from_chemical"
    )

    return kb
