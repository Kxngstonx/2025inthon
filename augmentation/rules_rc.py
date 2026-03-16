from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .common import safe_eval


@dataclass
class RCQualityStats:
    generated: int = 0
    accepted: int = 0
    avg_similarity: float = 0.0
    avg_value_delta: float = 0.0

    @property
    def accept_rate(self) -> float:
        if self.generated == 0:
            return 0.0
        return self.accepted / self.generated


def expression_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def generate_related_expressions(
    expr: str, num_variations: int = 3
) -> List[Tuple[str, int]]:
    """Generate related (expression, result) pairs for RC augmentation."""
    if expr.strip().isdigit():
        result = safe_eval(expr)
        return [(expr, result)] if result is not None else []

    original_result = safe_eval(expr)
    if original_result is None:
        return []

    strategy_candidates: dict = {
        "add_end": [],
        "add_paren": [],
        "multiply": [],
        "divide": [],
        "subtract": [],
        "add_front": [],
    }
    for offset in [1, 2, 3, 5, 9, 10, -1, -2, -3, -5]:
        new_expr = f"{expr}+{offset}"
        result = safe_eval(new_expr)
        if result is not None:
            strategy_candidates["add_end"].append((new_expr, result))
    for offset in [1, 2, 3, 5, 10, -1, -2, -3, -5, -10]:
        new_expr = f"({expr})+{offset}"
        result = safe_eval(new_expr)
        if result is not None:
            strategy_candidates["add_paren"].append((new_expr, result))
    for mult in [2, 3, 4, 5]:
        new_expr = f"({expr})*{mult}"
        result = safe_eval(new_expr)
        if result is not None:
            strategy_candidates["multiply"].append((new_expr, result))
    for divisor in [2, 3, 4, 5]:
        if original_result % divisor == 0:
            new_expr = f"({expr})//{divisor}"
            result = safe_eval(new_expr)
            if result is not None:
                strategy_candidates["divide"].append((new_expr, result))
    for offset in [1, 2, 3, 5, 10]:
        new_expr = f"({expr})-{offset}"
        result = safe_eval(new_expr)
        if result is not None:
            strategy_candidates["subtract"].append((new_expr, result))
    for offset in [1, 2, 3, 5, 9, 10]:
        new_expr = f"{offset}+({expr})"
        result = safe_eval(new_expr)
        if result is not None:
            strategy_candidates["add_front"].append((new_expr, result))

    variations = [(expr, original_result)]
    seen_exprs = {expr}
    strategy_keys = list(strategy_candidates.keys())
    strategy_indices = {k: 0 for k in strategy_keys}

    while len(variations) < num_variations:
        added = False
        for key in strategy_keys:
            candidates = strategy_candidates[key]
            idx = strategy_indices[key]
            if idx < len(candidates):
                new_expr, new_result = candidates[idx]
                if new_expr not in seen_exprs:
                    variations.append((new_expr, new_result))
                    seen_exprs.add(new_expr)
                    added = True
                    if len(variations) >= num_variations:
                        break
            strategy_indices[key] += 1
        if not added:
            break

    return variations[:num_variations] if len(variations) >= 3 else []


def _collect_edit_candidates(expr: str) -> List[str]:
    candidates: List[str] = []
    for old, new in [("+", "-"), ("-", "+"), ("*", "+"), ("+", "*")]:
        if old in expr:
            candidates.append(expr.replace(old, new, 1))
    for delta in [1, 2, 3]:
        candidates.append(f"({expr})+{delta}")
        candidates.append(f"({expr})-{delta}")
        candidates.append(f"{delta}+({expr})")
    return candidates


def generate_hard_negative_rc(
    base_expr: str,
    base_target: int,
    min_similarity: float,
    max_value_delta: int,
    max_candidates: int = 24,
) -> Tuple[List[Tuple[str, int, float, int]], RCQualityStats]:
    accepted: List[Tuple[str, int, float, int]] = []
    sims: List[float] = []
    deltas: List[int] = []

    generated_set = set()
    generated: List[str] = []

    for expr, _ in generate_related_expressions(base_expr, num_variations=max_candidates):
        if expr != base_expr and expr not in generated_set:
            generated_set.add(expr)
            generated.append(expr)

    for cand in _collect_edit_candidates(base_expr):
        if cand not in generated_set:
            generated_set.add(cand)
            generated.append(cand)

    stats = RCQualityStats(generated=len(generated))
    for cand in generated:
        value = safe_eval(cand)
        if value is None:
            continue
        delta = abs(value - base_target)
        if value == base_target:
            continue
        if delta > max_value_delta:
            continue
        sim = expression_similarity(base_expr, cand)
        if sim < min_similarity:
            continue
        accepted.append((cand, value, sim, delta))
        sims.append(sim)
        deltas.append(delta)

    accepted.sort(key=lambda x: (-x[2], x[3]))
    stats.accepted = len(accepted)
    if sims:
        stats.avg_similarity = float(sum(sims) / len(sims))
    if deltas:
        stats.avg_value_delta = float(sum(deltas) / len(deltas))
    return accepted, stats


def build_rc_candidates(expr: str, max_variations: int) -> List[Tuple[str, int]]:
    return generate_related_expressions(expr, num_variations=max_variations)

