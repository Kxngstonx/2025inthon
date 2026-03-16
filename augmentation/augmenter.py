from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple

from .rules_associative_commutative import (
    apply_commutative_property,
    augment_expression_associative,
)
from .rules_boundary import has_boundary_case
from .rules_ec import generate_equivalent_expressions
from .rules_rc import generate_related_expressions


@dataclass
class AugmentConfig:
    """Per-augmentation probabilities and limits. All 0 = no augmentation."""

    prob_associative: float = 0.0
    prob_commutative: float = 0.0
    prob_boundary: float = 0.0
    prob_ec: float = 0.0
    prob_rc: float = 0.0
    max_equivalent_variations: int = 5
    max_related_variations: int = 4
    boundary_max_tries: int = 10
    # Stage-2 pair/bundle controls
    aux_pair_rate: float = 0.5
    ec_pair_rate: float = 0.5
    rc_pair_rate: float = 0.5
    boundary_append_rate: float = 0.0
    boundary_append_samples: int = 1
    # RC hard-negative quality controls
    rc_min_similarity: float = 0.72
    rc_max_value_delta: int = 30
    rc_hard_negative_ratio: float = 1.0

    def is_enabled(self) -> bool:
        return (
            self.prob_associative > 0
            or self.prob_commutative > 0
            or self.prob_boundary > 0
            or self.prob_ec > 0
            or self.prob_rc > 0
        )


def augment_sample(
    expr: str,
    target: str,
    rng: random.Random,
    config: AugmentConfig,
    gen_expr_fn: Callable[
        [random.Random, int, Tuple[int, int]], Tuple[str, int]
    ],
    num_digits: Tuple[int, int],
    max_depth: int,
) -> Tuple[str, str]:
    """
    Optionally replace (expr, target) with an augmented variant.
    Order: boundary -> associative -> commutative -> EC -> RC.
    Returns (expr_out, target_out); on failure keeps (expr, target).
    """
    expr_out, target_out = expr, target

    if config.prob_boundary > 0 and rng.random() < config.prob_boundary:
        try:
            seed = rng.randint(0, 2**31 - 1)
            for attempt in range(config.boundary_max_tries):
                rng2 = random.Random(seed + attempt)
                depth = rng2.randint(0, max_depth)
                e, v = gen_expr_fn(rng2, depth, num_digits)
                if has_boundary_case(e):
                    expr_out, target_out = e, str(v)
                    break
        except Exception:
            pass

    if config.prob_associative > 0 and rng.random() < config.prob_associative:
        try:
            new_expr, was_transformed = augment_expression_associative(expr_out)
            if was_transformed and new_expr != expr_out:
                expr_out = new_expr
        except Exception:
            pass

    if config.prob_commutative > 0 and rng.random() < config.prob_commutative:
        try:
            new_expr = apply_commutative_property(expr_out)
            if new_expr != expr_out:
                expr_out = new_expr
        except Exception:
            pass

    if config.prob_ec > 0 and rng.random() < config.prob_ec:
        try:
            target_val = int(target_out)
            if target_val >= 0:
                candidates = generate_equivalent_expressions(
                    target_val,
                    max_variations=config.max_equivalent_variations,
                )
                if candidates:
                    expr_out = rng.choice(candidates)
        except (ValueError, TypeError):
            pass
        except Exception:
            pass

    if config.prob_rc > 0 and rng.random() < config.prob_rc:
        try:
            related = generate_related_expressions(
                expr_out,
                num_variations=config.max_related_variations,
            )
            if len(related) >= 1:
                e, v = rng.choice(related)
                expr_out, target_out = e, str(v)
        except Exception:
            pass

    return expr_out, target_out
