from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Tuple

from .rules_associative_commutative import build_assoc_candidate, build_comm_candidate
from .rules_boundary import sample_boundary_expressions
from .rules_ec import build_ec_candidates
from .rules_rc import build_rc_candidates, generate_hard_negative_rc


def build_training_bundle(
    expr: str,
    target: str,
    rng: random.Random,
    config: Any,
    gen_expr_fn: Callable[[random.Random, int, Tuple[int, int]], Tuple[str, int]],
    num_digits: Tuple[int, int],
    max_depth: int,
) -> Dict[str, Any]:
    base_target_int = int(target)
    bundle: Dict[str, Any] = {
        "base": {"expr": expr, "target": target},
        "aux_pairs": [],
        "ec_pairs": [],
        "rc_pairs": [],
        "boundary_extra": [],
        "rc_quality": {},
    }

    aux_pair_rate = float(getattr(config, "aux_pair_rate", 0.5))
    ec_pair_rate = float(getattr(config, "ec_pair_rate", 0.5))
    rc_pair_rate = float(getattr(config, "rc_pair_rate", 0.5))
    boundary_append_rate = float(getattr(config, "boundary_append_rate", 0.0))

    if getattr(config, "prob_associative", 0.0) > 0 and rng.random() < aux_pair_rate:
        aug_expr, transformed = build_assoc_candidate(expr)
        if transformed and aug_expr != expr:
            bundle["aux_pairs"].append(
                {"base_expr": expr, "aug_expr": aug_expr, "target": target, "kind": "associative"}
            )

    if getattr(config, "prob_commutative", 0.0) > 0 and rng.random() < aux_pair_rate:
        aug_expr = build_comm_candidate(expr)
        if aug_expr != expr:
            bundle["aux_pairs"].append(
                {"base_expr": expr, "aug_expr": aug_expr, "target": target, "kind": "commutative"}
            )

    if getattr(config, "prob_ec", 0.0) > 0 and rng.random() < ec_pair_rate:
        ec_candidates = build_ec_candidates(base_target_int, getattr(config, "max_equivalent_variations", 5) * 2)
        for cand in ec_candidates:
            if cand != expr:
                bundle["ec_pairs"].append({"base_expr": expr, "aug_expr": cand, "target": target})
                break

    if getattr(config, "prob_rc", 0.0) > 0 and rng.random() < rc_pair_rate:
        min_similarity = float(getattr(config, "rc_min_similarity", 0.72))
        max_value_delta = int(getattr(config, "rc_max_value_delta", 30))
        hard_rc, rc_stats = generate_hard_negative_rc(
            expr,
            base_target_int,
            min_similarity=min_similarity,
            max_value_delta=max_value_delta,
        )
        if hard_rc:
            cand_expr, cand_val, cand_sim, cand_delta = hard_rc[0]
            bundle["rc_pairs"].append(
                {
                    "base_expr": expr,
                    "aug_expr": cand_expr,
                    "base_target": target,
                    "aug_target": str(cand_val),
                    "similarity": cand_sim,
                    "value_delta": cand_delta,
                }
            )
        else:
            rc_candidates = build_rc_candidates(expr, max_variations=getattr(config, "max_related_variations", 4))
            for cand_expr, cand_val in rc_candidates:
                if cand_expr != expr and cand_val != base_target_int:
                    bundle["rc_pairs"].append(
                        {
                            "base_expr": expr,
                            "aug_expr": cand_expr,
                            "base_target": target,
                            "aug_target": str(cand_val),
                            "similarity": 0.0,
                            "value_delta": abs(cand_val - base_target_int),
                        }
                    )
                    break
            rc_stats = None

        if rc_stats is not None:
            bundle["rc_quality"] = {
                "generated": rc_stats.generated,
                "accepted": rc_stats.accepted,
                "accept_rate": rc_stats.accept_rate,
                "avg_similarity": rc_stats.avg_similarity,
                "avg_value_delta": rc_stats.avg_value_delta,
            }

    if getattr(config, "prob_boundary", 0.0) > 0 and rng.random() < boundary_append_rate:
        n_boundary_samples = int(getattr(config, "boundary_append_samples", 1))
        extra = sample_boundary_expressions(
            rng=rng,
            gen_expr_fn=gen_expr_fn,
            num_digits=num_digits,
            max_depth=max_depth,
            boundary_max_tries=getattr(config, "boundary_max_tries", 10),
            n_samples=max(1, n_boundary_samples),
        )
        for e, t in extra:
            bundle["boundary_extra"].append({"expr": e, "target": t})

    return bundle

