from __future__ import annotations

import random
import re
from typing import List

from .common import safe_eval


def _extract_operands(expr: str) -> List[int]:
    return [int(m) for m in re.findall(r"\d+", expr)]


def _has_valid_operands(expr: str, max_operand: int = 100000) -> bool:
    return all(op < max_operand for op in _extract_operands(expr))


def generate_equivalent_expressions(
    target_value: int, max_variations: int = 5
) -> List[str]:
    """Generate expressions that evaluate to target_value (for EC augmentation)."""
    categories: dict = {
        "simple": [],
        "addition": [],
        "subtraction": [],
        "multiplication": [],
        "division": [],
        "compound": [],
        "triple": [],
        "parentheses": [],
    }
    max_operand = 100000

    categories["simple"].append(str(target_value))

    addition_count = 0
    for a in range(1, min(target_value, 20)):
        b = target_value - a
        if b > 0 and a < max_operand and b < max_operand and addition_count < 10:
            categories["addition"].append(f"{a}+{b}")
            addition_count += 1

    if target_value >= 0:
        sub_count = 0
        for a in range(target_value + 1, min(target_value + 20, max_operand)):
            b = a - target_value
            if b > 0 and a < max_operand and b < max_operand and sub_count < 10:
                categories["subtraction"].append(f"{a}-{b}")
                sub_count += 1

    mult_count = 0
    for a in range(2, min(target_value, 20)):
        if target_value % a == 0:
            b = target_value // a
            if b > 0 and a < max_operand and b < max_operand and mult_count < 10:
                categories["multiplication"].append(f"{a}*{b}")
                if a != b:
                    categories["multiplication"].append(f"{b}*{a}")
                mult_count += 1

    if target_value > 0:
        for b in range(2, 10):
            a = target_value * b
            if a < max_operand and b < max_operand:
                categories["division"].append(f"{a}//{b}")

    if target_value > 0:
        compound_count = 0
        for a in range(2, min(10, target_value // 2 + 1)):
            for b in range(2, 10):
                product = a * b
                if product < target_value:
                    c = target_value - product
                    if (
                        c > 0
                        and a < max_operand
                        and b < max_operand
                        and c < max_operand
                        and compound_count < 5
                    ):
                        categories["compound"].append(f"{a}*{b}+{c}")
                        categories["compound"].append(f"{c}+{a}*{b}")
                        compound_count += 1
                        break
            if compound_count >= 5:
                break

        for a in range(1, min(target_value, 15, max_operand)):
            remainder = target_value - a
            for b in range(2, min(10, remainder // 2 + 1, max_operand)):
                if remainder % b == 0:
                    c = remainder // b
                    if (
                        c > 0
                        and a < max_operand
                        and b < max_operand
                        and c < max_operand
                        and compound_count < 10
                    ):
                        categories["compound"].append(f"{a}+{b}*{c}")
                        categories["compound"].append(f"{b}*{c}+{a}")
                        compound_count += 1
                        break
            if compound_count >= 10:
                break

        for a in range(2, min(10, max_operand)):
            for b in range(2, min(10, max_operand)):
                product = a * b
                if product > target_value:
                    c = product - target_value
                    if (
                        c > 0
                        and a < max_operand
                        and b < max_operand
                        and c < max_operand
                        and compound_count < 15
                    ):
                        categories["compound"].append(f"{a}*{b}-{c}")
                        compound_count += 1
                        break
            if compound_count >= 15:
                break

    if target_value >= 3:
        triple_count = 0
        for a in range(1, min(target_value - 1, 10, max_operand)):
            for b in range(1, min(target_value - a, 10, max_operand)):
                c = target_value - a - b
                if (
                    c > 0
                    and a < max_operand
                    and b < max_operand
                    and c < max_operand
                    and triple_count < 5
                ):
                    categories["triple"].append(f"{a}+{b}+{c}")
                    triple_count += 1
                    break
            if triple_count >= 5:
                break

    if target_value > 0:
        paren_count = 0
        for c in range(2, min(10, target_value // 2 + 1, max_operand)):
            if target_value % c == 0:
                ab_sum = target_value // c
                for a in range(1, min(ab_sum, 10, max_operand)):
                    b = ab_sum - a
                    if (
                        b > 0
                        and a < max_operand
                        and b < max_operand
                        and c < max_operand
                        and paren_count < 5
                    ):
                        categories["parentheses"].append(f"({a}+{b})*{c}")
                        categories["parentheses"].append(f"{c}*({a}+{b})")
                        paren_count += 1
                        break
                if paren_count >= 5:
                    break

        for a in range(2, min(10, target_value // 2 + 1, max_operand)):
            if target_value % a == 0:
                bc_sum = target_value // a
                for b in range(1, min(bc_sum, 10, max_operand)):
                    c = bc_sum - b
                    if (
                        c > 0
                        and a < max_operand
                        and b < max_operand
                        and c < max_operand
                        and paren_count < 10
                    ):
                        categories["parentheses"].append(f"{a}*({b}+{c})")
                        paren_count += 1
                        break
                if paren_count >= 10:
                    break

    category_order = [
        "simple",
        "multiplication",
        "division",
        "compound",
        "parentheses",
        "addition",
        "subtraction",
        "triple",
    ]
    all_variations: List[str] = []
    for cat in category_order:
        all_variations.extend(categories[cat])
    random.shuffle(all_variations)

    validated: List[str] = []
    seen = set()
    for ex in all_variations:
        if ex in seen:
            continue
        seen.add(ex)
        if not _has_valid_operands(ex, max_operand):
            continue
        result = safe_eval(ex)
        if result is not None and result == target_value:
            validated.append(ex)
            if len(validated) >= max_variations:
                break
    return validated


def build_ec_candidates(target: int, max_variations: int) -> List[str]:
    return generate_equivalent_expressions(target, max_variations=max_variations)

