from __future__ import annotations

import random
import re
from typing import Callable, List, Optional, Tuple

from .common import extract_paren_content


def _evaluate_expression(expr: str) -> Optional[int]:
    """Safe expression evaluation without eval (for boundary check)."""
    try:
        expr = expr.strip()
        if expr.isdigit():
            return int(expr)
        allowed = set("0123456789+-*/( )")
        if not all(c in allowed or c.isdigit() for c in expr):
            return None

        expr_safe = expr.replace("//", "__FLOORDIV__")
        while "(" in expr_safe:
            start = expr_safe.rfind("(")
            end = expr_safe.find(")", start)
            if end == -1:
                return None
            inner = expr_safe[start + 1 : end]
            inner_val = _evaluate_expression(inner.replace("__FLOORDIV__", "//"))
            if inner_val is None:
                return None
            expr_safe = expr_safe[:start] + str(inner_val) + expr_safe[end + 1 :]

        while "__FLOORDIV__" in expr_safe:
            match = re.search(r"(\d+)\s*__FLOORDIV__\s*(\d+)", expr_safe)
            if not match:
                break
            left = int(match.group(1))
            right = int(match.group(2))
            if right == 0:
                return None
            result = left // right
            expr_safe = expr_safe[: match.start()] + str(result) + expr_safe[match.end() :]

        while "*" in expr_safe:
            match = re.search(r"(\d+)\s*\*\s*(\d+)", expr_safe)
            if not match:
                break
            left = int(match.group(1))
            right = int(match.group(2))
            result = left * right
            expr_safe = expr_safe[: match.start()] + str(result) + expr_safe[match.end() :]

        while "+" in expr_safe or "-" in expr_safe:
            last_plus = expr_safe.rfind("+")
            last_minus = expr_safe.rfind("-")
            op_pos = -1
            op = None
            if last_plus > last_minus:
                if last_plus > 0 and expr_safe[last_plus - 1].isdigit():
                    op_pos = last_plus
                    op = "+"
            elif last_minus > 0 and expr_safe[last_minus - 1].isdigit():
                op_pos = last_minus
                op = "-"
            if op_pos == -1:
                break

            left_part = expr_safe[:op_pos].rstrip()
            right_part = expr_safe[op_pos + 1 :].lstrip()
            left_match = re.search(r"(\d+)\s*$", left_part)
            right_match = re.search(r"^(\d+)", right_part)
            if not (left_match and right_match):
                break

            left = int(left_match.group(1))
            right = int(right_match.group(1))
            result = left + right if op == "+" else left - right
            expr_safe = (
                left_part[: left_match.start()] + str(result) + right_part[right_match.end() :]
            )

        expr_safe = expr_safe.strip()
        if expr_safe.isdigit() or (expr_safe.startswith("-") and expr_safe[1:].isdigit()):
            return int(expr_safe)
        return None
    except Exception:
        return None


def _has_carry_in_addition(left: int, right: int) -> Tuple[bool, bool]:
    """(has_carry, digit_increase)."""
    left_abs = abs(left)
    right_abs = abs(right)
    max_input_digits = max(len(str(left_abs)), len(str(right_abs)))
    result = left_abs + right_abs
    result_digits = len(str(result))
    digit_increase = result_digits > max_input_digits

    max_len = max(len(str(left_abs)), len(str(right_abs)))
    left_str = str(left_abs).zfill(max_len)
    right_str = str(right_abs).zfill(max_len)
    has_carry = False
    carry = 0
    for i in range(max_len - 1, -1, -1):
        digit_sum = int(left_str[i]) + int(right_str[i]) + carry
        if digit_sum >= 10:
            has_carry = True
        carry = digit_sum // 10
    return has_carry, digit_increase


def _has_borrow_in_subtraction(left: int, right: int) -> Tuple[bool, bool]:
    """(has_borrow, digit_decrease)."""
    left_abs = abs(left)
    right_abs = abs(right)
    max_input_digits = max(len(str(left_abs)), len(str(right_abs)))
    result = left - right
    result_abs = abs(result)
    result_digits = len(str(result_abs)) if result != 0 else 1
    digit_decrease = result_digits < max_input_digits

    if result < 0:
        return True, digit_decrease

    max_len = max(len(str(left_abs)), len(str(right_abs)))
    right_str = str(right_abs).zfill(max_len)
    left_str = str(left_abs).zfill(max_len)
    has_borrow = False
    borrow = 0
    for i in range(max_len - 1, -1, -1):
        left_digit = int(left_str[i])
        right_digit = int(right_str[i])
        effective_left = left_digit - borrow
        if effective_left < right_digit:
            has_borrow = True
        borrow = 1 if effective_left < right_digit else 0
    return has_borrow, digit_decrease


def _find_outermost_add_sub_op(expr: str) -> Tuple[int, Optional[str], str, str]:
    """(op_pos, op, left_expr, right_expr)."""
    depth = 0
    outermost_pos = -1
    outermost_op = None
    for i, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in ["+", "-"]:
            if i > outermost_pos or outermost_pos == -1:
                outermost_pos = i
                outermost_op = char
    if outermost_pos == -1:
        return -1, None, "", ""
    left = expr[:outermost_pos].strip()
    right = expr[outermost_pos + 1 :].strip()
    return outermost_pos, outermost_op, left, right


def has_boundary_case(expr: str) -> bool:
    """True if expression involves carry (addition) or borrow (subtraction) with digit change."""
    expr = expr.strip()
    if expr.isdigit():
        return False

    paren_content = extract_paren_content(expr)
    if paren_content is not None:
        return has_boundary_case(paren_content)

    op_pos, op, left_expr, right_expr = _find_outermost_add_sub_op(expr)
    if op_pos == -1:
        return False

    left_val = _evaluate_expression(left_expr)
    right_val = _evaluate_expression(right_expr)
    if left_val is None or right_val is None:
        return has_boundary_case(left_expr) or has_boundary_case(right_expr)

    if op == "+":
        has_carry, digit_increase = _has_carry_in_addition(left_val, right_val)
        if has_carry and digit_increase:
            return True
    elif op == "-":
        has_borrow, digit_decrease = _has_borrow_in_subtraction(left_val, right_val)
        if has_borrow and digit_decrease:
            return True

    if has_boundary_case(left_expr) or has_boundary_case(right_expr):
        return True
    return False


def sample_boundary_expressions(
    rng: random.Random,
    gen_expr_fn: Callable[[random.Random, int, Tuple[int, int]], Tuple[str, int]],
    num_digits: Tuple[int, int],
    max_depth: int,
    boundary_max_tries: int,
    n_samples: int = 1,
) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    seed = rng.randint(0, 2**31 - 1)
    attempt = 0
    while len(samples) < n_samples and attempt < max(1, boundary_max_tries) * n_samples:
        rng2 = random.Random(seed + attempt)
        depth = rng2.randint(0, max_depth)
        expr, value = gen_expr_fn(rng2, depth, num_digits)
        if has_boundary_case(expr):
            samples.append((expr, str(value)))
        attempt += 1
    return samples

