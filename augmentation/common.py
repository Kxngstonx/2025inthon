from __future__ import annotations

from typing import Optional


def safe_eval(expr: str) -> Optional[int]:
    """Evaluate expression safely (supports //). Used by EC and RC."""
    if not expr or not expr.strip():
        return None
    try:
        result = eval(expr)
        return int(result)
    except (ZeroDivisionError, SyntaxError, NameError, TypeError, ValueError, Exception):
        return None


def extract_paren_content(expr: str) -> Optional[str]:
    """Extract content inside a single outer parentheses pair."""
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    return None
        if depth == 0:
            return expr[1:-1].strip()
    return None
