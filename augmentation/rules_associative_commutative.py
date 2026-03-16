from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

from .common import extract_paren_content


class ASTNode:
    pass


class NumberNode(ASTNode):
    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return f"Num({self.value})"


class BinOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"Op({self.op}, {self.left}, {self.right})"


def _tokenize(expression: str) -> List[str]:
    """Tokenize expression (e.g. ['(', '1', '+', '2', ')']). // is one token."""
    return re.findall(r"(\d+|\/\/|[+\-*/()])", expression)


def _to_rpn(tokens: List[str]) -> List[str]:
    precedence = {"+": 1, "-": 1, "*": 2, "//": 2}
    output_queue: List[str] = []
    operator_stack: List[str] = []

    for token in tokens:
        if token.isdigit():
            output_queue.append(token)
        elif token in precedence:
            while (
                operator_stack
                and operator_stack[-1] != "("
                and precedence[operator_stack[-1]] >= precedence[token]
            ):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == "(":
                operator_stack.pop()
    while operator_stack:
        output_queue.append(operator_stack.pop())
    return output_queue


def _rpn_to_ast(rpn_tokens: List[str]) -> ASTNode:
    stack: List[ASTNode] = []
    for token in rpn_tokens:
        if token.isdigit():
            stack.append(NumberNode(token))
        else:
            if len(stack) < 2:
                raise ValueError(f"Missing operands for operator '{token}'.")
            right = stack.pop()
            left = stack.pop()
            stack.append(BinOpNode(token, left, right))
    if len(stack) != 1:
        raise ValueError("Invalid expression: stack must have exactly one node.")
    return stack[0]


def _apply_associative_augmentation(node: ASTNode) -> ASTNode:
    """Apply associative law randomly: (a op b) op c <-> a op (b op c) for + and *."""
    if not isinstance(node, BinOpNode):
        return node

    node.left = _apply_associative_augmentation(node.left)
    node.right = _apply_associative_augmentation(node.right)

    if node.op not in ("+", "*"):
        return node

    can_transform_left = isinstance(node.left, BinOpNode) and node.left.op == node.op
    can_transform_right = isinstance(node.right, BinOpNode) and node.right.op == node.op
    if not (can_transform_left or can_transform_right):
        return node
    if random.random() < 0.5:
        return node

    if isinstance(node.left, BinOpNode) and node.left.op == node.op:
        try:
            new_right_child = BinOpNode(node.op, node.left.right, node.right)
            return BinOpNode(node.op, node.left.left, new_right_child)
        except Exception:
            return node

    if isinstance(node.right, BinOpNode) and node.right.op == node.op:
        try:
            new_left_child = BinOpNode(node.op, node.left, node.right.left)
            return BinOpNode(node.op, new_left_child, node.right.right)
        except Exception:
            return node

    return node


def _ast_to_string(
    node: ASTNode, parent_op_prec: int = 0, force_parens: bool = False
) -> str:
    precedence = {"+": 1, "-": 1, "*": 2, "//": 2}

    if isinstance(node, NumberNode):
        return node.value
    if isinstance(node, BinOpNode):
        current_op_prec = precedence.get(node.op, 0)
        left_str = _ast_to_string(node.left, current_op_prec, force_parens)
        right_str = _ast_to_string(node.right, current_op_prec, force_parens)
        if force_parens:
            if isinstance(node.left, BinOpNode) and node.left.op == node.op:
                left_str = f"({left_str})"
            if isinstance(node.right, BinOpNode) and node.right.op == node.op:
                right_str = f"({right_str})"
        elif isinstance(node.right, BinOpNode) and node.op in ("-", "//"):
            if precedence.get(node.right.op, 0) == current_op_prec:
                right_str = f"({right_str})"
        s = f"{left_str}{node.op}{right_str}"
        if not force_parens and current_op_prec < parent_op_prec:
            return f"({s})"
        return s
    raise TypeError("Unknown node type.")


def _ast_equals(ast1: ASTNode, ast2: ASTNode) -> bool:
    if type(ast1) != type(ast2):
        return False
    if isinstance(ast1, NumberNode):
        return ast1.value == ast2.value
    if isinstance(ast1, BinOpNode):
        return (
            ast1.op == ast2.op
            and _ast_equals(ast1.left, ast2.left)
            and _ast_equals(ast1.right, ast2.right)
        )
    return False


def augment_expression_associative(expression_string: str) -> Tuple[str, bool]:
    """Apply associative law to expression. Returns (new_expression, was_transformed)."""
    try:
        tokens = _tokenize(expression_string)
        rpn = _to_rpn(tokens)
        original_ast = _rpn_to_ast(rpn)
        augmented_ast = _apply_associative_augmentation(original_ast)
        was_transformed = not _ast_equals(original_ast, augmented_ast)
        new_expression = _ast_to_string(augmented_ast, force_parens=True)
        return new_expression, was_transformed
    except Exception:
        return expression_string, False


def _find_outermost_commutative_op(expr: str) -> Tuple[int, Optional[str]]:
    """Find outermost + or * (by precedence). Returns (position, op) or (-1, None)."""
    depth = 0
    outermost_pos = -1
    outermost_op: Optional[str] = None
    outermost_precedence = 999

    for i, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0:
            if char in ["+", "-"]:
                precedence = 1
                if precedence < outermost_precedence:
                    outermost_pos = i
                    outermost_op = char
                    outermost_precedence = precedence
                elif precedence == outermost_precedence and i > outermost_pos:
                    outermost_pos = i
                    outermost_op = char
            elif char in ["*", "//"]:
                precedence = 2
                if outermost_precedence == 999 and (i > outermost_pos or outermost_pos == -1):
                    outermost_pos = i
                    outermost_op = char
                    outermost_precedence = precedence

    if outermost_op in ["+", "*"]:
        return outermost_pos, outermost_op
    return -1, None


def _split_expression(expr: str, op_pos: int) -> Tuple[str, str, str]:
    left = expr[:op_pos].strip()
    right = expr[op_pos + 1 :].strip()
    op = expr[op_pos]
    return left, op, right


def apply_commutative_property(expr: str) -> str:
    """Apply commutative property at top level (swap operands of + or *). Recursive."""
    expr = expr.strip()
    if expr.isdigit():
        return expr

    paren_content = extract_paren_content(expr)
    if paren_content is not None:
        inner = apply_commutative_property(paren_content)
        if inner != paren_content:
            return f"({inner})"
        return expr

    op_pos, _ = _find_outermost_commutative_op(expr)

    if op_pos == -1:
        depth = 0
        outermost_pos = -1
        outermost_precedence = 999
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0:
                if char in ["+", "-"]:
                    precedence = 1
                    if precedence < outermost_precedence:
                        outermost_pos = i
                        outermost_precedence = precedence
                    elif precedence == outermost_precedence and i > outermost_pos:
                        outermost_pos = i
                elif char in ["*", "//"]:
                    precedence = 2
                    if outermost_precedence == 999 and (i > outermost_pos or outermost_pos == -1):
                        outermost_pos = i
                        outermost_precedence = precedence
        if outermost_pos != -1:
            left, operator, right = _split_expression(expr, outermost_pos)
            left_t = apply_commutative_property(left)
            right_t = apply_commutative_property(right)
            return f"{left_t}{operator}{right_t}"
        return expr

    left, operator, right = _split_expression(expr, op_pos)
    left_t = apply_commutative_property(left)
    right_t = apply_commutative_property(right)
    return f"{right_t}{operator}{left_t}"


def build_assoc_candidate(expr: str) -> Tuple[str, bool]:
    return augment_expression_associative(expr)


def build_comm_candidate(expr: str) -> str:
    return apply_commutative_property(expr)

