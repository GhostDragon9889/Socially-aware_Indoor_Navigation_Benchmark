"""Specialize only declared literal switches before comparing historical ASTs."""
import ast
from copy import deepcopy


def training_branch(node):
    class Specialize(ast.NodeTransformer):
        def visit_Name(self, value):
            if value.id == "pure_evaluation" and isinstance(value.ctx, ast.Load):
                return ast.copy_location(ast.Constant(False), value)
            return value

        def visit_UnaryOp(self, value):
            value = self.generic_visit(value)
            if isinstance(value.op, ast.Not) and isinstance(value.operand, ast.Constant):
                return ast.copy_location(ast.Constant(not value.operand.value), value)
            return value

        def visit_BoolOp(self, value):
            value = self.generic_visit(value)
            if isinstance(value.op, ast.And):
                # Preserve evaluation order: remove only leading literal True.
                while value.values and isinstance(value.values[0], ast.Constant):
                    if not value.values[0].value:
                        return value.values[0]
                    value.values.pop(0)
                if len(value.values) == 1:
                    return value.values[0]
            return value

        def visit_If(self, value):
            value = self.generic_visit(value)
            if isinstance(value.test, ast.Constant):
                return value.body if value.test.value else value.orelse
            return value

    return Specialize().visit(deepcopy(node))
