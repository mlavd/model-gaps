import random
import string
from dataclasses import dataclass
from typing import Callable, List

class Node:
    def evaluate(self, env:dict=None) -> bool:
        """Evaluates the node

        Args:
            env (dict, optional): Dictionary to store variables. Defaults to None.

        Returns:
            bool: True or False
        """
        if env is None: env = {}
        return self._evaluate(env)

    def _evaluate(self, env:dict) -> bool:
        """Internal evaluate with non-optional env."""
        raise NotImplementedError()


@dataclass
class AssignmentNode(Node):
    variable: 'VariableNode'
    value: Node
    TYPE = 'assignment'

    @staticmethod
    def random_int(variable, min=-10_000, max=10_000):
        """Assignment of variable to random integer"""
        value = ValueNode.random_int(min, max)
        return AssignmentNode(variable, value)

    def _evaluate(self, env):
        env[self.variable.name] = self.value.evaluate(env)

    def __str__(self):
        return f'{self.variable} = {self.value}'


@dataclass
class OperatorNode(Node):
    left: Node
    operator: str
    right: Node
    operation: Callable
    TYPE = 'operator'

    def __str__(self):
        return f'{self.left} {self.operator} {self.right}'

    def _evaluate(self, env):
        l_val = self.left.evaluate(env)
        r_val = self.right.evaluate(env)
        return self.operation(l_val, r_val)


class CalculationNode(OperatorNode):
    TYPE = 'calculation'
    OPERATORS = [ '+', '-', '*', '/', '%' ]

    OPERATIONS = {
        '+': lambda l, r: l + r,
        '-': lambda l, r: l - r,
        '*': lambda l, r: l * r,
        '/': lambda l, r: l / r,
        '%': lambda l, r: l % r,
    }

    @staticmethod
    def random(left, right):
        op = random.choice(CalculationNode.OPERATORS)
        return CalculationNode(left, op, right)

    def __init__(self, left, operator, right):
        super().__init__(
            left, operator, right,
            CalculationNode.OPERATIONS[operator])


class ComparisonNode(OperatorNode):
    TYPE = 'comparison'
    RELATIONAL = [ '<', '<=', '>', '>=' ]
    EQUALITY = [ '==', '!=' ]
    ALL_OPERATORS = RELATIONAL + EQUALITY

    OPERATIONS = {
        '<' : lambda l, r: l <  r,
        '<=': lambda l, r: l <= r,
        '>' : lambda l, r: l >  r,
        '>=': lambda l, r: l >= r,
        '==': lambda l, r: l == r,
        '!=': lambda l, r: l != r,
    }

    @staticmethod
    def random(left, right, include_equality=False):
        op = ComparisonNode.random_operator(include_equality)
        return ComparisonNode(left, op, right)

    @staticmethod
    def random_operator(include_equality=False):
        if include_equality:
            return random.choice(ComparisonNode.ALL_OPERATORS)
        else:
            return random.choice(ComparisonNode.RELATIONAL)

    def __init__(self, left, operator, right):
        super().__init__(
            left, operator, right,
            ComparisonNode.OPERATIONS[operator])


@dataclass
class ConditionalNode(Node):
    condition: Node
    if_node: Node
    else_node: Node
    TYPE = 'conditional'

    def _evaluate(self, env):
        self.was_true = self.condition.evaluate(env)
        if self.was_true:
            return self.if_node.evaluate(env)
        else:
            return self.else_node.evaluate(env)

    def __str__(self):
        r = f'if ({self.condition}) {{\n'
        r += f'\t{self.if_node}\n'
        r += '} else {\n'
        r += f'\t{self.else_node}\n'
        r += '}'
        return r


@dataclass
class SequenceNode(Node):
    nodes: List[Node]
    TYPE = 'sequence'

    def __str__(self):
        return '\n'.join(str(n) for n in self.nodes)

    def _evaluate(self, env):
        result = None
        for node in self.nodes:
            result = node.evaluate(env)

        return result


@dataclass
class ValueNode(Node):
    value: int
    TYPE = 'value'

    @staticmethod
    def random_int(min=-10_000, max=10_000):
        return ValueNode(random.randint(min, max))

    def _evaluate(self, env):
        return self.value

    def __str__(self):
        return str(self.value)


@dataclass
class VariableNode(Node):
    name: str
    TYPE = 'variable'
    VARS = string.ascii_lowercase

    @staticmethod
    def random_variables(count=1):
        if count == 1:
            name = random.choice(VariableNode.VARS)
            return VariableNode(name)

        assert count < len(VariableNode.VARS), 'Too many vars'
        valid = list(VariableNode.VARS)
        random.shuffle(valid)
        return [ VariableNode(valid.pop()) for _ in range(count) ]

    def _evaluate(self, env):
        return env[self.name]

    def __str__(self):
        return self.name
