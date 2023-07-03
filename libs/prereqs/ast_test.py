from .ast import *
from unittest.mock import MagicMock
import pytest

X = VariableNode('x')
Y = VariableNode('y')

class TestNode:
    def test__evaluate(self):
        with pytest.raises(NotImplementedError):
            Node()._evaluate({})


class TestAssignmentNode:
    def test_random_int(self):
        random.seed(10)

        for expected in [ 8723, -8933, 4053, 5812, 8942 ]:
            node = AssignmentNode.random_int(X)
            assert node.value.evaluate() == expected

    def test_evaluate(self):
        node = AssignmentNode(X, ValueNode(1))

        env = {}
        assert node.evaluate(env) is None
        assert 'x' in env
        assert env['x'] == 1

    def test_str(self):
        assert str(AssignmentNode(X, ValueNode(1))) == 'x = 1'
        assert str(AssignmentNode(Y, ValueNode(2))) == 'y = 2'


class TestOperatorNode(Node):
    def test_evaluate(self):
        a = ValueNode(1)
        b = ValueNode(2)

        def operation(l_val, r_val):
            assert l_val == 1
            assert r_val == 2
            return 3
        
        node = OperatorNode(a, None, b, operation)
        assert node.evaluate() == 3

    def test_str(self):
        node = OperatorNode(X, '$*$', Y, None)
        assert str(node) == 'x $*$ y'


class TestCalculationNode:
    @pytest.mark.parametrize('left,op,right,expected', [
        (2, '+', 3, 5),
        (2, '-', 3, -1),
        (2, '*', 3, 6),
        (2, '/', 3, 0.666),
        (2, '%', 3, 2),
    ])
    def test_operations(self, left, op, right, expected):
        left = ValueNode(left)
        right = ValueNode(right)

        node = CalculationNode(left, op, right)
        assert pytest.approx(node.evaluate(), 1e-3) == expected
    
    def test_random(self):
        random.seed(2)
        operators = [
            '+', '+', '+', '*', '-', '*', '*', '%', '-', '%',
            '+', '%', '-', '/', '/', '%', '*', '%', '/', '%'
        ]

        for expected in operators:
            node = CalculationNode.random(X, Y)
            assert node.operator == expected


class TestComparisonNode:
    @pytest.mark.parametrize('left,op,right,expected', [
        (1, '<',  2, True),  (2, '<',  2, False), (2, '<',  1, False),
        (1, '<=', 2, True),  (2, '<=', 2, True),  (2, '<=', 1, False),
        (1, '>',  2, False), (2, '>',  2, False), (2, '>',  1, True),
        (1, '>=', 2, False), (2, '>=', 2, True),  (2, '>=', 1, True),
        (1, '==', 2, False), (2, '==', 2, True),  (2, '==', 1, False),
        (1, '!=', 2, True),  (2, '!=', 2, False), (2, '!=', 1, True),
    ])
    def test_operators(self, left, op, right, expected):
        left = ValueNode(left)
        right = ValueNode(right)

        node = ComparisonNode(left, op, right)
        assert node.evaluate() == expected

    def test_random_with_equality(self):
        random.seed(42)
        operators = [
            '!=', '<', '<', '!=', '>', '<=', '<=', '<=', '!=', '<',
            '!=', '!=', '==', '<', '==', '>=', '<', '<', '<', '<='
        ]

        for expected in operators:
            node = ComparisonNode.random(X, Y, include_equality=True)
            assert node.operator == expected
    
    def test_random_without_equality(self):
        random.seed(42)
        operators = [
            '<', '<', '>', '<=', '<=', '<=', '<', '<', '>=', '<',
            '<', '<', '<=', '<=', '<', '<=', '>=', '<=', '>=', '>'
        ]

        for expected in operators:
            node = ComparisonNode.random(X, Y)
            assert node.operator == expected


class TestConditionalNode:
    def setup_method(self):
        self.if_node = MagicMock(Node)
        self.if_node.evaluate = MagicMock()

        self.else_node = MagicMock(Node)
        self.else_node.evaluate = MagicMock()

        self.condition = MagicMock(Node)

        self.node = ConditionalNode(
            self.condition, self.if_node, self.else_node)

    def test_evaluate_if(self):
        self.condition.evaluate = MagicMock(return_value=True)
        self.node.evaluate({})
        self.if_node.evaluate.assert_called_once()

    def test_evaluate_else(self):
        self.condition.evaluate = MagicMock(return_value=False)
        self.node.evaluate({})
        self.else_node.evaluate.assert_called_once()

    def test_str(self):
        self.condition.__str__ = lambda _: '[cond]'
        self.if_node.__str__ = lambda _: '[if]'
        self.else_node.__str__ = lambda _: '[else]'

        assert str(self.node) == \
            'if ([cond]) {\n\t[if]\n} else {\n\t[else]\n}'


class TestSequenceNode:
    def setup_method(self):
        self.a = MagicMock(Node)
        self.b = MagicMock(Node)
        self.c = MagicMock(Node)
        self.node = SequenceNode([self.a, self.b, self.c])
    
    def test_str(self):
        self.a.__str__ = lambda _: '[A]'
        self.b.__str__ = lambda _: '[B]'
        self.c.__str__ = lambda _: '[C]'
        assert str(self.node) == '[A]\n[B]\n[C]'
    
    @pytest.mark.parametrize('a_val,b_val,c_val,result', [
        (False, False, False, False),
        (False, False, True,  True),
        (False, True,  False, False),
        (False, True,  True,  True),
        (True,  False, False, False),
        (True,  False, True,  True),
        (True,  True,  False, False),
        (True,  True,  True,  True),
    ])
    def test_evaluate(self, a_val, b_val, c_val, result):
        self.a.evaluate = MagicMock(return_value=a_val)
        self.b.evaluate = MagicMock(return_value=b_val)
        self.c.evaluate = MagicMock(return_value=c_val)
        assert self.node.evaluate({}) == result


class TestValueNode:
    def test_str(self):
        assert str(ValueNode('a')) == 'a'
        assert str(ValueNode(1)) == '1'
    
    def test_evaluate(self):
        assert ValueNode('a').evaluate() == 'a'
        assert ValueNode(1).evaluate() == 1
    
    def test_random_int(self):
        random.seed(42)
        vals = [
            -6352, -9181, -988, -1976, -2686, -5428, -6642, 7870, -7152, 9349,
            3825, -8959, -9024, -6930, -2836, -2377, 6559, 9726, -9131, 8390
        ]

        for expected in vals:
            node = ValueNode.random_int()
            assert node.evaluate() == expected


class TestVariableNode:
    @pytest.mark.parametrize('name', [ 'x', 'y', 'z' ])
    def test_str(self, name):
        assert str(VariableNode(name)) == name

    @pytest.mark.parametrize('name,val', [
        ('x', 1), ('y', 2), ('z', 3)
    ])
    def test_evaluate(self, name, val):
        node = VariableNode(name)
        assert node.evaluate({ name: val }) == val
    
    def test_single_random(self):
        random.seed(42)
        assert VariableNode.random_variables().name == 'u'
    
    def test_multiple_random(self):
        random.seed(42)
        names = [
            'u', 'd', 'a', 'i', 'h', 'v', 'e', 'y', 'r', 'c',
            'n', 'x', 'o', 'b', 's', 'l', 'w', 'p', 'k', 'f'
        ]
        
        nodes = VariableNode.random_variables(20)
        for node, name in zip(nodes, names):
            assert node.name == name

    def test_too_many_random(self):
        with pytest.raises(Exception):
            VariableNode.random_variables(28)



class TestSequenceNode:
    def test_simple_false(self):
        x_assign = AssignmentNode(X, ValueNode(1))
        y_assign = AssignmentNode(Y, ValueNode(2))
        compare = ComparisonNode(X, '>', Y)
        sequence = SequenceNode([x_assign, y_assign, compare])

        assert str(sequence) == 'x = 1\ny = 2\nx > y'
        assert not sequence.evaluate()
    
    def test_simple_true(self):
        x_assign = AssignmentNode(X, ValueNode(3))
        y_assign = AssignmentNode(Y, ValueNode(2))
        compare = ComparisonNode(X, '>', Y)
        sequence = SequenceNode([x_assign, y_assign, compare])

        assert str(sequence) == 'x = 3\ny = 2\nx > y'
        assert sequence.evaluate()
        
