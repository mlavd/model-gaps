from .ast import *
from dataclasses import dataclass
import random

class Generator:
    def generate(self):
        raise NotImplementedError()

    def generator(self):
        while True:
            yield self.generate()

    def random_comparison(self, values):
        a_var, b_var = random.sample(values, 2)
        return ComparisonNode.random(a_var, b_var)

    def random_calculation(self, variables):
        var = random.choice(variables)
        
        # Swap where the variable and ints appear
        a = var
        b = ValueNode.random_int()
        if random.random() < 0.5: b, a = a, b

        # Build the node
        calc_node = CalculationNode.random(a, b)
        return AssignmentNode(var, calc_node)


class Task1Generator(Generator):
    """Task 1 - Comparison of ints
    > Test performance of models on simple relational operations.
    ```c
    [number 1] > [number 2]
    ```
    """

    def generate(self):
        left = ValueNode.random_int()
        right = ValueNode.random_int()
        operator = ComparisonNode.random_operator(include_equality=True)
        
        if operator in ComparisonNode.EQUALITY and random.random() > 0.5:
            right = left

        return ComparisonNode(left, operator, right)


class Task2Generator(Generator):
    """Task 2 - Tracking Values
    > Test ability of models to track values across statements
    ```c
    x = [number 1]
    y = [number 2]
    x > y
    ``` 
    """
    def generate(self):
        x_var, y_var = VariableNode.random_variables(2)

        x_asn = AssignmentNode.random_int(x_var)
        y_asn = AssignmentNode.random_int(y_var)
        compare = self.random_comparison([ x_var, y_var ])

        assignments = [ x_asn, y_asn ]
        random.shuffle(assignments)

        extra = self.extend(x_var, y_var)

        sequence = SequenceNode(assignments + extra + [compare])
        return sequence
    
    def extend(self, x_var, y_var):
        return []

@dataclass
class Task3Generator(Task2Generator):
    """Test 3 - Perform calculations
    > Test ability of models to perform/understand arithmetic calculations
    ```c
    x = [number 1]
    y = [number 2]
    x = x + 1
    y = y + 100
    y = x * 1
    x > y
    ```
    """
    min_calcs: int = 1
    max_calcs: int = 5
    
    def extend(self, x_var, y_var):
        variables = [ x_var, y_var ]
        num_calcs = random.randint(self.min_calcs, self.max_calcs)

        return [
            self.random_calculation(variables)
            for _ in range(num_calcs)
        ]


class Task4Generator(Task2Generator):
    """Test 4 - Control Flow
    > Test if models can track values through conditional statements.

    ```c
    x = [number 1]
    y = [number 2]

    if (x > [number 3]) {
        x = y + 1
    } else {
        y *= 1000
    }
    x > y
    ```
    """
    def extend(self, x_var, y_var):
        condition = self.random_comparison([
            x_var, y_var, ValueNode.random_int()
        ])
        if_node = self.random_calculation([ x_var, y_var ])
        else_node = self.random_calculation([ x_var, y_var ])
        # if_node = SequenceNode([self.random_calculation([ x_var, y_var ])])
        # else_node = SequenceNode([self.random_calculation([ x_var, y_var ])])

        return [ ConditionalNode(condition, if_node, else_node) ]

class Task5Generator(Generator):
    """Test 5 - Control Flow no Calculations
    > Test if models can track values through conditional statements.

    ```c
    x = [number 1]
    y = [number 2]

    if (x > [number 3]) {
        x > y
    } else {
        y <= x
    }
    x > y
    ```
    """
    def generate(self):
        variables = VariableNode.random_variables(4)
        random.shuffle(variables)

        condition = self.random_comparison(random.choices(variables, k=2) + [
            ValueNode.random_int()
        ])

        if_node = self.random_comparison(random.choices(variables, k=2))
        else_node = self.random_comparison(random.choices(variables, k=2))
        
        conditional = ConditionalNode(condition, if_node, else_node)

        assignments = [
            AssignmentNode.random_int(var)
            for var in variables
        ]
        random.shuffle(assignments)

        return SequenceNode(assignments + [ conditional ])
    
class Task6Generator(Task2Generator):
    """Test 6 - Nested conditionals
    > Test if models can track values through multiple  conditional statements.

    ```c
    x = [number 1]
    y = [number 2]

    if (x > [number 3]) {
        x = y + 1
    } else {
        y *= 1000
    }
    x > y
    ```
    """
    DECAY = 0.25

    def random_calc_or_cond(self, x_var, y_var, nesting_weight):
        if random.random() <= nesting_weight:
            return self.random_condition(x_var, y_var, nesting_weight - self.DECAY)
        else:
            return self.random_calculation([ x_var, y_var ])

    def random_condition(self, x_var, y_var, nesting_weight=0.75):
        return ConditionalNode(
            condition = self.random_comparison([ x_var, y_var, ValueNode.random_int() ]),
            if_node = self.random_calc_or_cond(x_var, y_var, nesting_weight),
            else_node = self.random_calc_or_cond(x_var, y_var, nesting_weight),
        )

    def extend(self, x_var, y_var):
        return [ self.random_condition(x_var, y_var) ]
