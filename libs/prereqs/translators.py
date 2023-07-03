from .ast import *
from collections import defaultdict, namedtuple
from sklearn.preprocessing import OneHotEncoder
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd
# import spektral
import string 


class Translator:
    def translate(self, node:Node) -> Any:
        return self._translate(node)

    def _translate(self, node:Node, **kwargs) -> Any:
        if node.TYPE == 'assignment': return self._assignment(node, **kwargs)
        if node.TYPE == 'calculation': return self._calculation(node, **kwargs)
        if node.TYPE == 'comparison': return self._comparison(node, **kwargs)
        if node.TYPE == 'conditional': return self._conditional(node, **kwargs)
        if node.TYPE == 'operator': return self._operator(node, **kwargs)
        if node.TYPE == 'sequence': return self._sequence_node(node, **kwargs)
        if node.TYPE == 'value': return self._value_node(node, **kwargs)
        if node.TYPE == 'variable': return self._variable_node(node, **kwargs)
        raise NotImplementedError(f'Invalid type: {type(node)}')

    def _calculation(self, node:CalculationNode, **kwargs) -> Any:
        return self._operator(node, **kwargs)
    
    def _comparison(self, node:ComparisonNode, **kwargs) -> Any:
        return self._operator(node, **kwargs)

    def _assignment(self, node:AssignmentNode) -> Any:
        raise NotImplementedError('_assignment')        
    
    def _conditional(self, node:ConditionalNode) -> Any:
        raise NotImplementedError('_conditional')

    def _operator(self, node:OperatorNode) -> Any:
        raise NotImplementedError('_operator')
    
    def _sequence_node(self, node:SequenceNode) -> Any:
        raise NotImplementedError('_value_node')
    
    def _value_node(self, node:ValueNode) -> Any:
        raise NotImplementedError('_value_node')
    
    def _variable_node(self, node:VariableNode) -> Any:
        raise NotImplementedError('_variable_node')


class BasicCTranslator(Translator):
    def __init__(self):
        self.depth = 0

    def _assignment(self, node:AssignmentNode) -> Any:
        return f'{node.variable} = {node.value};'
    
    def _conditional(self, node:ConditionalNode) -> Any:
        indent = '\t' * self.depth
        self.depth += 1
        r = f'if ({self._translate(node.condition)}) {{\n'
        r += f'{indent}\t{self._translate(node.if_node)}\n'
        r += f'{indent}}} else {{\n'
        r += f'{indent}\t{self._translate(node.else_node)}\n'
        r += f'{indent}}}'
        self.depth -= 1
        return r

    def _operator(self, node:OperatorNode) -> Any:
        l_val = self._translate(node.left)
        r_val = self._translate(node.right)
        return f'{l_val} {node.operator} {r_val}'
    
    def _sequence_node(self, node:SequenceNode) -> Any:
        return '\n'.join(self._translate(n) for n in node.nodes)
    
    def _value_node(self, node:ValueNode) -> Any:
        return str(node.value)
    
    def _variable_node(self, node:VariableNode) -> Any:
        return node.name


class PythonTranslator(Translator):
    def _assignment(self, node:AssignmentNode) -> Any:
        return f'{node.variable} = {node.value}'
    
    def _conditional(self, node:ConditionalNode) -> Any:
        r = f'if {self._translate(node.condition)}:\n'
        r += f'\t{self._translate(node.if_node)}\n'
        r += 'else:\n'
        r += f'\t{self._translate(node.else_node)}'
        return r

    def _operator(self, node:OperatorNode) -> Any:
        l_val = self._translate(node.left)
        r_val = self._translate(node.right)
        return f'{l_val} {node.operator} {r_val}'
    
    def _sequence_node(self, node:SequenceNode) -> Any:
        return '\n'.join(self._translate(n) for n in node.nodes)
    
    def _value_node(self, node:ValueNode) -> Any:
        return str(node.value)
    
    def _variable_node(self, node:VariableNode) -> Any:
        return node.name


class BasicFeatureTranslator(Translator):
    IDS = [
            '<empty>', '<', '<=', '>', '>=', '==', '!=',
            '=',
            '+', '-', '*', '/', '%',
            'if', 'do', 'else', 'end',
    ]
    VARS = [ '<empty>' ] + list(string.ascii_lowercase)
    OPERATOR = 0
    NUMBER = 1
    VARIABLE = 2

    def __init__(self):
        self.type_encoder = OneHotEncoder()
        self.type_encoder.fit([[self.OPERATOR], [self.NUMBER], [self.VARIABLE]])
        self.op_encoder = OneHotEncoder()
        self.op_encoder.fit([[x] for x in self.IDS])
        self.var_encoder = OneHotEncoder()
        self.var_encoder.fit([[v] for v in self.VARS])

    def _get_operator(self, op):
        return [[ self.OPERATOR, op, '<empty>', 0 ]]

    def _assignment(self, node:AssignmentNode) -> Any:
        return self._translate(node.variable) \
            + self._get_operator('=') \
            + self._translate(node.value)
    
    def _conditional(self, node:ConditionalNode) -> Any:
        return self._get_operator('if') \
            + self._translate(node.condition) \
            + self._get_operator('do') \
            + self._translate(node.if_node) \
            + self._get_operator('else') \
            + self._translate(node.else_node) \
            + self._get_operator('end')

    def _operator(self, node:OperatorNode) -> Any:
        return self._translate(node.left) \
            + self._get_operator(node.operator) \
            + self._translate(node.right)
    
    def _sequence_node(self, node:SequenceNode) -> Any:
        feats = []
        for child in node.nodes:
            feats += self._translate(child)

        return feats
    
    def _value_node(self, node:ValueNode) -> Any:
        return [[ self.NUMBER, '<empty>', '<empty>', node.value ]]
    
    def _variable_node(self, node:VariableNode) -> Any:
        return [[ self.VARIABLE, '<empty>', node.name, 0 ]]
    
    def normalize(self, feats):
        feats = np.array(feats)
        # feats[feats[:, 0] == self.NUMBER, 1] /= 10_000
        # feats[feats[:, 0] == self.NUMBER, 1] += 0.5
        # feats[feats[:, 0] == self.OPERATOR, 1] /= len(self.IDS)
        # feats[feats[:, 0] == self.VARIABLE, 1] /= len(self.VARS)
        types = self.type_encoder.transform(feats[:, 0:1].astype(int))
        ops   = self.op_encoder.transform(feats[:, 1:2])
        ids   = self.var_encoder.transform(feats[:, 2:3])
        nums = feats[:, 3].astype(int) / 10_000

        return np.hstack([types.A, ops.A, ids.A, nums[:, None] ])



Nodes = namedtuple('Nodes', ['ast', 'cfg'])

class NetworkXTranslator(Translator):
    def translate(self, node):
        self.G = nx.DiGraph()
        self.node = 0
        self.last_refs = defaultdict(lambda: None)
        self._translate(node)
        return self.G

    def add_node(self, label, **kwargs):
        self.node += 1
        self.G.add_node(self.node, label=label, **kwargs)
        return self.node
    
    def add_cfg(self, nids, nodes, **kwargs):
        for nid in nids:
            for cfg in nodes.cfg:
                self.G.add_edge(nid, cfg, label='CFG', **kwargs)
    
    def add_ast(self, nid, nodes, **kwargs):
        for ast in nodes.ast:
            self.G.add_edge(nid, ast, label='AST', **kwargs)
    
    def add_cdf(self, source, target, **kwargs):
        self.G.add_edge(source, target, label='CDG', **kwargs)

    def _assignment(self, node:AssignmentNode, **kwargs) -> Any:
        assign = self.add_node('ASSIGN', **kwargs)
        variable = self._translate(node.variable, order=0, assign=True)
        value = self._translate(node.value, order=1)

        self.last_refs[node.variable.name] = assign

        self.add_ast(assign, variable)
        self.add_ast(assign, value)
        return Nodes(ast=[assign], cfg=[assign])
        # return f'{node.variable} = {node.value}'
    
    def _conditional(self, node:ConditionalNode, **kwargs) -> Any:
        if_id = self.add_node('IFELSE')
        cond_id = self._translate(node.condition)
        if_node = self._translate(node.if_node)
        else_node = self._translate(node.else_node)

        self.add_ast(if_id, cond_id, value='CONDITION')
        self.add_ast(if_id, if_node, value='IF')
        self.add_ast(if_id, else_node, value='ELSE')

        self.add_cdf(if_node.cfg[0], cond_id.ast[0], value='true')
        self.add_cdf(else_node.cfg[0], cond_id.ast[0], value='false')

        return Nodes(ast=[if_id], cfg=if_node.cfg + else_node.cfg)

    def _operator(self, node:OperatorNode, **kwargs) -> Any:
        op_node = self.add_node('OPERATOR', type=node.operator)
        l_node = self._translate(node.left, order=0)
        r_node = self._translate(node.right, order=1)

        self.add_ast(op_node, l_node)
        self.add_ast(op_node, r_node)
        return Nodes(ast=[op_node], cfg=[op_node])
    
    def _sequence_node(self, node:SequenceNode) -> Any:
        block = self.add_node('BLOCK')
        last_cfg = []
        for i, node in enumerate(node.nodes):
            statement = self._translate(node, order=i)
            self.add_ast(block, statement)
            self.add_cfg(last_cfg, statement)
            last_cfg = statement.cfg
        return Nodes(ast=[block], cfg=last_cfg)
    
    def _value_node(self, node:ValueNode, **kwargs) -> Any:
        nid = self.add_node('VALUE', value=node.value, **kwargs)
        return Nodes(ast=[nid], cfg=[])
    
    def _variable_node(self, node:VariableNode, **kwargs) -> Any:
        is_assign = kwargs.pop('assign', False)

        nid = self.add_node('VARIABLE', nam=node.name, **kwargs)

        last_ref = self.last_refs[node.name]
        if last_ref and not is_assign:
            self.add_cdf(nid, last_ref)

        return Nodes(ast=[nid], cfg=[])

# class SpektralTranslator(NetworkXTranslator):
#     NODE_LABELS = { k:v for v, k in enumerate([
#         'ASSIGN', 'BLOCK', 'IFELSE', 'OPERATOR', 'VARIABLE',
#         'VALUE'
#     ])}

#     EDGE_LABELS = { 'AST': 0, 'CFG': 1, 'CDG': 2 }
#     VALUES = { 'CONDITION': 0, 'IF': 1, 'ELSE': 2, 'true': 3, 'false': 4 }

#     IDS = { k: v for v, k in enumerate([
#             '<empty>', '<', '<=', '>', '>=', '==', '!=',
#             '=',
#             '+', '-', '*', '/', '%',
#             'if', 'do', 'else', 'end',
#         ])
#     }

#     VARS = { k: v for v, k in enumerate(string.ascii_lowercase)}

#     def map(self, df, name, labels, has_na=False):
#         if name not in df.columns: return

#         seq = df[name].map(labels.get)

#         if has_na:
#             df[name] = (seq + 1).fillna(0) / len(labels)
#         else:
#             df[name] = seq / (len(labels) - 1)

#     def translate(self, node):
#         g = super().translate(node)

#         df_edges = nx.to_pandas_edgelist(g)
#         self.map(df_edges, 'label', SpektralTranslator.EDGE_LABELS)
#         self.map(df_edges, 'value', SpektralTranslator.VALUES, True)

#         df_nodes = pd.DataFrame.from_dict(g.nodes, orient='index')
#         self.map(df_nodes, 'label', SpektralTranslator.NODE_LABELS)
#         self.map(df_nodes, 'nam', SpektralTranslator.VARS, True)
#         self.map(df_nodes, 'type', SpektralTranslator.IDS, True)
#         df_nodes.value = (df_nodes.value / 10_000).fillna(-2)

#         df_nodes.order += 1
#         df_nodes.order = df_nodes.order.fillna(0) / df_nodes.order.max()

#         return spektral.data.Graph(
#             a = nx.adjacency_matrix(g), # Adjacency matrix
#             x = df_nodes.values, # Node features
#             e = df_edges.values[:, 2:], # Edge features
#             y = [int(node.evaluate())], # Labels
#         )
