import logging
import functools
from copy import deepcopy
import networkx as nx

from .yaml_loader import load_config_from_path
from .graph_construction import flatten_dict, find_references, load_properties_with_default_values, replace_arguments


class Pipeline:
    def __init__(self, graph, config, state_dict={}):
        self.graph = graph
        self.config = config
        self.state_dict = {
            **state_dict
        }

    @classmethod
    def create_from_config(cls, config, lookups, state_dict={}):
        assert config.get("Resources") is not None, "There is no Resources key in the configuration file"
        graph = nx.DiGraph()
        flattened_resources = flatten_dict(config["Resources"])
        graph = cls._add_nodes_to_graph(graph, flattened_resources, lookups)
        graph = cls._add_edges_to_graph(graph)
        return cls(graph, config, state_dict)

    @classmethod
    def create_from_config_path(cls, config_path, lookups):
        raw_config = load_config_from_path(config_path)
        replaced_config = load_config_from_path(config_path, with_variable_replacement=True)
        return cls.create_from_config(replaced_config, lookups, state_dict=dict(raw_config=raw_config))

    @staticmethod
    def _add_nodes_to_graph(graph, resources, lookups):
        for name, resource in resources.items():
            references = find_references(resource)
            properties = load_properties_with_default_values(resource["properties"], lookups)
            node = Node(name=name, references=references, **properties)
            graph.add_node(name, node=node)
        return graph

    @staticmethod
    def _add_edges_to_graph(graph):
        for name, node_wrapper in graph.nodes(data=True):
            node = node_wrapper["node"]
            for reference in node.references:
                referenced_node_name = reference.ref_node_name
                assert referenced_node_name in graph.nodes, f"The reference: {referenced_node_name} in node: {node.name} does not exist"
                graph.add_edge(referenced_node_name, name)
        return graph

    @property
    def sorted_node_names(self):
        return list(nx.algorithms.dag.topological_sort(self.graph))

    def get_node(self, name):
        return self.graph.nodes(data=True)[name]["node"]

    def get_node_output_lookup(self, name):
        return self.get_node(name).output

    def get_node_output(self, name, output_lookup_key=None):
        output_lookup = self.get_node_output_lookup(name)
        if output_lookup is None:
            return None

        if output_lookup_key is None and len(output_lookup) == 1:
            output = list(output_lookup.values())[0]
            return output
        else:
            return output_lookup[output_lookup_key]

    def run(self, to_node=None):
        for node_name in nx.algorithms.dag.topological_sort(self.graph):
            if to_node == node_name:
                break
            node = self.graph.nodes(data=True)[node_name]["node"]
            self._run_node(node)

    def _run_node(self, node):
        replace_arguments(self.graph, self.state_dict, node)
        node.create_output()


class Node:
    def __init__(self, name, references, pointer, partial, arguments, output_names):
        self.name = name
        self.references = references
        self.pointer = pointer
        self.arguments = arguments
        self.output_names = output_names
        self.partial = partial
        self.reference_replaced_arguments = None
        self.output = None

    def create_output(self):
        if self.output_names is None:
            logging.warning(f"Node: {self.name} has no output")
        else:
            assert not isinstance(self.output_names, str), f"The output_names for node: {self.name} can't be a string, only a list"
        if self.partial:
            assert len(
                self.output_names) == 1, "If the output of node: {self.name} is partial, then there should be one output, {len(self.output_names)} outputs are found"
            self.output = {self.output_names[0]: functools.partial(self.pointer, **self.reference_replaced_arguments)}
        else:
            output = self.pointer(**self.reference_replaced_arguments)
            iterable_output = [output] if len(self.output_names) == 1 else output
            self.output = {output_name: output_value for output_name, output_value in
                           zip(self.output_names, iterable_output)}

