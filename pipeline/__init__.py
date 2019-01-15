import types
import collections
from functools import partial

import networkx as nx


def listify(x):
    if not isinstance(x, collections.Iterable):
        return [x]
    else:
        return x

class PipelineGraph:

    def __init__(self, graph):
        self.graph = graph

    def ___getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            return getattr(self.graph, item)

    @staticmethod
    def create_pipeline_graph_from_config(config):
        pipeline_graph = nx.DiGraph()

        for node_name, node_config in config.items():
            node_dependencies = find_node_dependencies(node_config)
            pipeline_graph.add_node(node_name, config=node_config, dependencies=node_dependencies)

        for node, attributes in pipeline_graph.nodes(data=True):
            dependent_on_nodes = list(set(list(dep.keys())[0] for dep in attributes['dependencies']))
            for dependent_on_node in dependent_on_nodes:
                pipeline_graph.add_edge(dependent_on_node, node)
        return PipelineGraph(graph=pipeline_graph)

    def run_graph(self, reference_lookup, force_run=False):
        sorted_node_names = list(nx.algorithms.dag.topological_sort(self.graph))

        for name in sorted_node_names:
            node = self.graph.nodes(data=True)[name]
            node_output_lookup = node.get('output_lookup')
            if not force_run and node_output_lookup is not None:
                continue
            node_config = node['config']
            node_dependencies = node.get('dependencies')
            node_output = node_config.get('output', [])

            node_properties = node_config.get("properties")
            node_callable = reference_lookup[node_config['type']]
            #         print(name)
            #         print(node)
            if node_properties is not None:

                # 1. Replace the properties that reference values from another node with the actual values
                initialization_arguments = node_properties.get("initialization_arguments", dict())
                callable_arguments = node_properties.get("callable_arguments", dict())

                has_dependencies = len(node_dependencies) > 0
                if has_dependencies:
                    # replace the reference values in the arguments
                    initialization_arguments = replace_references(self.graph.nodes(data=True), initialization_arguments)
                    callable_arguments = replace_references(self.graph.nodes(data=True), callable_arguments)
                else:
                    pass

                # 2. Check if the callable of the node is a function or a class
                is_function = isinstance(node_callable, types.FunctionType)
                partial_initialization = node_properties.get("partial_initialization", False)
                partial_callable = node_properties.get("partial_callable", False)

                # 3. Initialize the callables accordingly
                node['callable'] = node_callable
                node['output'] = node_output
                if is_function:
                    # Make sure that if the callable is a function then there are no initialization arguments
                    assert len(
                        initialization_arguments) == 0, f"Function: {node_callable.__name__} cannot have initialization arguments: {initialization_arguments}, only callable arguments"

                    if partial_callable:
                        assert len(
                            node_output) <= 1, 'If this is a partial callable, then there should be one or less output for this step'
                        if len(node_output) == 0:
                            pass
                        output_name = node_output[0]
                        node['output_lookup'] = {
                            output_name: partial(node_callable, **callable_arguments)
                        }
                    else:
                        callable_output = listify(node_callable(**callable_arguments))
                        node['output_lookup'] = {
                            output_name: callable_output for output_name, callable_output in
                            zip(node['output'], callable_output)
                        }
                else:
                    assert (
                                   partial_initialization and partial_callable) is False, "Can't make both the initialization of a class and it's __call__ method both partial"
                    assert len(
                        node_output) == 1, 'If this is a step to initialize an object, then there should only be one output, the object itself'
                    output_name = node_output[0]
                    if partial_initialization:
                        node['output_lookup'] = {
                            output_name: partial(node_callable, **initialization_arguments)
                        }
                    else:
                        initialized_node_object = node_callable(**initialization_arguments)
                        if partial_callable:
                            patch_call(initialized_node_object,
                                       partial(initialized_node_object.__call__, **callable_arguments))
                        node['output_lookup'] = {
                            output_name: initialized_node_object
                        }
            else:
                node['callable'] = node_callable
                callable_output = listify(node_callable())
                node['output_lookup'] = {
                    output_name: callable_output for output_name, callable_output in zip(node['output'], callable_output)
                }

def find_node_dependencies(node_config):
    dependencies = []
    node_properties = node_config.get("properties")
    if node_properties is None:
        pass
    else:
        initialization_dependencies = []
        initialization_arguments = node_properties.get("initialization_arguments", dict())
        assert isinstance(initialization_arguments,
                          dict), f"Please make sure the initialization arguments is of type dict for node {node_config}"
        for arg_name, arg_value in initialization_arguments.items():
            if isinstance(arg_value, dict) and arg_value.get("Ref") is not None:
                initialization_dependencies.append(arg_value.get("Ref"))
        dependencies.extend(initialization_dependencies)

        callable_dependencies = []
        callable_arguments = node_properties.get("callable_arguments", dict())
        assert isinstance(callable_arguments,
                          dict), f"Please make sure the callable arguments is of type dict for node {node_config}"
        for arg_name, arg_value in callable_arguments.items():
            if isinstance(arg_value, dict) and arg_value.get("Ref") is not None:
                callable_dependencies.append(arg_value.get("Ref"))
        dependencies.extend(callable_dependencies)

    return dependencies


# Reference from: https://stackoverflow.com/questions/38541015/how-to-monkey-patch-a-call-method
def patch_call(instance, func):
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
            return func(*arg, **kwarg)

    instance.__class__ = _


def replace_references(graph, arguments):
    for arg_name, arg_value in arguments.items():
        if isinstance(arg_value, dict) and arg_value.get("Ref") is not None:
            reference_node_name = list(arg_value.get("Ref").keys())[0]
            reference_node_output_name = list(arg_value.get("Ref").values())[0]

            try:
                arguments[arg_name] = graph[reference_node_name]['output_lookup'][reference_node_output_name]
            except KeyError as e:
                print(f"KeyError: {e}")
                print(f"Reference node name: {reference_node_name}")
                print(f"Reference node output lookup: {graph[reference_node_name]['output_lookup']}")
    return arguments



