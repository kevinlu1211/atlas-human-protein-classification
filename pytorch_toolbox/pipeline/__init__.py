import copy
import types
import collections
import inspect
from functools import partial
import logging

import networkx as nx


def listify(x):
    if not isinstance(x, collections.Iterable):
        return [x]
    else:
        return x


class PipelineGraph:

    def __init__(self, graph, config):
        self.graph = graph
        self.config = config
        self.state_dict = {
            "config": copy.deepcopy(config)
        }

    def ___getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            return getattr(self.graph, item)

    @staticmethod
    def create_pipeline_graph_from_config(config):
        pipeline_graph = nx.DiGraph()

        for node_name, node_config in config.items():
            node_references = find_node_references(node_config)
            pipeline_graph.add_node(node_name, config=node_config, references=node_references)

        for node, attributes in pipeline_graph.nodes(data=True):
            refer_to_nodes = list(set(list(ref.keys())[0] for ref in attributes['references']))
            for refer_to_node in refer_to_nodes:
                pipeline_graph.add_edge(refer_to_node, node)
        return PipelineGraph(graph=pipeline_graph, config=config)

    def get_node(self, name):
        return self.graph.nodes(data=True)[name]

    @property
    def sorted_node_names(self):
        return list(nx.algorithms.dag.topological_sort(self.graph))

    def run(self, reference_lookup, to_node=None):
        sorted_node_names = self.sorted_node_names

        for name in sorted_node_names:
            logging.info(f"Currently running step: {name}")
            node = self.get_node(name)
            node_config = node['config']
            node_references = node.get('references')
            node_output = node_config.get('output')

            node_properties = node_config.get("properties")
            node_callable = reference_lookup[node_config['type']]
            #         print(name)
            #         print(node)
            if node_properties is not None:

                # 1. Replace the properties that reference values from another node with the actual values
                initialization_arguments = node_properties.get("initialization_arguments", dict())
                callable_arguments = node_properties.get("callable_arguments", dict())

                has_references = len(node_references) > 0
                if has_references:
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
                needs_state_dict = "state_dict" in inspect.signature(node_callable).parameters
                only_one_return_value = isinstance(node_output, str)
                if is_function:
                    # Make sure that if the callable is a function then there are no initialization arguments
                    assert len(
                        initialization_arguments) == 0, f"Function: {node_callable.__name__} cannot have initialization arguments: {initialization_arguments}, only callable arguments"

                    if partial_callable:
                        assert not len(node_output) == 1, 'If this is a partial callable, then there should be one output for this step'
                        if needs_state_dict:
                            node['output_lookup'] = {
                                node_output: partial(node_callable, **callable_arguments,
                                                     state_dict=self.state_dict)
                            }
                        else:
                            node['output_lookup'] = {
                                node_output: partial(node_callable, **callable_arguments)
                            }

                    else:
                        if needs_state_dict:
                            callable_output = node_callable(**callable_arguments, state_dict=self.state_dict)
                        else:
                            callable_output = node_callable(**callable_arguments)
                        if node_output is None:
                            continue
                        if not only_one_return_value:
                            node['output_lookup'] = {
                                output_name: output for output_name, output in
                                zip(node['output'], callable_output)
                            }
                        else:
                            node['output_lookup'] = {node_output: callable_output}
                else:
                    assert (partial_initialization and partial_callable) is False, "Can't make both the initialization of a class and it's __call__ method both partial"


                    # Now we check if we want to call a function that isn't that default __call__ method of the class
                    callable_function_name = node_properties.get("callable_function_name")

                    if partial_initialization:
                        node['output_lookup'] = {
                            node_output: partial(node_callable, **initialization_arguments)
                        }
                    else:
                        initialized_node_object = node_callable(**initialization_arguments)
                        if partial_callable:
                            if callable_function_name is None:
                                patch_call(initialized_node_object,
                                           partial(initialized_node_object.__call__, **callable_arguments))
                            else:
                                patch_call(initialized_node_object,
                                           partial(getattr(initialized_node_object, callable_function_name),
                                                   **callable_arguments))
                            node['output_lookup'] = {
                                node_output: initialized_node_object
                            }
                        else:
                            callable_output = node_callable()
                            if node_output is None:
                                continue
                            if not only_one_return_value:
                                node['output_lookup'] = {
                                    output_name: output for output_name, output in
                                    zip(node['output'], callable_output)
                                }
                            else:
                                node['output_lookup'] = {node['output']: callable_output}
            if to_node is not None and name == to_node:
                break

def find_references_from_arguments(arg_values):
    referenced_argument_values = []
    if isinstance(arg_values, dict) and arg_values.get("Ref") is not None:
        node_references = arg_values.get("Ref")
        if isinstance(node_references, list):
            for node_reference in node_references:
                referenced_argument_values.append(node_reference)
        else:
            referenced_argument_values.append(node_references)
    return referenced_argument_values


def find_node_references(node_config):
    dependencies = []
    node_properties = node_config.get("properties")
    if node_properties is None:
        pass
    else:
        initialization_dependencies = []
        initialization_arguments = node_properties.get("initialization_arguments", dict())
        assert isinstance(initialization_arguments,
                          dict), f"Please make sure the initialization arguments is of type dict for node {node_config}"
        for _, arg_values in initialization_arguments.items():
            initialization_dependencies.extend(find_references_from_arguments(arg_values))
        dependencies.extend(initialization_dependencies)

        callable_dependencies = []
        callable_arguments = node_properties.get("callable_arguments", dict())
        assert isinstance(callable_arguments,
                          dict), f"Please make sure the callable arguments is of type dict for node {node_config}"
        for _, arg_values in callable_arguments.items():
            callable_dependencies.extend(find_references_from_arguments(arg_values))
        dependencies.extend(callable_dependencies)

    return dependencies


# Reference from: https://stackoverflow.com/questions/38541015/how-to-monkey-patch-a-call-method
def patch_call(instance, func):
    class _(type(instance)):
        def __call__(self, *arg, **kwarg):
            return func(*arg, **kwarg)

    instance.__class__ = _


def replace_references(graph, arguments):
    for arg_name, arg_values in arguments.items():
        if isinstance(arg_values, dict) and arg_values.get("Ref") is not None:
            node_references = arg_values.get("Ref")
            if isinstance(node_references, list):
                referenced_argument_values = []
                for node_reference in node_references:
                    reference_node_name = list(node_reference.keys())[0]
                    reference_node_output_name = list(node_reference.values())[0]
                    try:
                        referenced_argument_values.append(
                            graph[reference_node_name]['output_lookup'][reference_node_output_name])
                    except KeyError as e:
                        print(f"KeyError: {e}")
                        print(f"Reference node name: {reference_node_name}")
                        print(f"Reference node output lookup: {graph[reference_node_name]['output_lookup']}")
                arguments[arg_name] = referenced_argument_values
            else:
                reference_node_name = list(node_references.keys())[0]
                reference_node_output_name = list(node_references.values())[0]
                try:
                    arguments[arg_name] = graph[reference_node_name]['output_lookup'][reference_node_output_name]
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(f"Reference node name: {reference_node_name}")
                    print(f"Reference node output lookup: {graph[reference_node_name]['output_lookup']}")
    return arguments

# def replace_references(graph, arguments):
#     for arg_name, arg_value in arguments.items():
#         if isinstance(arg_value, dict) and arg_value.get("Ref") is not None:
#             reference_node_name = list(arg_value.get("Ref").keys())[0]
#             reference_node_output_name = list(arg_value.get("Ref").values())[0]
#
#             try:
#                 arguments[arg_name] = graph[reference_node_name]['output_lookup'][reference_node_output_name]
#             except KeyError as e:
#                 print(f"KeyError: {e}")
#                 print(f"Reference node name: {reference_node_name}")
#                 print(f"Reference node output lookup: {graph[reference_node_name]['output_lookup']}")
#     return arguments
