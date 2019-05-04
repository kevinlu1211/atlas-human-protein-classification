import inspect
from copy import deepcopy

from .yaml_loader import Reference


def flatten_resources_dict(d):
    flattened = dict()
    for k, v in d.items():
        if v.get("properties") is not None:
            flattened.update(**{k: v})
        else:
            flattened.update(**flatten_resources_dict(v))
    return flattened


def find_references(resource):
    references = []
    if isinstance(resource, dict):
        for _, value in resource.items():
            references.extend(find_references(value))
    elif isinstance(resource, list):
        for value in resource:
            references.extend(find_references(value))
    elif isinstance(resource, Reference):
        references.append(resource)
    else:
        pass
    return references


def replace_arguments(graph, state_dict, node):
    try:
        arguments = node.arguments
        needs_state_dict = "state_dict" in inspect.signature(node.pointer).parameters
        if arguments is not None:
            reference_replaced_arguments = replace_references(graph, deepcopy(arguments))
            if needs_state_dict:
                reference_replaced_arguments["state_dict"] = state_dict
            return reference_replaced_arguments
    except AttributeError as e:
        return {}


def replace_should_run(graph, node):
    return replace_references(graph, deepcopy(node.should_run))


def replace_references(graph, arguments):
    if isinstance(arguments, dict):
        for name, argument in arguments.items():
            arguments[name] = replace_references(graph, argument)
    elif isinstance(arguments, list):
        for i, argument in enumerate(arguments):
            arguments[i] = replace_references(graph, argument)
    elif isinstance(arguments, Reference):
        # reassign name it make the intent clearer
        reference = arguments
        ref_node = graph.nodes(data=True)[reference.ref_node_name]["node"]
        ref_node_outputs = ref_node.output
        assert ref_node_outputs is not None, f"Node: {reference.ref_node_name} has no output"
        assert reference.output_name in ref_node_outputs, f"Node: {reference.output_name} has no output named {reference.output_name}"
        arguments = ref_node_outputs[reference.output_name]
    else:
        return arguments
    return arguments
