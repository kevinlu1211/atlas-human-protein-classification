import yaml
from copy import deepcopy
from dataclasses import dataclass

class PyTorchToolboxLoader(yaml.SafeLoader):
    pass


def var_constructor(loader, node):
    return Variable(name=node.value)


@dataclass
class Variable:
    name: str


def ref_constructor(loader, node):
    ref_node_name_and_output_name = node.value.split(".")
    assert len(ref_node_name_and_output_name) == 2
    ref_node_name, output_name = ref_node_name_and_output_name
    return Reference(ref_node_name=ref_node_name, output_name=output_name)


@dataclass
class Reference:
    ref_node_name: str
    output_name: str


def replace_config_variables(config, resource_key="Resources", variable_key="Variables"):
    config = deepcopy(config)
    try:
        replaced_resources = replace_variables(config[resource_key], config[variable_key])
        config[resource_key] = replaced_resources
    except KeyError:
        return config
    return config


def replace_variables(resources, variables):
    if isinstance(resources, dict):
        for name, resource in resources.items():
            resources[name] = replace_variables(resource, variables)
    elif isinstance(resources, list):
        for i, resource in enumerate(resources):
            resources[i] = replace_variables(resource, variables)
    elif isinstance(resources, Variable):
        resources = variables[resources.name]
    else:
        return resources
    return resources


# This tells the loader that when it sees "!Var" it will pass the value proceeding the !path value into the var constructor
PyTorchToolboxLoader.add_constructor('!Var', var_constructor)
# This tells the loader that when it sees "!Ref" it will pass the value proceeding the !path value into the var constructor
PyTorchToolboxLoader.add_constructor('!Ref', ref_constructor)