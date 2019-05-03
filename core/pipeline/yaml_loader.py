from pathlib import Path
import yaml
from copy import deepcopy
from dataclasses import dataclass

RESOURCE_KEY = "Resources"
VARIABLE_KEY = "Variables"


class PyTorchToolboxLoader(yaml.SafeLoader):
    pass


class PyTorchToolboxDumper(yaml.SafeDumper):
    pass


@dataclass
class Variable:
    node_name: str
    variable_name: str

    @property
    def is_nested(self):
        return self.node_name is not None


def var_constructor(loader, node):
    if "." in node.value:
        node_name_and_variable_name = node.value.split(".")
        assert len(node_name_and_variable_name) == 2
        node_name, variable_name = node_name_and_variable_name
        return Variable(node_name=node_name, variable_name=variable_name)
    else:
        return Variable(node_name=None, variable_name=node.value)


def var_representer(dumper, data):
    if data.is_nested:
        value = f"{data.node_name}.{data.variable_name}"
    else:
        value = f"{data.variable_name}"
    return dumper.represent_scalar('!Var', f"{value}")


@dataclass
class Reference:
    ref_node_name: str
    output_name: str


def ref_constructor(loader, node):
    ref_node_name_and_output_name = node.value.split(".")
    assert len(ref_node_name_and_output_name) == 2
    ref_node_name, output_name = ref_node_name_and_output_name
    return Reference(ref_node_name=ref_node_name, output_name=output_name)


def ref_representer(dumper, data):
    value = f"{data.ref_node_name}.{data.output_name}"
    return dumper.represent_scalar('!Ref', f"{value}")


def replace_config_variables(config, resource_key, variable_key):
    config = deepcopy(config)
    replaced_resources = replace_variables(config[resource_key], config[variable_key])
    config[resource_key] = replaced_resources
    return config


def replace_variables(resources, variables):
    if isinstance(resources, dict):
        for name, resource in resources.items():
            resources[name] = replace_variables(resource, variables)
    elif isinstance(resources, list):
        for i, resource in enumerate(resources):
            resources[i] = replace_variables(resource, variables)
    elif isinstance(resources, Variable):
        # name reassignment to make intent clearer that this resource is a variable
        if resources.is_nested:
            resources = nested_variable_replacement(resources, variables)
        else:
            resources = variables[resources.variable_name]
    else:
        return resources
    return resources


def nested_variable_replacement(resource, variables):
    return variables[resource.node_name][resource.variable_name]


PyTorchToolboxLoader.add_constructor('!Var', var_constructor)
PyTorchToolboxDumper.add_representer(Variable, var_representer)

PyTorchToolboxLoader.add_constructor('!Ref', ref_constructor)
PyTorchToolboxDumper.add_representer(Reference, ref_representer)


def load_config_from_string(config_string, with_variable_replacement=False):
    config = yaml.load(config_string, Loader=PyTorchToolboxLoader)
    if with_variable_replacement:
        config = replace_config_variables(config, resource_key=RESOURCE_KEY, variable_key=VARIABLE_KEY)
    return config


def load_config_from_path(path, with_variable_replacement=False):
    with Path(path).open("r") as f:
        config = load_config_from_string(f, with_variable_replacement)
    return config


def dump_config_to_string(config):
    return yaml.dump(config, Dumper=PyTorchToolboxDumper, default_flow_style=False)


def dump_config_to_path(config, save_path):
    with save_path.open('w') as yaml_file:
        yaml.dump(config, yaml_file, Dumper=PyTorchToolboxDumper, default_flow_style=False)
