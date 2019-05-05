import networkx as nx
from pytorch_toolbox.pipeline import Node, Reference, flatten_resources_dict, find_references, replace_arguments, \
    replace_references


def test_flatten_resources_dict():
    """
    flatten_resource_dict returns all the dictionaries that are one level above a dictionary that contains
    a key with "properties"
    """
    dictionary = {
        "Resources": {
            "Group_A": {
                "SubGroup_A": {
                    "Resource_A1": {
                        "properties": "Group_A.SubGroup_A.Resource_A1"
                    },
                    "Resource_A2": {
                        "properties": {
                            "arguments": {
                                "arg1": "Group_A.SubGroup_A.Resource_A2"
                            }
                        }
                    }
                }
            },
            "Group_B": {
                "Resource_B": {
                    "properties": "Group_B.Resource_B"
                }
            },
            "Resource_C": {
                "properties": "Resource_C"
            }
        }
    }

    flattened_resources = flatten_resources_dict(dictionary)

    assert flattened_resources == {
        "Resource_A1": {
            "properties": "Group_A.SubGroup_A.Resource_A1"
        },
        "Resource_A2": {
            "properties": {
                "arguments": {
                    "arg1": "Group_A.SubGroup_A.Resource_A2"
                }
            }
        },
        "Resource_B": {
            "properties": "Group_B.Resource_B"
        },
        "Resource_C": {
            "properties": "Resource_C"
        }
    }


def test_find_references():
    dictionary = {
        "Resources": {
            "Group_A": {
                "SubGroup_A": {
                    "Resource_A1": {
                        "properties": Reference(ref_node_name="A1", output_name="A1")
                    },
                    "Resource_A2": {
                        "properties": {
                            "arguments": {
                                "arg1": Reference(ref_node_name="A2", output_name="A2")
                            }
                        }
                    }
                }
            },
            "Group_B": {
                "Resource_B": {
                    "properties": Reference(ref_node_name="B", output_name="B")
                }
            },
            "Resource_C": {
                "properties": Reference(ref_node_name="C", output_name="C")
            }
        }
    }

    references = sorted(find_references(dictionary), key=lambda x: x.ref_node_name)

    assert references == [
        Reference(ref_node_name="A1", output_name="A1"),
        Reference(ref_node_name="A2", output_name="A2"),
        Reference(ref_node_name="B", output_name="B"),
        Reference(ref_node_name="C", output_name="C")
    ]


def test_replace_arguments():
    graph, node_a, _ = _setup_graph_two_dependent_nodes()
    node_a.create_output()
    reference_replaced_arguments = replace_arguments(graph, state_dict={}, node=node_b)
    assert reference_replaced_arguments == {
        "x": 1
    }


# This is kind of bad as we are testing two things, refactor later
def test_replace_arguments_and_run_output():
    graph, node_a, node_b = _setup_graph_two_dependent_nodes()
    node_a.create_output()
    reference_replaced_arguments = replace_arguments(graph, state_dict={}, node=node_b)
    node_b.reference_replaced_arguments.update(**reference_replaced_arguments)
    node_b.create_output()
    assert node_b.output == {"output_1": 2}


def _setup_graph_two_dependent_nodes():
    pointer_a = lambda: 1
    pointer_b = lambda x: x + 1
    node_a = Node(name="A",
                  references=[],
                  pointer=pointer_a,
                  partial=False,
                  arguments={},
                  output_names=["output_1"],
                  should_run=True)
    node_b = Node(name="B",
                  references=[Reference(ref_node_name="A", output_name="output_1")],
                  pointer=pointer_b,
                  partial=False,
                  arguments={"x": Reference(ref_node_name="A", output_name="output_1")},
                  output_names=["output_1"],
                  should_run=True)
    graph = nx.DiGraph()
    graph.add_node("A", node=node_a)
    graph.add_node("B", node=node_b)
    return graph, node_a, node_b


def test_replace_references():
    pointer_a = lambda: (1, 2)
    node_a = Node(name="A", references=[], pointer=pointer_a, partial=False, arguments={},
                  output_names=["output_1", "output_2"], should_run=True)
    arguments_with_reference_to_node_a = {
        "arg1": Reference(ref_node_name="A", output_name="output_1"),
        "arg2": Reference(ref_node_name="A", output_name="output_2")
    }
    graph = nx.DiGraph()
    graph.add_node("A", node=node_a)
    node_a.create_output()

    arguments_with_references_replaced = replace_references(graph, arguments_with_reference_to_node_a)

    assert arguments_with_references_replaced == {
        "arg1": 1,
        "arg2": 2
    }
