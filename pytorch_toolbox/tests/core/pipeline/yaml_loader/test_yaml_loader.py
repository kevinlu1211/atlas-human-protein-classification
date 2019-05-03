from core.pipeline import load_config_from_string


def test_variable_replacement():
    document = """
    Variables:
      single_variable: hello
    Resources:
      TestVariableReplacement:
        single_variable: !Var single_variable
    """

    config = load_config_from_string(document, with_variable_replacement=True)

    assert config == {
        'Resources': {'TestVariableReplacement': {'single_variable': 'hello'}},
        'Variables': {'single_variable': 'hello'}
    }


def test_variable_in_group_replacement():
    document = """
    Variables:
      TestVariableInGroupReplacement:
        single_variable: hello
    Resources:
      TestVariableInGroupReplacement:
        single_variable: !Var TestVariableInGroupReplacement.single_variable
    """

    config = load_config_from_string(document, with_variable_replacement=True)

    assert config == {
        "Variables": {"TestVariableInGroupReplacement":
                          {"single_variable": "hello"}
                      },
        "Resources": {"TestVariableInGroupReplacement":
                          {"single_variable": "hello"}
                      }
    }


def test_variables_in_list_replacement():
    document = """
    Variables:
      variable_one: hello
      variable_two: bye
    Resources:
      TestVariableInListReplacement:
        - !Var variable_one
        - !Var variable_two
    """
    config = load_config_from_string(document, with_variable_replacement=True)
    assert config == {
        'Variables': {'variable_one': 'hello', 'variable_two': 'bye'},
        'Resources': {'TestVariableInListReplacement': ['hello', 'bye']}
    }


def test_variable_in_list_of_dict_replacement():
    document = """
    Variables:
      variable_one: hello
    Resources:
      TestVariableInListOfDictReplacement:
        - key1: !Var variable_one
    """
    config = load_config_from_string(document, with_variable_replacement=True)
    assert config == {
        'Variables': {'variable_one': 'hello'},
        'Resources': {'TestVariableInListOfDictReplacement': [{'key1': 'hello'}]}
    }


def test_variable_in_list_of_nested_dict_replacement():
    document = """
    Variables:
      variable_one: hello
    Resources:
      TestVariableInListOfNestedDictReplacement:
        - key1: 
            key2: !Var variable_one
    """
    config = load_config_from_string(document, with_variable_replacement=True)
    assert config == {
        'Variables': {'variable_one': 'hello'},
        'Resources': {'TestVariableInListOfNestedDictReplacement': [{'key1': {'key2': 'hello'}}]}
    }
