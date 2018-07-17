import re


def load_parallel_state_dict(model, states, prefix=r'module\.'):
    """
    safely load model state

    Since nn.Dataparallel module has name with 'module' as prefix,
    parse that prefix
    """
    modified_dict = {}
    for key, value in states.items():
        modified_dict[re.sub(prefix, '', key)] = value
    model.load_state_dict(modified_dict)
