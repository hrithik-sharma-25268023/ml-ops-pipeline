"""read parameters from yaml"""

import yaml

def load_params(yaml_path: str) -> dict:
    """reads the parameters from yaml"""

    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)
    return params
