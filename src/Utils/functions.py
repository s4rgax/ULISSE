import yaml
import torch

def validate_execution_config(config):
    run_train = config['execution']['phases'].get('train', False)
    run_test = config['execution']['phases'].get('test', False)
    run_explainer = config['execution']['phases'].get('explain', False)

    if not run_train and not run_test and not run_explainer:
        raise ValueError("No execution phase selected. Enable at least one phase (train or test)")
    return run_train, run_test, run_explainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_model_config(config, model_name):
    if 'models' not in config:
        raise ValueError("No 'models' section found in config file")
    if model_name not in config['models']:
        available_models = list(config['models'].keys())
        raise ValueError(f"Model '{model_name}' not found in config. Available models: {available_models}")
    return config['models'][model_name]


def compute_metrics_from_conf_matrix(conf_matrix):
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TN = conf_matrix[0, 0]
    return TP, FP, FN, TN


def compute_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def transform_batch_positions(batch):
    positions = []
    for i in range(len(batch[0])):
        positions.append((
            batch[0][i].item(),
            batch[1][i].item()
        ))
    return positions

def get_safe_from_dict (elem: dict, attr_name: str, default_value=None):
    """
    Get a value from a dictionary safely, returning a default value if the key does not exist.
    """
    if attr_name in elem:
        return elem[attr_name]
    else:
        return default_value