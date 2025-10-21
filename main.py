import argparse

from src.Train.pipeline import Runner
from src.Utils.functions import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Training pipeline for semantic segmentation')
    parser.add_argument('--data_config', type=str, required=True, help='Path to data config file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--model', type=str, required=True, help='Model name from config (e.g., resnet50)')
    parser.add_argument('--additional_tag', type=str, required=False, help='Additional tag for the model name', default='')
    return parser.parse_args()


def run(data_config_path, model_config_path, model_name, additional_tag):
    data_config = load_config(data_config_path)
    model_config = load_config(model_config_path)
    exec = Runner(data_config, model_config, model_name, additional_tag)
    exec.run()



# main.py
if __name__ == "__main__":
    args = parse_args()
    run(args.data_config, args.model_config, args.model, args.additional_tag)