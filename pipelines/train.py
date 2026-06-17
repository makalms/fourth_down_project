"""
Main entry point for the application for training a model.
Application flow:
1. Load configuration from config.ini file using the config parser (name of the particular config must be specified as a command line argument when running the main.py file)
2. According to the specifications in the config, train the specified model using the specified data and parameters.
"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import parse_config
from dataloading import fetch_data
from models import Embedding_Model


def handle_argv():

    # Function that will be called to handle command line arguments. It should parse the command line arguments to get the name of the config to use and return it.
    parser = argparse.ArgumentParser(description='Train a model using the specified config.')
    parser.add_argument('--config', type=str, required=True, help='The name of the config to use (must be specified in config.ini)')
    args = parser.parse_args()
    return args.config

if __name__ == '__main__':

    # Parse command line arguments to get the name of the config to use
    config_name = handle_argv()

    # Load the config using the config parser
    config = parse_config(config_name)

    # According to the specifications in the config, train the specified model using the specified data and parameters.
    model_type = config.get('model_type')
    if model_type == 'embedding_model':
        model = Embedding_Model(config)
    # elif model_type == 'other_model_type':
        # model = other_model_type(config)
    else:
        raise ValueError(f"Model type {model_type} not recognized. Please specify a valid model type in the config.")

    # Fetch the data using the dataloader according to the specifications in the config
    X, A, Y = fetch_data(config)

    # Train the model using the fetched data
    model.train(X, A, Y)