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
from models import BaseLSTM
from models import BaseCNN


def handle_argv():

    # Function that will be called to handle command line arguments. It should parse the command line arguments to get the name of the config to use and return it.
    parser = argparse.ArgumentParser(description='Train a model using the specified config.')
    parser.add_argument('--config', type=str, required=True, help='The name of the config to use (must be specified in config.ini)')
    parser.add_argument('--name', type=str, required=True, help='The name to give to the trained model (this will be used to save the model and its parameters)')
    args = parser.parse_args()
    return args.config, args.name

if __name__ == '__main__':

    # Parse command line arguments to get the name of the config to use
    config_name, model_name = handle_argv()

    # Load the config using the config parser
    config = parse_config(config_name)

    # According to the specifications in the config, train the specified model using the specified data and parameters.
    model_type = config.get('model_type')
    if model_type == 'embedding_model':
        model = Embedding_Model(config)
    elif model_type == 'lstm_model':
        model = BaseLSTM(config)
        model.to(model.device)  # Move the model to the specified device (CPU or GPU)
    elif model_type == 'cnn_model':
        from models.base_cnn import BaseCNN
        model = BaseCNN(config)
        model.to(model.device)  # Move the model to the specified device (CPU or GPU)
    # elif model_type == 'other_model_type':
        # model = other_model_type(config)
    else:
        raise ValueError(f"Model type {model_type} not recognized. Please specify a valid model type in the config.")
    
    # Train the model using the fetched data
    model.training_loop(model_name)