"""
This module contains functions for parsing a configuration file
"""

import configparser

def parse_config(config_name):
    # Function that will be called to parse the config file. It should take in the name of the config to use and return a dictionary of the parameters specified in the config.
    config = configparser.ConfigParser()
    config.read('config.ini')
    if config_name not in config:
        raise ValueError(f"Config {config_name} not found in config.ini")
    return dict(config[config_name])