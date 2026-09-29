"""
Functions for dataprocessing, including splitting the data into training and testing sets
"""

import torch
import pandas as pd


def train_test_split(X, A, Y, config):

    # Function that will be called to split the data into training and testing sets. It should take in the data in the format X,A,Y, where:
    # X: A list of the confounders/state
    # A: The action/treatment
    # Y: The outcome/label
    # All three should be in the form of pandas dataframes that can then be further processed
    # Take the parameters for splitting the data from the config (e.g. which seasons to use for training and which for testing)
    # Return the split data in the same format X_train, A_train, Y_train, X_test, A_test, Y_test
    
    train_seasons = config['training_seasons']
    test_seasons = config['testing_seasons']

    X_train = X[X['season'].isin(train_seasons)]
    A_train = A[A['season'].isin(train_seasons)]
    Y_train = Y[Y['season'].isin(train_seasons)]

    X_test = X[X['season'].isin(test_seasons)]
    A_test = A[A['season'].isin(test_seasons)]
    Y_test = Y[Y['season'].isin(test_seasons)]

    return X_train, A_train, Y_train, X_test, A_test, Y_test


def data_to_tensors(X, A, Y):
    # Function that will be called to convert the data from pandas dataframes to PyTorch tensors. It should take in the data in the format X,A,Y, where:
    # X: A list of the confounders/state
    # A: The action/treatment
    # Y: The outcome/label
    # All three should be in the form of pandas dataframes that can then be further processed
    # Return the data in the same format X_tensor, A_tensor, Y_tensor, where all three are PyTorch tensors
    
    X_tensor = torch.tensor(X.values)
    A_tensor = torch.tensor(A.values)
    Y_tensor = torch.tensor(Y.values)

    return X_tensor, A_tensor, Y_tensor


def get_state_embeddings(X, config):

    # Function that will be called to get the state embeddings for the confounders/state X. It should take in the confounders/state X in the form of a tensor, and use the specified embedding model to get the state embeddings. It should return the state embeddings in the form of a tensor.

    # Get the embedding model specified in the config
    embedding_model_name = config['embedding_model']
    if embedding_model_name == None:
        raise ValueError("No embedding model specified in the config. Please specify an embedding model to use for getting state embeddings.")
    
    # Initialize the embedding model
    model = Embedding_Model(config)

    # Get the state embeddings using the embedding model
    state_embeddings = model.predict(X)

    return state_embeddings

