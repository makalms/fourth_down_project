import torch
from torch import nn

from .model import Model

class Embedding_Model(nn.Module, Model):

    def __init__(self, config):
        # Initialize the embedding model. It should take in the parameters for the embedding model from the config
        # If a path to a pretrained embedding model is provided in the config, load the pretrained model and freeze its weights. Otherwise, initialize a new embedding model.
        pass
        

    def predict(self, X):
        # Function that will be called to make predictions using the embedding model. It should take in the confounders/state X and return the predicted embeddings.
        return None