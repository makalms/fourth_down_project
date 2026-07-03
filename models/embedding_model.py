import torch
from torch import nn
import os

from .model import Model

    
class Embedding_Model(nn.Module, Model):
    def __init__(self, config):
        super(Embedding_Model, self).__init__()
        # Initialize the embedding model. It should take in the parameters from the config
        self.config = config
        self.cat_emb_dims = config.get('cat_emb_dims', [])
        
        # Create the embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in self.cat_emb_dims
        ])
        
        # If a path to a pretrained embedding model is provided in the config, 
        # load the pretrained model and freeze its weights.
        pretrained_path = config.get('pretrained_embedding_path', None)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained embeddings from {pretrained_path}...")
            self.load_state_dict(torch.load(pretrained_path))
            # Freeze the weights so they don't update during LSTM training
            for param in self.parameters():
                param.requires_grad = False
        else:
            print("Initializing a new embedding model from scratch.")

    def forward(self, X_cat):
        # PyTorch strictly requires the logic to live in the forward() method
        return self.predict(X_cat)

    def predict(self, X_cat):
        """
        Function that will be called to make predictions using the embedding model. 
        It takes in the categorical confounders/state X and returns the predicted embeddings.
        """
        emb_outputs = []
        # Pass each categorical feature through its respective embedding layer
        for i, emb_layer in enumerate(self.embeddings):
            emb_outputs.append(emb_layer(X_cat[:, :, i]))
            
        # Concatenate them all into one flat array per play
        return torch.cat(emb_outputs, dim=-1)