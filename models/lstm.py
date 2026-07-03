"""
General models class that all models that can be trained should extend so we can have one common main function that defines command line arguments and then calls the specific training functions and such for the different functions. This will allow us to train the different models and have one main function that can implement all the boilerplate command line handling and such. If we want to write a practical predictor, we can then also have it load the trained model and use it to make predictions. This class allows for us to implement this irrespective of which model we end up using to actually do the predictions.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from .embedding_model import Embedding_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class embedded_LSTM():
    def __init__(self, config):
        """
        Initialize the model. It takes in the parameters for the model from the config.
        This class holds both the Embedder and the core LSTM PyTorch logic.
        """
        self.config = config
        self.device = device
        
        # 1. Initialize the Embedding Model (Class 1)
        self.embedder = Embedding_Model(config).to(self.device)
        
        # 2. Calculate the total input size for the LSTM
        total_emb_size = sum(emb_dim for _, emb_dim in config['cat_emb_dims'])
        num_continuous = config['num_continuous']
        total_lstm_input_size = total_emb_size + num_continuous
        hidden_size = config.get('hidden_size', 64)
        
        # 3. Initialize the core LSTM and output heads (PyTorch primitives)
        self.lstm = nn.LSTM(
            input_size=total_lstm_input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        ).to(self.device)
        
        # Multi-task heads: One for WPA (Regression) and one for Play Type (Classification)
        self.wpa_head = nn.Linear(hidden_size, 1).to(self.device)
        self.action_head = nn.Linear(hidden_size, 3).to(self.device)
        
        # 4. Set up optimizers and loss functions
        self.criterion_classification = nn.CrossEntropyLoss()
        self.criterion_regression = nn.MSELoss()
        
        # We optimize both the embeddings and the LSTM simultaneously 
        # (unless embeddings were frozen in the Embedding_Model init)
        params = list(self.embedder.parameters()) + list(self.lstm.parameters()) + \
                 list(self.wpa_head.parameters()) + list(self.action_head.parameters())
                 
        self.optimizer = optim.Adam(params, lr=config.get('learning_rate', 0.001))

    def predict(self, X):
        """
        Function that will be called to make predictions using the model. 
        It takes in the confounders/state X and returns the predicted outcome/label.
        """
        self.embedder.eval()
        self.lstm.eval()
        
        with torch.no_grad():
            cont_x = X['cont'].to(self.device)
            cat_x = X['cat'].to(self.device)
            
            # 1. Get embeddings from our Embedding_Model
            embedded_features = self.embedder.predict(cat_x)
            
            # 2. Combine continuous variables with the new embeddings
            fused_input = torch.cat([cont_x, embedded_features], dim=-1)
            
            # 3. Pass through LSTM
            lstm_out, _ = self.lstm(fused_input)
            final_step_out = lstm_out[:, -1, :] # Grab the context of the final play
            
            # 4. Generate predictions
            predicted_wpa = self.wpa_head(final_step_out).squeeze()
            action_logits = self.action_head(final_step_out)
            predicted_actions = torch.argmax(action_logits, dim=1)
            
        return predicted_actions, predicted_wpa

    def train(self, X, A, Y):
        """
        Function that will be called to train the model. 
        It takes in the confounders/state X, the action/treatment A, and the outcome/label Y.
        """
        self.embedder.train()
        self.lstm.train()
        
        epochs = self.config.get('epochs', 10)
        batch_size = self.config.get('batch_size', 64)
        dataset_size = len(A)
        
        print(f"\nStarting training loop on {self.device}...")
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            # Manually shuffle and batch the dataset
            indices = torch.randperm(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_idx = indices[i : i + batch_size]
                
                # Move batches to GPU
                cont_batch = X['cont'][batch_idx].to(self.device)
                cat_batch = X['cat'][batch_idx].to(self.device)
                action_batch = A[batch_idx].to(self.device)
                wpa_batch = Y[batch_idx].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward Pass
                embedded_features = self.embedder.predict(cat_batch)
                fused_input = torch.cat([cont_batch, embedded_features], dim=-1)
                
                lstm_out, _ = self.lstm(fused_input)
                final_step_out = lstm_out[:, -1, :]
                
                wpa_pred = self.wpa_head(final_step_out).squeeze()
                action_logits = self.action_head(final_step_out)
                
                # Compute combined loss (Classification + Regression)
                loss_cls = self.criterion_classification(action_logits, action_batch)
                loss_reg = self.criterion_regression(wpa_pred, wpa_batch)
                total_loss = loss_cls + loss_reg
                
                # Backpropagation
                total_loss.backward()
                self.optimizer.step()
                
                running_loss += total_loss.item()
                
            avg_loss = running_loss / (dataset_size / batch_size)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
            

        print("Training complete. Models and metrics ready for saving.")