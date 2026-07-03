"""
General models class that all models that can be trained should extend so we can have one common main function that defines command line arguments and then calls the specific training functions and such for the different functions. This will allow us to train the different models and have one main function that can implement all the boilerplate command line handling and such. If we want to write a practical predictor, we can then also have it load the trained model and use it to make predictions. This class allows for us to implement this irrespective of which model we end up using to actually do the predictions.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from .embedding_model import Embedding_Model
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class embedded_LSTM(nn.Module):
    def __init__(self, cat_emb_dims, num_continuous, hidden_size):
        super(embedded_LSTM, self).__init__()
        
        # 1. The Embedding Layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in cat_emb_dims
        ])
        
        total_emb_size = sum(emb_dim for _, emb_dim in cat_emb_dims)
        total_lstm_input_size = total_emb_size + num_continuous
        
        # 2. The LSTM Layer
        self.lstm = nn.LSTM(
            input_size=total_lstm_input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. The Multi-Task Output Heads
        self.wpa_head = nn.Linear(hidden_size, 1)
        self.action_head = nn.Linear(hidden_size, 3)

    def forward(self, cont_x, cat_x):
        """
        The standard forward pass used during training.
        Outputs raw action_logits (for CrossEntropyLoss) and wpa_pred (for MSELoss).
        """
        emb_outputs = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded_feature = emb_layer(cat_x[:, :, i])
            emb_outputs.append(embedded_feature)
            
        embedded_cat_x = torch.cat(emb_outputs, dim=-1)
        fused_input = torch.cat([cont_x, embedded_cat_x], dim=-1)
        
        lstm_out, _ = self.lstm(fused_input)
        final_step_out = lstm_out[:, -1, :]
        
        wpa_pred = self.wpa_head(final_step_out).squeeze()
        action_logits = self.action_head(final_step_out)
        
        return action_logits, wpa_pred

    def predict(self, cont_x, cat_x):
        """Inference method that safely disables gradients, fixes dimensions, and outputs final classes."""
        
        # 1. Enforce strict data types and push to the GPU
        cont_x = cont_x.to(dtype=torch.float32, device=device)
        cat_x = cat_x.to(dtype=torch.long, device=device)
        
        # 2. Automatically add the sequence dimension if it is missing
        # Transforms [batch, features] into [batch, 1, features]
        if cont_x.dim() == 2:
            cont_x = cont_x.unsqueeze(1)
        if cat_x.dim() == 2:
            cat_x = cat_x.unsqueeze(1)
            
        # Store training state
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Get the raw outputs from the forward pass
            action_logits, wpa_pred = self.forward(cont_x, cat_x)
            
            # Convert the 3 raw action logits into a single final class prediction
            predicted_actions = torch.argmax(action_logits, dim=1)
            
        # Restore training state
        if was_training:
            self.train()
            
        return predicted_actions, wpa_pred
    
    def fit(self, X_cont, X_cat, A, Y, epochs=10, batch_size=64, lr=0.001):
        """
        The integrated training loop.
        Accepts raw PyTorch tensors, formats them, and trains the model.
        """
        # 1. Enforce strict data types to prevent PyTorch crashes
        X_cont = X_cont.to(dtype=torch.float32)
        X_cat = X_cat.to(dtype=torch.long)
        A = A.to(dtype=torch.long)
        Y = Y.to(dtype=torch.float32)

        # 2. Automatically add the sequence dimension if it is missing
        # Transforms [batch, features] into [batch, 1, features]
        if X_cont.dim() == 2:
            X_cont = X_cont.unsqueeze(1)
        if X_cat.dim() == 2:
            X_cat = X_cat.unsqueeze(1)

        # 3. Create the DataLoader
        dataset = TensorDataset(X_cont, X_cat, A, Y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 4. Initialize Optimizer and Loss Functions
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.MSELoss()

        # Ensure the model is in training mode
        self.train()
        print(f"Starting training on {device} for {epochs} epochs...")

        # 5. The Training Loop
        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch_cont, batch_cat, batch_A, batch_Y in train_loader:
                # Push batches to the AMD graphics card
                batch_cont = batch_cont.to(device)
                batch_cat = batch_cat.to(device)
                batch_A = batch_A.to(device)
                batch_Y = batch_Y.to(device)
                
                optimizer.zero_grad()
                
                # Forward Pass
                action_logits, wpa_pred = self.forward(batch_cont, batch_cat)
                
                # Calculate Combined Loss
                loss_cls = criterion_cls(action_logits, batch_A)
                loss_reg = criterion_reg(wpa_pred, batch_Y)
                total_loss = loss_cls + loss_reg
                
                # Backpropagate
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")