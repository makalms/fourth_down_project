"""
Basic LSTM model for predicting win probability. This model has an embedding layer for the features which are both categorical and continuous, and then a LSTM layer followed by a dense layer to output the win probability as a continuous value between 0 and 1. Since this has a continuous output, we will use a mean squared error loss function to train the model.
"""

from logging import config
from models.model import Model
import utils.dataprocessing as dp

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseLSTM(Model, nn.Module):

    def __init__(self, config):

        Model.__init__(self, config)
        nn.Module.__init__(self)

        # Architecture of the LSTM model. This will be a simple LSTM model with two input layers, one layer for the categorical features (the action) and one layer for the continuous features (the confounders/state, number given by input_size), and then a LSTM layer followed by a dense layer to output the win probability as a continuous value between 0 and 1. Since this has a continuous output, we will use a mean squared error loss function to train the model.

        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 3)
        self.input_size = config.get("input_size", 5)
        self.bidirectional = config.get("bidirectional", True)
        self.output_size = config.get("output_size", 1)
        self.device = config.get("device", 'cuda')  # Use 'cuda' for GPU or 'cpu' for CPU
        # Switch to using cpu if cuda is not available
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU.")
            self.device = 'cpu'
        self.batch_size = config.get("batch_size", 64)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.epochs = config.get("epochs", 500)
        self.patience = config.get("patience", 10)

        # hidden_size * 2 because it's bidirectional
        lstm_output_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        # State input layer with embedding
        self.state_input_layer = nn.Linear(self.input_size, 8)

        # Action input layer for one-hot encodings
        self.action_input_layer = nn.Linear(6, 6) # Assuming 6 unique actions, output size is 6)

        # Combination layer to combine the state and action embeddings
        self.combination_layer = nn.Linear(8 + 6, self.hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(lstm_output_size, self.output_size)

        # Initialize the LSTM weights using Xavier initialization
        def init_weights(m):
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_ih" in name:  # Input-to-hidden weights
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:  # Hidden-to-hidden weights
                        nn.init.orthogonal_(param)
                    elif "bias" in name:  # Bias terms
                        nn.init.zeros_(param)
            
        self.apply(init_weights)


    def forward(self, x, a):
        # Forward pass through the LSTM model. It should take in the confounders/state X and the action/treatment A and return the predicted outcome/label.
        # If bidirectional is True, then make it bidirectional

        # Pass through the state input layer
        state_embedding = self.state_input_layer(x)

        # Pass through the action input layer
        action_embedding = self.action_input_layer(a)

        # Concatenate the state and action embeddings
        combined_input = torch.cat((state_embedding, action_embedding), dim=2)

        # Pass through the combination layer
        combined_input = self.combination_layer(combined_input)

        # Initialize hidden and cell states
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, self.batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * num_directions, self.batch_size, self.hidden_size).to(self.device)

        # Pass through BiLSTM layer
        out, _ = self.lstm(combined_input, (h0, c0))  # LSTM returns output and (hidden, cell)

        # Apply Layer Normalization
        out = self.layer_norm(out)

        # last_output = out[:, -1, :]
        last_output = torch.mean(out, dim=1)

        last_output = self.dropout_layer(last_output)
        output = self.fc(last_output)

        return output
        

    def predict(self, X):
        # Function that will be called to make predictions using the LSTM model. It should take in the confounders/state X and return the predicted outcome/label.
        pass


    def training_loop(self):

        # Load the data from huggingface and split it into training and testing sets according to the specifications in the config. Also split the data into X, A, Y according to the specifications in the config.

        dataset_name = self.config.get("dataset_name", "afzalmengal/nfl-4th-downs-dataset")
        dataset = load_dataset(dataset_name)
        # check if this is a DatasetDict or a Dataset
        if isinstance(dataset, dict):
            # concatenate the train and test splits into one dataset
            dataset = pd.concat([dataset[split].to_pandas() for split in dataset.keys()])
        else:
            dataset = dataset.to_pandas()

        # Split the data into training and testing sets according to the specifications in the config. Also split the data into X, A, Y according to the specifications in the config.
        X_cols = self.config.get("x_columns", ['qtr', 'game_seconds_remaining', 'yardline_100', 'ydstogo', 'score_differential'])
        A_cols = self.config.get("a_columns", ['play_type'])
        Y_cols = self.config.get("y_columns", ['epa'])
        train_seasons = self.config.get("training_seasons", [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
        test_seasons = self.config.get("testing_seasons", [2020, 2021])

        # Check for NaN values, get rid of all rows containing them in any of the relevant columns, print the number of rows dropped.
        original_rows = len(dataset)
        dataset = dataset.dropna(subset=X_cols + A_cols + Y_cols)
        dropped_rows = original_rows - len(dataset)
        if dropped_rows > 0:
            print(f"Number of rows dropped due to NaN values: {dropped_rows}")

        train = dataset[dataset['season'].isin(train_seasons)]
        test = dataset[dataset['season'].isin(test_seasons)]

        train_X = train[X_cols]
        train_A = train[A_cols]
        train_Y = train[Y_cols]
        test_X = test[X_cols]
        test_A = test[A_cols]
        test_Y = test[Y_cols]

        # Preprocessing the data to be in the right shape for the LSTM model. They are currently in the shape of dataframes with one column for each feature/output/action, and we need to convert them to tensors
        # All features are numerical, the action, which we also input is categorical, and the output is continuous.
        # We also want to apply some normalization and standardization to the data

        # Normalize and standardize the data (X_train, Y_train, X_test, Y_test) using MinMaxScaler and StandardScaler from sklearn. We will fit the scalers on the training data and then transform both the training and testing data. We will not scale the action A since it is categorical. It will be one hot encoded
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        train_X_scaled = x_scaler.fit_transform(train_X)
        train_Y_scaled = y_scaler.fit_transform(train_Y.values.reshape(-1, 1)).flatten()
        test_X_scaled = x_scaler.transform(test_X)
        test_Y_scaled = y_scaler.transform(test_Y.values.reshape(-1, 1)).flatten()

        # One-hot encode the action
        train_A_encoded = pd.get_dummies(train_A, columns=A_cols, dtype=int)
        test_A_encoded = pd.get_dummies(test_A, columns=A_cols, dtype=int)

        # Reshape the data to include sequence length (1)
        train_X_scaled = train_X_scaled.reshape(train_X_scaled.shape[0], 1, train_X_scaled.shape[1])
        test_X_scaled = test_X_scaled.reshape(test_X_scaled.shape[0], 1, test_X_scaled.shape[1])
        train_A_encoded = train_A_encoded.values.reshape(train_A_encoded.shape[0], 1, train_A_encoded.shape[1])
        test_A_encoded = test_A_encoded.values.reshape(test_A_encoded.shape[0], 1, test_A_encoded.shape[1])
        train_Y_scaled = train_Y_scaled.reshape(train_Y_scaled.shape[0], 1)
        test_Y_scaled = test_Y_scaled.reshape(test_Y_scaled.shape[0], 1)

        # Convert to tensors
        train_X_tensor = torch.FloatTensor(train_X_scaled).to(self.device)
        train_Y_tensor = torch.FloatTensor(train_Y_scaled).to(self.device)
        train_A_tensor = torch.FloatTensor(train_A_encoded).to(self.device)
        test_X_tensor = torch.FloatTensor(test_X_scaled).to(self.device)
        test_Y_tensor = torch.FloatTensor(test_Y_scaled).to(self.device)
        test_A_tensor = torch.FloatTensor(test_A_encoded).to(self.device)

        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(train_X_tensor, train_A_tensor, train_Y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers = 4,
            drop_last = True
        )
        # Create dataloaders
        test_dataloader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(test_X_tensor, test_A_tensor, test_Y_tensor),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers = 4,
            drop_last = True
        )

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Metrics to track the best model
        best_loss = float('inf')
        train_losses = []
        test_losses = []

        # Training loop
        no_improvement_counter = 0
        for epoch in range(self.epochs):

            print(f"Epoch {epoch+1}/{self.epochs}...")
            self.train()  # Set the model to training mode
            train_loss = 0.0
            for X_batch, A_batch, Y_batch in train_dataloader:
                optimizer.step()
                outputs = self.forward(X_batch, A_batch)
                loss = criterion(outputs, Y_batch)
                if torch.isnan(loss):
                    print("NaN loss encountered. Stopping training.")
                    return
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            # Evaluate on the test set
            self.eval()  # Set the model to evaluation mode
            test_loss = 0.0
            with torch.no_grad():
                for X_batch, A_batch, Y_batch in test_dataloader:
                    outputs = self.forward(X_batch, A_batch)
                    loss = criterion(outputs, Y_batch)
                    test_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} complete\n Train Loss: {train_loss/len(train_dataloader):.4f}\n Test Loss: {test_loss/len(test_dataloader):.4f}\n")

            train_losses.append(train_loss/len(train_dataloader))
            test_losses.append(test_loss/len(test_dataloader))

            # Plot training and test loss to files
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig('loss_plot.png')

            # Check for improvement
            if test_loss < best_loss:
                best_loss = test_loss
                no_improvement_counter = 0
                # Save the model
                torch.save(self.state_dict(), 'best_model.pth')
                print(f"Model saved with test loss: {best_loss/len(test_dataloader):.4f}")
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
