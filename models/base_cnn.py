"""
Basic CNN model for predicting win probability. This model has an embedding layer for the features which are both categorical and continuous, and then a LSTM layer followed by a dense layer to output the win probability as a continuous value between 0 and 1. Since this has a continuous output, we will use a mean squared error loss function to train the model.
"""

from logging import config
import os
from models.model import Model
import utils.dataprocessing as dp

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BaseCNN(Model, nn.Module):

    def __init__(self, config):

        Model.__init__(self, config)
        nn.Module.__init__(self)

        # Architecture of the LSTM model. This will be a simple LSTM model with two input layers, one layer for the categorical features (the action) and one layer for the continuous features (the confounders/state, number given by input_size), and then a LSTM layer followed by a dense layer to output the win probability as a continuous value between 0 and 1. Since this has a continuous output, we will use a mean squared error loss function to train the model.

        self.hidden_size = config.get("hidden_size", 128)
        self.num_layers = config.get("num_layers", 3)
        self.input_size = config.get("input_size", 5)
        self.output_size = config.get("output_size", 1)
        self.a_mappings = config.get("a_mappings", {'go': 0, 'punt': 1, 'fg': 2})
        self.device = config.get("device", 'cuda')  # Use 'cuda' for GPU or 'cpu' for CPU
        # Switch to using cpu if cuda is not available
        if self.device == 'cuda' and torch.cuda.is_available() == False:
            print("CUDA is not available. Switching to CPU.")
            self.device = 'cpu'
        self.batch_size = config.get("batch_size", 64)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.epochs = config.get("epochs", 500)
        self.patience = config.get("patience", 10)

        # set seed for reproducibility
        torch.manual_seed(42)

        self.cnn_in = nn.Conv1d(
            in_channels=1,  # Since we are treating the input as a single channel
            out_channels=self.hidden_size,  # Number of filters
            kernel_size=3,  # Size of the filter
            stride=1,  # Stride of the convolution
            padding=1  # Padding to keep the output size the same as input size
        )
        
        self.cnn_inner = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,  # Number of filters
            kernel_size=3,  # Size of the filter
            stride=1,  # Stride of the convolution
            padding=1  # Padding to keep the output size the same as input size
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        # State input layer with embedding
        self.state_input_layer = nn.Linear(self.input_size, 8)

        # Action input layer for one-hot encodings
        self.action_input_layer = nn.Linear(len(self.a_mappings), len(self.a_mappings))

        # Combination layer to combine the state and action embeddings
        self.combination_layer = nn.Linear(8 + len(self.a_mappings), self.hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Initialize the CNN weights using Xavier initialization
        def init_weights(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
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
        combined_input = torch.cat((state_embedding, action_embedding), dim=1)

        # Pass through the combination layer
        combined_input = self.combination_layer(combined_input).to(self.device)

        # Pass through first CNN layer, followed by ReLU activation
        out = self.cnn_in(combined_input.unsqueeze(1))  # Add channel dimension
        out = torch.relu(out)

        # Pass through second CNN layer num_layers times, always followed by ReLU activation
        for i in range(self.num_layers - 1):
            out = self.cnn_inner(out)
            out = torch.relu(out)

        # Apply Layer Normalization
        out = self.layer_norm(out)

        # last_output = out[:, -1, :]
        last_output = torch.mean(out, dim=1)

        last_output = self.dropout_layer(last_output)
        output = self.fc(last_output)

        return output
        

    def predict(self, X, A):
        # Function that will be called to make predictions using the LSTM model. It should take in the confounders/state X and return the predicted outcome/label.
        
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = self.forward(X, A).cpu().numpy()
        return predictions


    def training_loop(self, model_name):

        save_path = f"runs/{model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        A_cols = self.config.get("a_columns", ['label'])
        Y_cols = self.config.get("y_columns", ['epa'])
        train_seasons = self.config.get("training_seasons", [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
        test_seasons = self.config.get("testing_seasons", [2020, 2021])

        # Check for NaN values, get rid of all rows containing them in any of the relevant columns, print the number of rows dropped.
        original_rows = len(dataset)
        dataset = dataset.dropna(subset=X_cols + A_cols + Y_cols)
        dropped_rows = original_rows - len(dataset)
        if dropped_rows > 0:
            print(f"Number of rows dropped due to NaN values: {dropped_rows}")
            print(f"Number of rows remaining: {len(dataset)}")

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
        x_scaler = MinMaxScaler()
        y_scaler = StandardScaler()

        train_X_scaled = x_scaler.fit_transform(train_X)
        train_Y_scaled = y_scaler.fit_transform(train_Y.values.reshape(-1, 1)).flatten()
        test_X_scaled = x_scaler.transform(test_X)
        test_Y_scaled = y_scaler.transform(test_Y.values.reshape(-1, 1)).flatten()

        # One-hot encode the action (use mappings to make the action into a one-hot encoding, where each column is a possible action and the value is 1 if that action was taken and 0 otherwise)
        train_A_encoded = pd.DataFrame(columns=self.a_mappings.keys())
        for action in self.a_mappings.keys():
            train_A_encoded[action] = (train_A.values.flatten() == action).astype(int)
        train_A_encoded = train_A_encoded.values
        test_A_encoded = pd.DataFrame(columns=self.a_mappings.keys())
        for action in self.a_mappings.keys():
            test_A_encoded[action] = (test_A.values.flatten() == action).astype(int)
        test_A_encoded = test_A_encoded.values

        # Convert to tensors
        train_X_tensor = torch.FloatTensor(train_X_scaled)
        train_Y_tensor = torch.FloatTensor(train_Y_scaled)
        train_A_tensor = torch.FloatTensor(train_A_encoded)
        test_X_tensor = torch.FloatTensor(test_X_scaled)
        test_Y_tensor = torch.FloatTensor(test_Y_scaled)
        test_A_tensor = torch.FloatTensor(test_A_encoded)

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
                X_batch, A_batch, Y_batch = X_batch.to(self.device), A_batch.to(self.device), Y_batch.to(self.device)
                optimizer.step()
                outputs = self.forward(X_batch, A_batch)
                loss = criterion(outputs.reshape(-1), Y_batch.reshape(-1))
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
                    X_batch, A_batch, Y_batch = X_batch.to(self.device), A_batch.to(self.device), Y_batch.to(self.device)
                    outputs = self.forward(X_batch, A_batch)
                    loss = criterion(outputs.reshape(-1), Y_batch.reshape(-1))
                    test_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs} complete\n Train Loss: {train_loss/len(train_dataloader):.4f}\n Test Loss: {test_loss/len(test_dataloader):.4f}\n")

            train_losses.append(train_loss/len(train_dataloader))
            test_losses.append(test_loss/len(test_dataloader))

            # Plot training and test loss to files
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='navy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(save_path + '/train_loss_plot_cnn.png')
            plt.close()
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(save_path + '/test_loss_plot_cnn.png')
            plt.close()

            # Check for improvement
            if test_loss < best_loss:
                best_loss = test_loss
                no_improvement_counter = 0
                # Save the model
                torch.save(self.state_dict(), save_path + '/best_model_cnn.pth')
                print(f"Model saved with test loss: {best_loss/len(test_dataloader):.4f}")
                # Save the hyperparameters to a text file
                with open(save_path + '/hyperparameters_cnn.txt', 'w') as f:
                    for key, value in self.config.items():
                        f.write(f"{key}: {value}\n")
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
    
    def validation_loop(self, model_name):

        save_path = f"runs/{model_name}"

        # Load the best model
        self.load_state_dict(torch.load(save_path + '/best_model_cnn.pth'))
        self.eval()  # Set the model to evaluation mode

        # Load the data from huggingface and get the validation set according to the specifications in the config. Also split the data into X, A, Y according to the specifications in the config.
        dataset_name = self.config.get("dataset_name", "afzalmengal/nfl-4th-downs-dataset")
        dataset = load_dataset(dataset_name)
        if isinstance(dataset, dict):
            dataset = pd.concat([dataset[split].to_pandas() for split in dataset.keys()])
        
        validation_seasons = self.config.get("validation_seasons", [2020, 2021])
        validation = dataset[dataset['season'].isin(validation_seasons)]

        # Training set to be used for scaling the validation set
        train_seasons = self.config.get("training_seasons", [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
        train = dataset[dataset['season'].isin(train_seasons)]

        # We want to validate the model by calculating the output of the model for all possible actions for each state in the validation set. We also want to compare to a rule-advised model that only picks between go and fg in safe field goal range, all three actions in the maybe field goal range, and only go and punt outside of field goal range. 

        # Get the X, A columns from the validation set
        X_cols = self.config.get("x_columns", ['qtr', 'game_seconds_remaining', 'yardline_100', 'ydstogo', 'score_differential'])
        A_cols = self.config.get("a_columns", ['label'])
        Y_cols = self.config.get("y_columns", ['epa'])

        # Drop any rows with NaN values in the relevant columns and print the number of rows dropped and the number of rows remaining
        original_rows = len(validation)
        validation = validation.dropna(subset=X_cols + A_cols + Y_cols)
        print(f"Number of rows dropped due to NaN values in validation set: {original_rows - len(validation)}")
        print(f"Number of rows remaining in validation set: {len(validation)}")

        validation_X = validation[X_cols]
        validation_A = validation[A_cols]
        validation_Y = validation[Y_cols]

        # Preprocess the validation data using the same scalers as the training data
        x_scaler = MinMaxScaler()
        y_scaler = StandardScaler()
        train = dataset[dataset['season'].isin(train_seasons)]
        train_X = train[X_cols]
        train_Y = train[Y_cols]
        x_scaler.fit_transform(train_X)
        y_scaler.fit_transform(train_Y.values.reshape(-1, 1))
        validation_X_scaled = x_scaler.transform(validation_X)
        validation_Y_scaled = y_scaler.transform(validation_Y.values.reshape(-1, 1)).flatten()

        validation_X_tensor = torch.FloatTensor(validation_X_scaled).to(self.device)

        # Create a list of tensors for each possible action, where each tensor is the one-hot encoding of the action repeated for the number of validation samples. This will allow us to pass in all possible actions for each state in the validation set to the model and get the predicted output for each action.
        validation_A_tensors = []
        for action in self.a_mappings.keys():
            action_index = self.a_mappings[action]
            action_tensor = torch.zeros((validation_X_tensor.shape[0], len(self.a_mappings))).to(self.device)
            action_tensor[:, action_index] = 1
            validation_A_tensors.append(action_tensor)

        # Pass the validation data through the model for each possible action and get the predicted output for each action. Then pick the action with the highest output for each state in the validation set.
        outputs = {}
        for action, action_tensor in zip(self.a_mappings.keys(), validation_A_tensors):
            outputs[action] = self.predict(validation_X_tensor, action_tensor)
        print(outputs)

        # For the true validation labels, look at the actual action taken in the validation set and get the corresponding output from the model for that action. This will allow us to compare how far off the model's predictions were from the actual actions taken in the validation set.= in terms of predicted output. Save the output vs. actual wpa as a list of pairs of (predicted_output, actual_output) for each state in the validation set
        output_vs_actual = []
        for i in range(validation_X_tensor.shape[0]):
            actual_action = validation_A.values.flatten()[i]
            actual_action_index = self.a_mappings.get(actual_action, None)
            if actual_action_index is not None:
                actual_output = outputs[actual_action][i]
                output_vs_actual.append((float(actual_output[0]), float(validation_Y_scaled[i])))
            else:
                print(f"Warning: Actual action '{actual_action}' not found in action mappings. Skipping this sample.")
        print(f"Average absolute difference between predicted and actual outputs: {np.mean([abs(pair[0] - pair[1]) for pair in output_vs_actual])}")

        # Plot the distribution of the absolute differences between the predicted outputs and actual outputs in a histogram. Save the plot to a file.
        plt.figure(figsize=(4, 4))
        plt.hist([abs(pair[0] - pair[1]) for pair in output_vs_actual], bins=100, alpha=1, color='navy')
        plt.xlabel('Abs. Difference Predicted vs. Actual WPA')
        plt.savefig(save_path + '/absolute_differences_histogram_cnn.png')
        plt.close()

        # For each state in the validation set, pick the action with the highest output and store it in a list
        predicted_actions = []
        for i in range(validation_X_tensor.shape[0]):
            state_outputs = {action: outputs[action][i] for action in self.a_mappings.keys()}
            predicted_action = max(state_outputs, key=state_outputs.get)
            predicted_actions.append(predicted_action)

        # Also get the predicted action for the rule-advised model that only picks between go and fg in safe field goal range, all three actions in the maybe field goal range, and only go and punt outside of field goal range.
        # Safe fieldgoal range: yardline_100 < 25
        # Maybe fieldgoal range: 25 <= yardline_100 <= 50
        # Outside fieldgoal range: yardline_100 > 50
        # Both limits are very conservatively set, at least up to the 30 yard line is usually considered safe for a field goal in the NFL, and beyond the 50 yard line is close to the record for the longest field goal in NFL history
        predicted_actions_rule_advised = []
        for i in range(validation_X_tensor.shape[0]):
            state_outputs = {action: outputs[action][i] for action in self.a_mappings.keys()}
            yardline_100 = validation['yardline_100'].values.flatten()[i]
            if yardline_100 < 25:
                possible_actions = ['go', 'fg']
            elif yardline_100 <= 50:
                possible_actions = ['go', 'fg', 'punt']
            else:
                possible_actions = ['go', 'punt']
            # Get the predicted action with the highest output among the possible actions
            possible_outputs = {action: state_outputs[action] for action in possible_actions}
            predicted_action = max(possible_outputs, key=possible_outputs.get)
            predicted_actions_rule_advised.append(predicted_action)

        # Unencode the predicted actions and actual actions from their one-hot encodings to their original values using the mappings
        predicted_actions = [self.a_mappings.get(a, 0) for a in predicted_actions]
        predicted_actions = [list(self.a_mappings.keys())[list(self.a_mappings.values()).index(a)] for a in predicted_actions]
        predicted_actions_rule_advised = [self.a_mappings.get(a, 0) for a in predicted_actions_rule_advised]
        predicted_actions_rule_advised = [list(self.a_mappings.keys())[list(self.a_mappings.values()).index(a)] for a in predicted_actions_rule_advised]
        actual_actions = [self.a_mappings.get(a, 0) for a in validation_A.values.flatten()]
        actual_actions = [list(self.a_mappings.keys())[list(self.a_mappings.values()).index(a)] for a in actual_actions]

        # For plotting purposes, add some light noise to the ydstogo, so they aren't lined up in perfect vertical lines. This will make the plot easier to read.
        validation['ydstogo'] = validation['ydstogo'] + np.random.normal(0, 0.2, size=validation.shape[0])

        # Plot the distribution of the predicted actions vs. actual actions in a plot showing ydstogo vs. game_seconds_remaining, with the color of the points differentiating actions. There will be three plots, one for the predicted actions, one for the actual actions, and one for the rule-advised model's predicted actions. The x-axis will be ydstogo and the y-axis will be game_seconds_remaining. The points will be colored according to the action taken. Save the plot to a file.
        colours = ['navy', 'orange', 'maroon']
        colour_map = {action: colour for action, colour in zip(self.a_mappings.keys(), colours)}
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].scatter(validation['ydstogo'], validation['game_seconds_remaining'], c=[colour_map[a] for a in predicted_actions], alpha=0.5, s=5)
        ax[0].set_title('CNN Predictions')
        ax[0].set_xlabel('Yards to Go')
        ax[0].set_ylabel('Game Seconds Remaining')
        ax[1].scatter(validation['ydstogo'], validation['game_seconds_remaining'], c=[colour_map[a] for a in predicted_actions_rule_advised], alpha=0.5, s=5)
        ax[1].set_title('CNN Predictions + Rules')
        ax[1].set_xlabel('Yards to Go')
        ax[1].set_ylabel('Game Seconds Remaining')
        ax[2].scatter(validation['ydstogo'], validation['game_seconds_remaining'], c=[colour_map[a] for a in actual_actions], alpha=0.5, s=5)
        ax[2].set_title('Actual Actions')
        ax[2].set_xlabel('Yards to Go')
        ax[2].set_ylabel('Game Seconds Remaining')
        # shared legend for all three plots, outside the plots
        fig.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=action, markersize=5, markerfacecolor=colour_map[action]) for action in self.a_mappings.keys()], loc='center right', title='Actions')
        fig.tight_layout()
        plt.savefig(save_path + '/validation_plot_cnn.png')
        plt.close()

        # Plot the same thing again, but this time with the score differential on the x-axis
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].scatter(validation['score_differential'], validation['game_seconds_remaining'], c=[colour_map[a] for a in predicted_actions], alpha=0.5, s=5)
        ax[0].set_title('CNN Predictions')
        ax[0].set_xlabel('Score Differential')
        ax[0].set_ylabel('Game Seconds Remaining')
        ax[1].scatter(validation['score_differential'], validation['game_seconds_remaining'], c=[colour_map[a] for a in predicted_actions_rule_advised], alpha=0.5, s=5)
        ax[1].set_title('CNN Predictions + Rules')
        ax[1].set_xlabel('Score Differential')
        ax[1].set_ylabel('Game Seconds Remaining')
        ax[2].scatter(validation['score_differential'], validation['game_seconds_remaining'], c=[colour_map[a] for a in actual_actions], alpha=0.5, s=5)
        ax[2].set_title('Actual Actions')
        ax[2].set_xlabel('Score Differential')
        ax[2].set_ylabel('Game Seconds Remaining')
        # shared legend for all three plots, outside the plots
        fig.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=action, markersize=5, markerfacecolor=colour_map[action]) for action in self.a_mappings.keys()], loc='center right', title='Actions')
        fig.tight_layout()
        plt.savefig(save_path + '/validation_plot_score_diff_cnn.png')
        plt.close()

        # Also plot it as an overlay onto a football field, with the yardline_100 on the x-axis, and the yards to go on the y-axis, and the color of the points differentiating actions. There will be two plots, one for the predicted actions and one for the actual actions. The football field should be 100 yards long and 53 yards wide, with the end zones being 10 yards deep. The field should have white lines every 10 yards, with the yard numbers labeled on the field.
        
        # Get max and min of yards to go, to set the limits of the y-axis. We should also adjust the y-axis to be somewhat proportional to the x-axis, so that the field looks like a football field.
        max_ydstogo = validation['ydstogo'].max()
        min_ydstogo = validation['ydstogo'].min()
        colours = ['navy', 'orange', 'maroon']
        colour_map = {action: colour for action, colour in zip(self.a_mappings.keys(), colours)}
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))
        # Make the plots look like a football field, with green field, white lines with yard numbers, and the end zones with the team names. The field should be 100 yards long and 53 yards wide, with the end zones being 10 yards deep.
        for i in range(3):
            ax[i].set_facecolor('green')
            ax[i].set_xlim(-10, 110)
            ax[i].set_ylim(min_ydstogo, max_ydstogo)
            # Draw the yard lines
            for yard in range(0, 101, 10):
                ax[i].axvline(x=yard, color='white', linewidth=2)
                if yard > 0 and yard < 100:
                    ax[i].text(yard, max_ydstogo - (max_ydstogo - min_ydstogo) * 0.1, str(yard), color='white', fontsize=8, fontweight='bold', ha='center', va='center')
                    ax[i].text(yard, min_ydstogo + (max_ydstogo - min_ydstogo) * 0.1, str(yard), color='white', fontsize=8, fontweight='bold', ha='center', va='center', rotation=180)
            # Draw the end zones (dark orange with navy blue text for the Broncos, and navy blue with neon green text for the Seahawks)
            ax[i].axvspan(-10, 0, color='darkorange')
            ax[i].axvspan(100, 110, color='navy')
            ax[i].text(-5, (max_ydstogo + min_ydstogo) / 2, 'Endzone', color='navy', fontsize=20, fontweight='bold', ha='center', va='center', rotation=90)
            ax[i].text(105, (max_ydstogo + min_ydstogo) / 2, 'Endzone', color='darkorange', fontsize=20, fontweight='bold', ha='center', va='center', rotation=270)
            # Label the axes
            ax[i].set_xlabel('Yardline (100 = Opponent End Zone)', fontsize=12)
            ax[i].set_ylabel('Yards to Go', fontsize=12)
        # Plot the predicted actions
        ax[0].scatter(validation['yardline_100'], validation['ydstogo'], c=[colour_map[a] for a in predicted_actions], alpha=0.5, s=10)
        ax[0].set_title('CNN Predictions', fontsize=14)
        # Plot the predicted actions for the rule-advised model
        ax[1].scatter(validation['yardline_100'], validation['ydstogo'], c=[colour_map[a] for a in predicted_actions_rule_advised], alpha=0.5, s=10)
        ax[1].set_title('CNN Predictions + Rules', fontsize=14)
        # Plot the actual actions
        ax[2].scatter(validation['yardline_100'], validation['ydstogo'], c=[colour_map[a] for a in actual_actions], alpha=0.5, s=10)
        ax[2].set_title('Actual Actions', fontsize=14)
        # Shared legend for both plots, outside the plots
        fig.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=action, markersize=5, markerfacecolor=colour_map[action]) for action in self.a_mappings.keys()], loc='center right', title='Actions')
        fig.tight_layout()
        plt.savefig(save_path + '/validation_plot_field_cnn.png')
        plt.close()

        # Finally, we will calculate the accuracy, precision, recall, and F1 score of the predicted actions vs. the actual actions, and print them to the console. We will also print the confusion matrix.
        print("Validation Metrics for CNN Model:")
        accuracy_cnn = accuracy_score(actual_actions, predicted_actions)
        precision_cnn = precision_score(actual_actions, predicted_actions, average='weighted', zero_division=0)
        recall_cnn = recall_score(actual_actions, predicted_actions, average='weighted', zero_division=0)
        f1_cnn = f1_score(actual_actions, predicted_actions, average='weighted', zero_division=0)
        print(f"Accuracy: {accuracy_cnn:.4f}")
        print(f"Precision: {precision_cnn:.4f}")
        print(f"Recall: {recall_cnn:.4f}")
        print(f"F1 Score: {f1_cnn:.4f}")
        cm_cnn = confusion_matrix(actual_actions, predicted_actions)
        print("Confusion Matrix:")
        print(cm_cnn)

        print("Validation Metrics for CNN Model + Rules:")
        accuracy_cnn_rule_advised = accuracy_score(actual_actions, predicted_actions_rule_advised)
        precision_cnn_rule_advised = precision_score(actual_actions, predicted_actions_rule_advised, average='weighted', zero_division=0)
        recall_cnn_rule_advised = recall_score(actual_actions, predicted_actions_rule_advised, average='weighted', zero_division=0)
        f1_cnn_rule_advised = f1_score(actual_actions, predicted_actions_rule_advised, average='weighted', zero_division=0)
        print(f"Accuracy: {accuracy_cnn_rule_advised:.4f}")
        print(f"Precision: {precision_cnn_rule_advised:.4f}")
        print(f"Recall: {recall_cnn_rule_advised:.4f}")
        print(f"F1 Score: {f1_cnn_rule_advised:.4f}")
        cm_cnn_rule_advised = confusion_matrix(actual_actions, predicted_actions_rule_advised)
        print("Confusion Matrix:")
        print(cm_cnn_rule_advised)

        # Also save the metrics to a text file
        with open(save_path + '/validation_metrics_cnn.txt', 'w') as f:
            f.write("Validation Metrics for CNN Model:\n")
            f.write(f"Accuracy: {accuracy_cnn:.4f}\n")
            f.write(f"Precision: {precision_cnn:.4f}\n")
            f.write(f"Recall: {recall_cnn:.4f}\n")
            f.write(f"F1 Score: {f1_cnn:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm_cnn) + "\n\n")

            f.write("Validation Metrics for CNN Model + Rules:\n")
            f.write(f"Accuracy: {accuracy_cnn_rule_advised:.4f}\n")
            f.write(f"Precision: {precision_cnn_rule_advised:.4f}\n")
            f.write(f"Recall: {recall_cnn_rule_advised:.4f}\n")
            f.write(f"F1 Score: {f1_cnn_rule_advised:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm_cnn_rule_advised) + "\n")
        
        # Finally plot the confusion matrices as heatmaps and save them to a file
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='rocket_r', xticklabels=self.a_mappings.keys(), yticklabels=self.a_mappings.keys())
        plt.xlabel('Predicted Actions')
        plt.ylabel('Actual Actions')
        plt.title('CNN Model')
        plt.savefig(save_path + '/confusion_matrix_cnn.png')
        fig.tight_layout()
        plt.close()
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm_cnn_rule_advised, annot=True, fmt='d', cmap='rocket_r', xticklabels=self.a_mappings.keys(), yticklabels=self.a_mappings.keys())
        plt.xlabel('Predicted Actions')
        plt.ylabel('Actual Actions')
        plt.title('CNN Model + Rules')
        fig.tight_layout()
        plt.savefig(save_path + '/confusion_matrix_cnn_rule_advised.png')
        plt.close()



        




