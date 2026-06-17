"""
General models class that all models that can be trained should extend so we can have one common main function that defines command line arguments and then calls the specific training functions and such for the different functions. This will allow us to train the different models and have one main function that can implement all the boilerplate command line handling and such. If we want to write a practical predictor, we can then also have it load the trained model and use it to make predictions. This class allows for us to implement this irrespective of which model we end up using to actually do the predictions.
"""


class Model():

    def __init__(self, config):
        # Initialize the model. It should take in the parameters for the model from the config
        self.config = config
        pass

    def predict(self, X):
        # Function that will be called to make predictions using the model. It should take in the confounders/state X and return the predicted outcome/label.
        pass

    def train(self, X, A, Y):
        # Function that will be called to train the model. It should take in the confounders/state X, the action/treatment A, and the outcome/label Y and train the model accordingly.
        # THe training function should save the model and metrics after training. Ideally we would use mlflow for this. There is quite a bit of boilerplate code for this, and I'm wondering if this should be implemented here in the base model class so that we don't have to implement it for each specific model. However, I'm unsure if this would still work for the basic regression models, to track the same metrics and such.
        pass