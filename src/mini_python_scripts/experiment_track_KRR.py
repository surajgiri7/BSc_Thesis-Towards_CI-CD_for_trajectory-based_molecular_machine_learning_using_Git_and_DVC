import random

from dvclive import Live
from PIL import Image

import pickle
from import_dataset import compounds, energies

def train_model(pkl_path, input_data):
    # Load the saved model from the .pkl file
    model = pickle.load(pkl_path)

    # Use the loaded model to make predictions on the input data
    predictions = model.predict(input_data)

    return predictions

pkl_dataset_path = "../output/dataset/"
pkl_model_path = "../output/models/KRR_model.pkl"

EPOCHS = 2

with Live(save_dvc_exp=True) as live:
    live.log_param("epochs", EPOCHS)

    for epoch in range(EPOCHS):
        # Load the dataset
        # X_train = joblib.load(f"{dataset}/X_train.pkl")
        # Y_train = joblib.load(f"{dataset}/Y_train.pkl")
        # X_test = joblib.load(f"{dataset}/X_test.pkl")
        # Y_test = joblib.load(f"{dataset}/Y_test.pkl")

        # Train the model
        predictions = train_model(pkl_model_path, compounds)

        # Calculate the accuracy
        accuracy = sum(predictions == Y_test) / len(Y_test)

        # Log the accuracy to DVC
        live.log("accuracy", accuracy)

        # Log the image to DVC
        image = Image.open(f"{dataset}/image.png")
        live.log_image("image", image)
        
        live.next_step()