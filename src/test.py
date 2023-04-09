# author: Suraj Giri
# BSc Thesis, CS, Contructor University

# in this script, we will be taking the model saved into the 
# KRR_model.pkl file and use it to predict the energies of the
# molecules in the test set

# We will also carry out various tests to check the accuracy of
# the model like calculating the mean absolute error, mean squared
# error, root mean squared error, and the coefficient of determination

# importing the libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import qml
from qml.kernels import matern_kernel
import pickle
import csv
import yaml
from qml.math import cho_solve
import json

from dvclive import Live # for logging the metrics

def test_n_evaluate(K, alpha, X, Y, live, type):
    """
    This function will take the already saved KRR model and the 
    already saved X and Y arrays from the pickle files. 
    It will then train the model on the X set and evaluate 
    the model on the Y set. It will then return the errors 
    and the predicted values.
    """
    # calculating the predicted energies using the KRR model and the X dataset
    Y_pred = np.dot(K, alpha)

    # calculating the mean absolute error
    MAE = np.mean(np.abs(Y_pred - Y))

    # calculating the mean squared error
    MSE = np.mean((Y_pred - Y)**2)

    # calculating the root mean squared error
    RMSE = np.sqrt(np.mean((Y_pred - Y)**2))

    # calculating the coefficient of determination
    SS_tot = np.sum((Y - np.mean(Y))**2) # total sum of squares
    SS_res = np.sum((Y - Y_pred)**2) # residual sum of squares
    COD = 1 - (SS_res/SS_tot) # coefficient of determination

        # Listing the errors
    errors = [(MAE, MSE, RMSE, COD)]

    #### logging the metrics
    if not live.summary:
        live.summary = {"errors": {}}
    live.summary["errors"]["MAE"] = MAE
    live.summary["errors"]["MSE"] = MSE
    live.summary["errors"]["RMSE"] = RMSE
    live.summary["errors"]["COD"] = COD

    molecule = [{"molecule": str(i+1), "actual": str(Y[i]), "predicted": str(Y_pred[i])} for i in range(len(Y))]

    data = {}
    for i, item in enumerate(molecule):
        data[f"molecule_{i+1}"] = item

    # logging the actual and predicted energies for each molecule to the ./output/test/live/{type}.json file
    json_path = "./output/test/actual_pred"
    if not os.path.exists(os.path.join(json_path)):
        os.makedirs(os.path.join(json_path))
    json_file = os.path.join(json_path, f"{type}.json")
    with open(json_file, "w") as f:
        # json.dump(data, f, indent=4)
        json.dump(molecule, f, indent=4)


# loading the KRR model from the KRR_model.pkl file
with open("./output/models/KRR_model.pkl", "rb") as f:
    KRR_model = pickle.load(f)
sigma, order, metric, alpha, K_test_train, K = KRR_model

# Importing the X_test and Y_test dataset that was outputted from the train.py script
with open("./output/dataset/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("./output/dataset/Y_test.pkl", "rb") as f:
    Y_test = pickle.load(f)

# Importing the X_train and Y_train dataset that was outputted from the train.py script
with open("./output/dataset/X_train.pkl", "rb") as f:
    X_train = pickle.load(f)
with open("./output/dataset/Y_train.pkl", "rb") as f:
    Y_train = pickle.load(f)

# testing and evaluating the model on test dataset
test_path = "./output/test"
# live = Live(test_path, dvcyaml=False)
live = Live(os.path.join(test_path, "live"), dvcyaml=False)
test_n_evaluate(K_test_train, alpha, X_test, Y_test, live, "test")
test_n_evaluate(K, alpha, X_train, Y_train, live, "train")
live.make_summary()
