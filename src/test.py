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

# loading the KRR model from the KRR_model.pkl file
with open("./output/models/KRR_model.pkl", "rb") as f:
    KRR_model = pickle.load(f)
sigma, order, metric, alpha, K_test_train = KRR_model
# print("sigma: ", sigma)
# print("order: ", order)
# print("metric: ", metric)
# print("alpha: ", alpha)
# print("K_test_train: ", K_test_train)
# print("K_test_train.shape: ", K_test_train.shape)


# Importing the X_test dataset that was outputted from the train.py script
with open("./output/dataset/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
print("X_test: \n", X_test)

# Importing the Y_test dataset that was outputted from the train.py script
with open("./output/dataset/Y_test.pkl", "rb") as f:
    Y_test = pickle.load(f)
print("Y_test: \n", Y_test)

# calculating the predicted energies using the KRR model and the X_test dataset
Y_pred_test = np.dot(K_test_train, alpha)
print("Y_pred_test: \n", Y_pred_test)

# calculating the mean absolute error
MAE = np.mean(np.abs(Y_pred_test - Y_test))
print("MAE: ", MAE)

# calculating the mean squared error
MSE = np.mean((Y_pred_test - Y_test)**2)
print("MSE: ", MSE)

# calculating the root mean squared error
RMSE = np.sqrt(np.mean((Y_pred_test - Y_test)**2))
print("RMSE: ", RMSE)

# calculating the coefficient of determination
SS_tot = np.sum((Y_test - np.mean(Y_test))**2) # total sum of squares
SS_res = np.sum((Y_test - Y_pred_test)**2) # residual sum of squares
COD = 1 - (SS_res/SS_tot) # coefficient of determination
print("COD: ", COD)


