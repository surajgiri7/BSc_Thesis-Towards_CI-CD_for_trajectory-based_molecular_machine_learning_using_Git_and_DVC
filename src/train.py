# author: Suraj Giri
# BSc Thesis, CS, Contructor University

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

# Importing the dataset from the prepared_data.pkl file
with open("./output/prepared_data.pkl", "rb") as f:
    compounds, energies = pickle.load(f)

# Feature Engineering
# Generating Coulomb matrices for every molecule in the dataset
for mol in compounds:
    mol.generate_coulomb_matrix(size=12, sorting="row-norm")

# making a big 2D array with coloumb matrix of all the molecules
X = np.array([mol.representation for mol in compounds])

# Splitting the dataset into the Training set and Test set with 80:20 ratio
X_train = X[:int(0.8*len(X))]
X_test = X[int(0.8*len(X)):]
Y_train = np.array([mol.properties for mol in compounds[:int(0.8*len(X))]])
Y_test = energies[int(0.8*len(X)):]

print("Training set: ")
print(X_train)
print(X_train.shape)
print(Y_train)
print(Y_train.shape)
print("\n")
print("Testing set: ")
print(X_test)
print(X_test.shape)
print(Y_test)
print(Y_test.shape)

# Loading the parameters from the params.yaml file
params = yaml.safe_load(open("./params.yaml"))["train"]
sigma = params["sigma"]
order = params["order"]
metric = params["metric"]

# Selecting and Defining the Matern Kernel for the KRR model
# Defining the Kernel K as a numpy array for the training set
K = matern_kernel(X_train, X_train, sigma, order, metric)

print("Kernel K: ")
print(K.shape)
print(K)

# KRR model using QML
# Adding a small lambda to the diagonal of the kernel matrix for "Ridge Regularization"
K[np.diag_indices_from(K)] += 1e-8

# Solving the linear system of equations using the Cholesky decomposition
alpha = cho_solve(K, Y_train)

print("Alpha: ")
print(alpha)
print(alpha.shape)

# Predicting the Test set results using the KRR model
# Calculating the kernel matrix between test and training set using same sigma
K_test_train = matern_kernel(X_test, X_train, sigma, order, metric)

# Calculating the predicted energies
Y_pred = np.dot(K_test_train, alpha)
print("Predicted Energies: ")
print(Y_pred)
print(Y_pred.shape)

# Calculating the mean absolute error
MAE = np.mean(np.abs(Y_pred - Y_test))
print("Mean Absolute Error: ", MAE)

# Plotting the loglog plot of the MAE vs Training set size
# Generating the list of MAE for learning curve
# Taking 10%, 20%, 30%, 40%, 50%, 60%, 70% of the training set
MAE_list = []
X_train_subset_size = []
for i in range(1,9):
    X_train_subset = X[:int(i*0.1*len(X))]
    Y_train_subset = np.array([mol.properties for mol in compounds[:int(i*0.1*len(X))]])
    sigma = sigma
    K = matern_kernel(X_train_subset, X_train_subset, sigma, order, metric)
    K[np.diag_indices_from(K)] += 1e-8
    alpha = cho_solve(K, Y_train_subset)
    K_test_train = matern_kernel(X_test, X_train_subset, sigma, order, metric)
    Y_pred = np.dot(K_test_train, alpha)
    MAE = np.mean(np.abs(Y_pred - Y_test))
    MAE_list.append(MAE)
    X_train_subset_size.append(X_train_subset.shape[0])
    print(X_train_subset.shape[0])
    print("MAE: ", MAE)

print("X_train_subset_size: \n", X_train_subset_size)
print("MAE List: \n", MAE_list)


# Creating the output folders ../output/plots and ../output/models if they don't exist
if not os.path.exists('./output/plots'):
    os.makedirs('./output/plots')
if not os.path.exists('./output/models'):
    os.makedirs('./output/models')

# Plotting the loglog plot of learning curve of MAE vs training set sizes
plt.figure(figsize=(10, 6))
plt.loglog(X_train_subset_size, MAE_list, 'o-')
# plt.grid(True)
plt.ylabel('MAE')
plt.xlabel('Training set size')
plt.title('Learning Curve: MAE vs Training Set Size')
plt.plot()
# saving the plot
plt.savefig('./output/plots/Learning_Curve.png')

# saving the metrics in a csv file
with open('./output/metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['sigma', 'order', 'metric', 'MAE'])
    writer.writerow([sigma, order, metric, MAE])

# saving the metrics in a json file
with open('./output/metrics.json', 'w') as f:
    json.dump({'sigma': sigma, 'order': order, 'metric': metric, 'MAE': MAE}, f)

# Saving the KRR model as a pickle file
with open('./output/models/KRR_model.pkl', 'wb') as f:
    pickle.dump([sigma, order, metric, alpha, K_test_train, K], f)

# Saving the devised dataset into pickle files for future use
if not os.path.exists('./output/dataset'):
    os.makedirs('./output/dataset')
# exporting the X_train, Y_train, X_test, Y_test into ../output/dataset as pickle files for using in test step
with open('./output/dataset/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('./output/dataset/Y_train.pkl', 'wb') as f:
    pickle.dump(Y_train, f)
with open('./output/dataset/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('./output/dataset/Y_test.pkl', 'wb') as f:
    pickle.dump(Y_test, f)
with open('./output/dataset/whole_dataset.pkl', 'wb') as f:
    pickle.dump(X, f)
