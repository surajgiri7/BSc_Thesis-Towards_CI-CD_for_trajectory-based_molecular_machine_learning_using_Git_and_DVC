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

# Preprocessing imported data
from import_dataset import compounds, energies
print("Compounds: ")
print(compounds[0:10])

# Feature Engineering
# Generating Coulomb matrices for every molecule in the dataset
for mol in compounds:
    mol.generate_coulomb_matrix(size=12, sorting="row-norm")

# making a big 2D array with coloumb matrix of all the molecules
X = np.array([mol.representation for mol in compounds])
print(len(X))
print("Coloumb matrix for first element: \n",X[0])



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


# Selecting and Defining the Gaussian Kernel for the KRR model
# Defining the Kernel width
sigma = 25.0
order = 0
metric = 'l1'

# Defining the Kernel K as a numpy array for the training set
K = matern_kernel(X_train, X_train, sigma, order, metric)

print(K.shape)
print(K)


# KRR model using QML
# Adding a small lambda to the diagonal of the kernel matrix for "Ridge Regularization"
K[np.diag_indices_from(K)] += 1e-8

# Solving the linear system of equations using the Cholesky decomposition
from qml.math import cho_solve
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
    sigma = 1000.0
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
if not os.path.exists('./output/model'):
    os.makedirs('./output/model')

# Saving the KRR model as a pickle file inside ../output/model
model = {
    'alpha': alpha,
    'sigma': sigma,
    'X_train': X_train,
    'Y_train': Y_train,
    'X_test': X_test,
    'Y_test': Y_test,
    'Y_pred': Y_pred,
    'MAE': MAE,
    'MAE_list': MAE_list,
    'X_train_subset_size': X_train_subset_size,
    'Y_train_subset': Y_train_subset,
}

with open ('./output/model/KRR_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Saving the MAE list for each training set size as a csv file inside ../output/plots
with open('./output/metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Training Set Size', 'MAE'])
    for i, mae in enumerate(MAE_list):
        writer.writerow([X_train_subset_size[i], mae])


# Plotting the graph
# Plotting the loglog plot of learning curve of MAE vs training set sizes
plt.figure(figsize=(6, 6))
plt.loglog(X_train_subset_size, MAE_list, 'o-')
plt.xlabel('Training set size')
plt.ylabel('MAE')
plt.title('Learning Curve: MAE vs Training set size - LOGLOG PLOT')
# saving the plot
plt.savefig('./output/plots/Learning_Curve.png')
# plt.show()
