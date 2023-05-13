# author: Suraj Giri
# BSc Thesis, CS, Contructor University

import os
import re
import qml
import numpy as np
import pickle

# Define the folder where the xyz files are located
molcules_folder = './dataset/C6H6_molecules/'

# Define the path to the energy .dat file
energy_file = './dataset/E_def2-tzvp.dat'

# Get a list of all the xyz filenames in the folder
xyz_files = [f for f in os.listdir(molcules_folder) if f.endswith('.xyz')]

# Sort the filenames based on their molecule number (assuming the filename format is "molecule_X.xyz")
xyz_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

# Load the energies from the .dat file
energies = np.loadtxt(energy_file)
print (energies)

# lisitng the molecules as compounds class
compounds = [qml.Compound(xyz=os.path.join(molcules_folder, xyz_file)) for xyz_file in xyz_files] 
print(compounds[0:10])

for mol in compounds:
    mol.properties = energies[compounds.index(mol)]
    # print(mol.name)
    # print(mol.name, mol.properties)

# energy_molecule = np.array([mol.properties for mol in compounds])
# print(energy_molecule)

# Feature Engineering
# Generating Coulomb matrices for every molecule in the dataset
for mol in compounds:
    # mol.generate_coulomb_matrix(size=12, sorting="row-norm")
    mol.generate_coulomb_matrix(size=12, sorting="unsorted")

print("Coloumb matrix of the first molecule: ")
print(compounds[0].representation)

# Saving the dataset into output/prepared_data.pkl
if not os.path.exists("./output"):
    os.makedirs("./output")

with open("./output/prepared_data.pkl", "wb") as f:
    pickle.dump([compounds, energies], f)
