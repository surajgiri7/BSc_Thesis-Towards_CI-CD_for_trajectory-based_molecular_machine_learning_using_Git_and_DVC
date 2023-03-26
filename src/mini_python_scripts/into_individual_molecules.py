# author: Suraj Giri
# BSc Thesis, CS, Contructor University

import os

# Define the path to the input and output files
input_file = './dataset/C6H6.xyz'
output_folder = './dataset/C6H6_molecules/'
os.makedirs(output_folder, exist_ok=True) # Create the output folder if it doesn't already exist

# Open the input file for reading
with open(input_file, 'r') as f:
    # Initialize counters for the molecule index and the atom index
    molecule_index = 0
    atom_index = 0
    # Loop over each line in the input file
    for line in f:
        line = line.rstrip() #rstrip() removes trailing whitespace
        # If the line starts with a number, it is the number of atoms in the next molecule
        if line.isdigit():
            # Increment the molecule index and reset the atom index
            molecule_index += 1
            atom_index = 0
            # Define the filename for the current molecule
            filename = f'C6H6_{molecule_index}.xyz'
            # Define the path to the output file for the current molecule
            output_file = os.path.join(output_folder, filename)
            # Open the output file for writing
            with open(output_file, 'w') as g:
                # Write the number of atoms to the output file
                g.write(line+'\n')
                # Write the comment line to the output file
                g.write(next(f))
        else:
            # Increment the atom index
            atom_index += 1
            # Write the atom coordinates to the output file
            with open(output_file, 'a') as g:
                g.write(line+'\n')



# following code is for splitting the file into individual molecules where the top two lines are removed.
# import os
# # Define the path to the input file and create a directory to store output files
# input_path = '../C6H6.xyz'
# output_dir = '../molecules_separate/'
# os.makedirs(output_dir, exist_ok=True)

# # Read in the input file
# with open(input_path, 'r') as f:
#     contents = f.read()

# # Split the contents into individual molecule representations
# molecule_list = contents.strip().split('\n')

# # Save each molecule as a separate file in xyz format
# for i, molecule in enumerate(molecule_list):
#     # Skip the first molecule if it is empty (due to the separator at the beginning of the file)
#     if not molecule:
#         continue
#     molecule = molecule.strip().split('\n')
#     molecule = '\n'.join([line.strip() for line in molecule])
#     # Save the molecule as a separate file in xyz format
#     filename = f'molecule_{i+1}.xyz'
#     output_path = os.path.join(output_dir, filename)
#     with open(output_path, 'w') as f:
#         f.write(molecule)
