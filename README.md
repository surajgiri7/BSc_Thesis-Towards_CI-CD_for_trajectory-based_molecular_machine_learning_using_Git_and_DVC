# BSc_Thesis_DVC
### Thesis topic: **"Towards CI/CD for trajectory-based molecular machine learning using Git and DVC"**
#### Author: Suraj Giri 
#### Supervisor: Prof. Dr. Peter Zaspel
### **Preliminary Dataset** : a **.xyz** file with molecular trajectories of 10k C6H6 molecules and a **.dat** file with energies for those 10k molecules.

#### Major Tasks:
1. Developing **Kernel Ridge Regression in DVC Structure** 
2. Developing **Testing Software in DVC Structure**
3. Jupyter Notebooks to showcase features of DVC for ML

## **1. Kernel Ridge Regression in DVC Structure**
### **1.1. Data Preprocessing and Feature Engineering**
1. Divided the .xyz file into 10k individual .xyz files for each molecule.
2. Processing the trajectories into `Compounds` class in `QML` library.
3. Saving the `Compounds` objects into the `Pickle (.pkl)` files.

For Feature Engineering, we used the `Coulomb Matrix` representations.
Colomb Matrix can be calculated using formula:
$M_{ij} = \begin{cases}
                0.5Z_i^{2.4} & \text{if } i = j\\
                \frac{Z_iZ_j}{|\boldsymbol{R}_i - \boldsymbol{R}_j|} & \text{if } i \neq j
            \end{cases}$

where, $Z_i$ and $Z_j$ are the atomic numbers of the atoms $i$ and $j$, and $R_i$ and $R_j$ are the positions of the atoms $i$ and $j$. And $||\boldsymbol{R}_i - \boldsymbol{R}_j||$ is the euclidean distance between the $i^{th}$ and $j^{th}$ atoms.

### **1.2. Model Training**
1. Splitted the dataset into training and testing sets.
2. Assigned the corresponding energies to all the `Compounds`.
3. Firstly trained with `Gaussian Kernel` and then with `Matern Kernel`.
4. Finally, saved the trained models into the `Pickle (.pkl)` files.

### **1.3. Model Testing**
1. Loaded the trained models from the `Pickle (.pkl)` files.
2. Loaded the testing dataset which was already saved into the `Pickle (.pkl)` files.
3. Calcculated various evaluation metrics like `MAE` and `RMSE` for the trained models.
4. Finally, saved the results into various `metrics` files.

### **1.4. DVC Structure**
For tracking, versioning and managing our dataset we used `Google Drive` as remote storage. We utilized `DVC` to manage our dataset and `Git` to manage our code. With DVC, we tracked the dependencies, metrics, and results of our trained models, along with the Pickle (.pkl) files of our dataset and trained models, and also the configuration files such as YAML files and JSON files.

The DVC structure of our project is as follows:
```
BSc_Thesis_DVC
├── .dvc/
├── dataset/
│   ├── C6H6_molecules/
│   ├── C6H6.xyz
│   ├── E_def2-tzvp.dat
│   └── E_def2-tzvp_copy.dat
├── output/
│   ├── dataset/
│   ├── models/
│   ├── plots/
│   ├── test/
│   ├── metrics.csv
│   ├── metrics.json
│   └──prepared_data.pkl
├── src
│   ├── mini_python_scripts/
│   ├── prepare.py
│   ├── requirements.txt
│   ├── test.py
│   └── train.py
├── .dvcignore
├── .gitignore
├── dataset.dvc
├── DVC_features.ipynb
├── dvc.lock
├── dvc.yaml
├── params.yaml
└── README.md
```

### **Steps to reproduce the results:**
1. Clone this repository.
2. Install the dependencies using `pip install -r src/requirements.txt`.
3. Pull the dataset from the remote storage using `dvc pull`.
4. Reproduce the `dvc pipeline` using `dvc repro`.

**Note 1:** To access the dataset from remote storage, one needs to have access to the Google Drive folder where the dataset is stored and has to authenticate the access flow by following additional steps.

**Note 2:** Since QML is  not supported in Apple Silicon processors, it is advised to run the code in Intel processors.

One can also reproduce the dvc pipeline with changed parameters bychanging them in the `params.yaml` file and then running `dvc repro`.

After reproducing the whole dvc pipeline, the visual structure of the pipeline can be seen using `dvc dag` which would look like this:
```
        +-------------+    
        | dataset.dvc |    
        +-------------+    
          **        **     
        **            **   
       *                ** 
+---------+               *
| prepare |             ** 
+---------+           **   
          **        **     
            **    **       
              *  *         
           +-------+       
           | train |       
           +-------+       
+------+ 
| test | 
+------+ 
```

To add the stages of the dvc pipeline to the dvc.yaml file, which is the **Human Readable** file that contains the information about all the stages of the dvc pipeline, we used the following command:
```
dvc stage add ...
```

There are various similar commands in DVC which can be used to track the dependencies, metrics, and results of the trained models. Some of them are:
1. `dvc metrics show` : to show the metrics of the trained models.
2. `dvc metrics diff` : to show the difference between the metrics of the trained models after changing the parameters.
3. `dvc exp show` : to show all the parameters and metrics in the specific experiments of the trained models.
4. `dvc exp diff` : to show the difference between all the parameters and metrics in the specific experiments of the trained models after changing the parameters.
5. `dvc plots show` : to show the plots of the trained models.
6. `dvc plots diff` : to show the difference between the plots of the trained models after changing the parameters.

Finally, we can push the changes to the remote storage using `dvc push` and to the Git repository using `git push`.

