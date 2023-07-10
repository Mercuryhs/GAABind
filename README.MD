# GAABind
Code and trained model for our paper **GAABind: A Geometry-Aware Attention-Based
Network for Accurate Protein-Ligand Binding Pose
and Binding Affinity Prediction**

![Overall FrameWork](Figure1_overall.png)


## Installation dependencies
Our model is tested in Linux with the following packages:
- PyTorch == 1.12.0
- RDkit == 2021.09.5
- Numpy == 1.23.5
- Pandas == 1.5.2
- tqdm
- os



## Dataset
The processed dataset of PDBBind and SARS-CoV-2 main protease can be downloaded from the [google drive](https://drive.google.com/file/d/1uZDKhqBrNSvke29Wozq7TzwzwWYrzbU1/view?usp=drive_link).



## Usage
**Predict your own molecules with the trained model.**
### Step 1: Preprocess the dataset using the following format:
```shell
example_data/
├── Mpro-x11271
│   ├── ligand.txt
│   ├── pocket.txt
│   └── receptor.pdb
└── Mpro-x12715
    ├── ligand.txt
    ├── pocket.txt
    └── receptor.pdb
```
The ligand file can be in .sdf, .mol2, or .mol format, or you can provide the ligand's SMILES representation in a .txt file. The pocket.txt contains the residues of binding pockets, each residue is named by the chain, residue number, and three-letter abbreviation as follows:
```
A_166_GLU
A_49_MET
A_143_GLY
A_181_PHE
A_145_CYS
...
```
**Run the following command to preprocess the dataset**
```shell
python preprocess.py --data_path [dataset_path]
```


### Step 2: Predict the dataset using the following command
```shell

python predict.py --data_dir [dataset_path] --save_path [save_path] 
```
### Step 3: Generate the ligand pose 
```shell
python docking/docking.py --predict_path [save_path]
```



## Reference
```
Under Review
```