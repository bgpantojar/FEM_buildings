# FEM_buildings

This repository contains the codes for generating finite element models of masonry buildings. The hereby methodology was presented in the paper ["Automated image-based generation of finite element models for masonry buildings" by Pantoja-Rosero et., al. (2023)](https://doi.org/10.1007/s10518-023-01726-7)

<p align="center">
  <img src=docs/images/fem_01.png>
</p>

<p align="center">
  <img src=docs/images/fem_02.png>
</p>

<p align="center">
  <img src=docs/images/fem_03.png>
</p>

## How to use it? (Note: tested for ubuntu 18.04lts)

### 1. Clone repository

Clone repository in your local machine. All codes related with method are inside the `src` directory.

### 2. Download data and CNN weights

Example input data can be downloaded from [Dataset for automated image-based generation of finite element models for masonry buildings](https://doi.org/10.5281/zenodo.8094306). This datased contains 3 main folders (weights, data and results). Extract the folders `data/` and `weights/` and place them inside the repository folder.

#### 2a. Repository directory

The repository directory should look as:

```
FEM_buildings
└───data
└───docs
└───examples
└───src
└───weights
```

### 3. Environment

Create a conda environment and install python packages. At the terminal in the repository location.

`conda create -n FEM_buildings python=3.8.13`

`conda activate FEM_buildings `

`pip install -r requirements.txt`

### 4. Third party software

- Meshroom and Polyfit: The method needs as input Structure from Motion information and LOD2 model (see data folder) that are computed by [Meshroom](https://github.com/alicevision/meshroom) and [Polyfit](https://github.com/LiangliangNan/PolyFit) respectively. Please refeer to the links to know how to use their methodologies.

- FreeCAD: In addition to create the final 3D DADT building models, it is necessary [FreeCAD](https://www.freecadweb.org/downloads.php) python console and its methods. You can either download the appimage and extract their content as `freecad_dev` or download the folder here [freecad_dev](https://drive.google.com/file/d/1LvjPHkhyo_gdBkCyHqN6uEqLqCGaB3vG/view?usp=sharing). Place the folder `freecad_dev` in the repository location. The repository directory should look as:

```
FEM_buildings
└───data
└───docs
└───examples
└───freecad_dev
  └───usr
    └───bin
    └───...
└───src
└───weights
└───LICENSE
└───README.md
└───requirements.txt
└───.gitignore
```

- Matlab: the some of the scripts are written in matlab (it will be updated soon to python). Follow the next steps to make matlab usable from terminal:

  - Install matlab in your pc
  - Add the path of the Matlab's binary file to the environment file -> in the terminal write: `sudo vim /etc/environment` and add at the end :/usr/local/MATLAB/R2022b/bin" and save
  - in the terminal `export PATH=$PATH:/usr/local/MATLAB/R2022b/bin`

- Julia and Amaru: If in the examples the flag for modal_analysis is true, it need Amaru FEM software for the analysis with solid elements. Follow the next steps:
  - Install Julia
    - Download binaries from [Julia](https://julialang.org/downloads/) (Generic Linux binaries for x86)
    - Extract files
    - Copy the bin's folder path (/home/pantojas/julia-1.3.1-linux-x86_64/julia-1.3.1/bin)
    - open terminal and write `sudo vim /etc/environment`
    - Add at the final the :bin path (:/home/pantojas/julia-1.3.1-linux-x86_64/julia-1.3.1/bin) and save
    - in the terminal `export PATH=$PATH:/home/pantojas/julia-1.3.1-linux-x86_64/julia-1.3.1/bin`
  - Install [Amaru](https://github.com/NumSoftware/Amaru.jl)
    - open terminal and call julia `julia`
    - press the pkg key "]" and then write `dev https://github.com/NumSoftware/Amaru`
    - done, you are all set up.

### 5. Testing method with pusblished examples

Inside the folder `examples/` we have provide the input scripts that our algorithm needs. Two input scripts are necessary: `..._LOD3.py` and `..._FEM.py`. To run for instance the example `p2_00_School_FEM.py` simply open the terminal inside the src folder (with the environment activated) and write the next command:

`python ../examples/p2_00_School_FEM.py`

The algorithm first will create the LOD3 model and then postprocess it to generate the FEM models. Run the other examples similarly to the previous inline command.

`IMPORTANT` change the paths according your pc of: 1) line 3 in the file LOD3_builder.sh; 2) line 6 in the file MAIN_2_writeAnalyses.m; 3) line 3 in the file MAIN_3_processAnalyses.m

### 6. Creating your own digital twin as LOD3 model

Create a folder `your_example` inside the `data\` folder. Inside `your_example` create extra folders with the next structure:

```
FEM_buildings
└───data
  └───your_example
    └───images
      └───im1
    └───polyfit
    └───sfm
...
```

The methodology requires as input the next:

- sfm.json: file containing the sfm information (camera poses and structure). Add to the default `Meshroom` pipeline a node `ConverSfMFormat` and connect its input to the SfMData output from the node `StructureFromMotion`. In the node `ConverSfMFormat` options modify the SfM File Format to json. After running the modified `Meshroom` pipeline, this file is output in the folder `MeshroomCache/ConvertSfMFormat/a_folder_id/`. Copy that file inside the `your_example/sfm/`
- A registered view image for each facade containing the openings: For each facade, place one image in which the openings are visible in the folder `data/your_example/images/im1/`.
- polyfit.obj: use `Polyfit` pipeline either with the sparse or dense point cloud produced by the `Meshroom` pipeline. Note that it might be necessary to pre-process the point clouds deleting noisy points before running `Polyfit`. Save the output file as polyfit.obj or polyfit_dense.obj and place it in the folder `data/your_example/polyfit/`

Check the files of the data examples provided if neccessary to create the input data.

Finally create the two input scripts (`your_example_LOD3.py` and `your_example_FEM.py`) following the contents the given examples. Open the terminal inside the src folder (with the environment activated) and write the next command:

`python ../examples/your_example_FEM.py`

### 7. Results

The results will be saved inside `results` folder with the following structure:

```
FEM_buildings
└───results
  └───your_example
    └───EFM_analysis
    └───EFM_discretization
    └───FEM_solids
    └───LOD
...
```

#### 7.a Final repository directory

The repository directory after runing the medothology looks as:

```
FEM_buildings
└───data
└───docs
└───examples
└───freecad_dev
└───results
└───src
└───weights
└───LICENSE
└───README.md
└───requirements.txt
└───.gitignore
```

### 8. Citation

We kindly ask you to cite us if you use this project, dataset or article as reference.

Paper:

```
@article{Pantoja-Rosero2023c,
title = {Automated image-based generation of finite element models for masonry buildings},
journal = {Bulleting of Earthquake engineering},
year = {2023},
doi = {https://doi.org/10.1007/s10518-023-01726-7},
url = {https://link.springer.com/article/10.1007/s10518-023-01726-7},
author = {B.G. Pantoja-Rosero and R. Achanta and K. Beyer},
}
```

Dataset:

```
@dataset{Pantoja-Rosero2023c-ds,
  author       = {Pantoja-Rosero, Bryan German and
                  Achanta, Radhakrishna and
                  Beyer, Katrin},
  title        = {Dataset for automated image-based generation of finite element models for masonry buildings},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v.0.0},
  doi          = {10.5281/zenodo.8094306},
  url          = {https://doi.org/10.5281/zenodo.8094306}
}
```
