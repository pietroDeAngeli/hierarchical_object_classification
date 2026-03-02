Continual Hierarchical object recognition
==============

# Dataset
The dataset used in available [here](https://figshare.com/s/e14dd1861c775572eace)

# How to run the code

1 - Create a conda environment using the environment file and activate it (modified it becasue it's too old), you will need to install the modules:
- Cython
- libmr
- pytorch-ignite 
manually due to a bug in that module. 

2 - Run runexp.sh

The plots will appear in the folder `outputs/`

# Directories

## Inputs
These files contains model settings to test the system over different supervision "probablity". 

## Recsiam
It's the core of the project, contains the code to lead data, creating embeddings and implement the logics of the agnt (envoironment, policy...). 

 - init.py: make the package importable and set stuff
 - agent.py: Logic of the agent, policy of iteration and supervision
 - data.py: load the dataset, dataloader and spliting of the dataset
 - embeddings.py: computation and management of the embeddings
 - models.py: wrapper and definitions of the models, feature extractor
 - memory.py / evm.py: implementation of the memory to identify the objects
 - sampling.py: sampling to build the batch
 - supervision.py: manage the labels, human intervention and labelling. 
 - cfghelpers.py / utils.py: helper of the configurations, I/O and utils
 - openworld.py: functionality for open world scenario (new objects detection)

## Scripts
 - fs2desc.py: generates a file descriptor of the dataset
 - pre_embed.py: compute the embeddings of the dataset (pre-processing)
 - json_train.py: start the training using a .json found in the directory `inputs`
 - plot_hierarchy.py: Plots the hierarchy and saves in the directory `outputs` 

 # UPDATES
 - changed libraries' versions
 - changed recsaim/model.py so that it returns a list (compatibility)
 - changed recsaim/agent.py to return a bool_ (numpy compatibility)
 - changed recsaim/evm.py so that it uses float istead of np.float (compatibility)
 - changed recsaim/(openworld, memory) for numpy compatibility 