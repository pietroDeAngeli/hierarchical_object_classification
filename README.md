Continual Hierarchical object recognition
==============

# Dataset
The dataset used in available [here](https://figshare.com/s/e14dd1861c775572eace)

# How to run the code

1 - Create a conda environment using the environment file and activate it (modified it becasue it's too old), you will need to install the modules:
- Cython
- libmr
- pytorch-ignite 
- transformers
- python-dotenv
manually due to a bug in that module. 

2 - Create a `.env` file in the root directory and add your Hugging Face token:
```env
HF_TOKEN=your_hf_token_here
```

3 - Run runexp.sh

The plots will appear in the folder `outputs/`

# Directories

## Inputs
These files contains model settings to test the system over different supervision "probablity". These are automatically updated by `build_hierarchy.py` to match the generated tree depth.

## Recsiam
It's the core of the project, contains the code to lead data, creating embeddings and implement the logics of the agnt (envoironment, policy...). 

 - init.py: make the package importable and set stuff
 - agent.py: Logic of the agent, policy of iteration and supervision
 - data.py: load the dataset, dataloader and spliting of the dataset (adapted for static images)
 - embeddings.py: computation and management of the embeddings (DINOv2 integrated)
 - models.py: wrapper and definitions of the models, feature extractor
 - memory.py / evm.py: implementation of the memory to identify the objects
 - sampling.py: sampling to build the batch
 - supervision.py: manage the labels, human intervention and labelling. 
 - cfghelpers.py / utils.py: helper of the configurations, I/O and utils
 - openworld.py: functionality for open world scenario (new objects detection)

## Scripts
 - build_hierarchy.py: builds the taxonomic tree, copies images, and updates experiment configs
 - fs2desc.py: generates a file descriptor of the dataset
 - embed_dataset.py: compute the embeddings of the dataset (pre-processing)
 - json_train.py: start the training using a .json found in the directory `inputs`
 - plot_hierarchy.py: Plots the hierarchy and saves in the directory `outputs` 

 # UPDATES
 - converted framework from Video sequences to static Image datasets.
 - integrated DINOv2-small feature extraction with automatic 224x224 interpolation.
 - fixed `list_collate` to properly stack image tensors into batches.
 - fixed `Agent.process_next` indexing to preserve 2D embedding shapes for memory updates.
 - automated experiment config updates (prob levels) based on taxonomy depth.
 - translated all internal comments and logs to English.
 - fixed `json_train.py` missing pickle import.
 - added `HF_TOKEN` management via python-dotenv.
 - changed libraries' versions
 - changed recsaim/model.py so that it returns a list (compatibility)
 - changed recsaim/agent.py to return a bool_ (numpy compatibility)
 - changed recsaim/evm.py so that it uses float istead of np.float (compatibility)
 - changed recsaim/(openworld, memory) for numpy compatibility 
 - changed json_train for compatibility