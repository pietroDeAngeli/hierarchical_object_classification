Hierarchical Object Classification
==============

# Dataset
The dataset you find here is just a small subset of COCO (10 examples per class), optimization on the cover set will be necessary to scale to bigger dataset. 

# How to run the code

1 - Create a conda environment using the environment file and activate it, you will need to install the modules manually:
- Cython
- libmr
- pytorch-ignite 
- transformers
- python-dotenv
- check the yml file in the bottom.


2 - Create a `.env` file in the root directory and add your Hugging Face token:
```env
HF_TOKEN=your_hf_token_here
```

3 - Run runexp.sh

This will produce an online training of the hierarchy as in the original paper Hierarchial Object Learning.
The hierarchy is built dynamically in a continous learning setting.  

Pay attention to remove the results directory before running it, otherwise the file will be lazy and it will just produce the plots. 

# In progress
 - Evaluation metrics
 - Static hierarchy training (batch)
 - One root and one chil for each class (batch)
 - No hierarchy but a EVM for class
 - Vector storing of the embeddings (e.g. FAISS)