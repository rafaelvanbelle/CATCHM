# CATCHM
Code repository for the paper: CATCHM: A novel network-based credit card fraud detection approach using node representation learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5654760.svg)](https://doi.org/10.5281/zenodo.5654760)

## Requirements:

The code has been tested under Python 3.6. The required packages can be installed using the following
command:

``$ pip install -r requirements.txt``

To guarantee that you have the right package versions, you can use Anaconda to set up a virtual environment and install the dependencies from ``requirements.txt``.


## Running the code

### Step 1: Download the dataset
The data used in the paper is confidential and cannot be made publicly available. However, a transaction fraud dataset is available on Kaggle. 
The structure and characteristics are very similar to the data from our paper. 

You can download the dataset from Kaggle:
https://www.kaggle.com/ranjeetshrivastav/fraud-detection-dataset

Make sure to unzip and store the 'transactions' folder someplace easily accessible.

### Step 2:

The demo code is available as a Jupyter Notebook. 
There are two notebooks:

- deepwalk_demo.ipynb
- pagerank_demo.ipynb

The first notebook contains the deepwalk approach with inductive mean pooling extension. The embeddings are used to train an XGBoost classification model. 
The second notebook contains the inductive pagerank approach [2] and serves as a benchmark. More details on this benchmark can be found in [2].

Both notebook files can be opened with Jupyter Notebook

``jupyter notebook``

## References

[1] Van Belle Rafael, De Weerdt Jochen. "A novel network-based credit card fraud detection approach using noderepresentation learning."

[2] Van Vlasselaer, Véronique, Cristián Bravo, Olivier Caelen, Tina Eliassi-Rad, Leman Akoglu, Monique Snoeck, and Bart Baesens. 2015. “APATE: A Novel Approach for Automated Credit Card Transaction Fraud Detection Using Network-Based Extensions.” Decision Support Systems 75 (July): 38–48.
