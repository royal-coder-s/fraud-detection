# src/download_data.py
import kagglehub
import os

# Specify the directory where you want to save the data (within your project)
data_dir = "data/creditcardfraud"  # Or wherever you want to put it

# Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud", path=data_dir, force=True, unzip=True)  # force=True to overwrite if already downloaded, unzip=True to unzip the file after download.

print("Path to dataset files:", path)