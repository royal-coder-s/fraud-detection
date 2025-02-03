# src/download_data.py
import kagglehub
import os

data_dir = "data/creditcardfraud"  # Or wherever you want to put it

os.makedirs(data_dir, exist_ok=True)

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud", path=data_dir, force=True, unzip=True) 

print("Path to dataset files:", path)
