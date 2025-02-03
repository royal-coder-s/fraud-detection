# src/download_data.py
import os
import kaggle
import pandas as pd

# Get Kaggle credentials from environment variables
kaggle_username = os.environ.get("KAGGLE_USERNAME")
kaggle_key = os.environ.get("KAGGLE_KEY")

if not kaggle_username or not kaggle_key:
    raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set.")

os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

dataset_name = "mlg-ulb/creditcardfraud"
data_dir = "data/creditcardfraud"

os.makedirs(data_dir, exist_ok=True)

try:
    kaggle.api.dataset_download_files(dataset_name, path=data_dir, unzip=True, force=True)
    print(f"Dataset '{dataset_name}' downloaded to '{data_dir}'")

print("Path to dataset files:", path)
=======
    csv_file_path = os.path.join(data_dir, "creditcard.csv")
    df = pd.read_csv(csv_file_path)
    print("Data loaded into DataFrame.")

    print(df.head())
    print(df.info())

except Exception as e:
    print(f"Error downloading or loading dataset: {e}")
    exit(1)
