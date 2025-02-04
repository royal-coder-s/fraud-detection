# src/preprocess_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data():
    data = pd.read_csv('data/creditcardfraud/creditcard.csv')
    print("Dataset loaded successfully!")
    print(f"Shape of the dataset: {data.shape}")
    return data

# Perform exploratory data analysis (EDA)
def perform_eda(data):
    print("\nExploratory Data Analysis (EDA):")
    
    # Check for missing values
    print("\nMissing values:\n", data.isnull().sum())
    
    # Class distribution
    print("\nClass distribution:\n", data['Class'].value_counts())
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=data)
    plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
    plt.savefig('../reports/class_distribution.png')
    plt.close()
    
    # Summary statistics
    print("\nSummary statistics:\n", data.describe())

# Preprocess the data
def preprocess_data(data):
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save the processed data
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)
    
    print("\nData preprocessing completed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

# Main function
def main():
    data = load_data()
    perform_eda(data)
    preprocess_data(data)

if __name__ == "__main__":
    main()