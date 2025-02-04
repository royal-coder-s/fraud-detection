import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler

data_dir = "data/creditcardfraud"
csv_file_path = os.path.join(data_dir, "creditcard.csv")

try:
    df = pd.read_csv(csv_file_path)
    print("Data loaded successfully.")
    print(f"Shape of the dataset: {df.shape}")
    print(df.head())

    # 1. Basic Statistics for Numerical Features:
    print("\nDescriptive Statistics for Numerical Features:")
    print(df.describe())

    # 2. Information about Data Types and Missing Values:
    print("\nData Types and Missing Values:")
    print(df.info())

    # 3. Class Distribution:
    print("\nClass Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)

    # 4. Visualize Class Distribution:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
    plt.ylabel('Number of Transactions')
    plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    file_path = os.path.join(reports_dir, 'class_distribution.png')
    print(f"Saving plot to: {file_path}")
    plt.savefig(file_path)
    plt.close()

    plt.show()

    # 5. Check for Missing Values:
    print("\nMissing Values:")
    print(df.isnull().sum())

    # 6. Distribution of Transaction Amount and Time:
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    amount_val = df['Amount'].values
    time_val = df['Time'].values

    sns.distplot(amount_val, ax=ax[0], color='r')
    ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
    ax[0].set_xlim([min(amount_val), max(amount_val)])

    sns.distplot(time_val, ax=ax[1], color='b')
    ax[1].set_title('Distribution of Transaction Time', fontsize=14)
    ax[1].set_xlim([min(time_val), max(time_val)])

    file_path = os.path.join(reports_dir, 'amount_time_distribution.png')
    print(f"Saving plot to: {file_path}")
    fig.savefig(file_path)
    plt.close(fig)

    plt.show()

    # 7. Scaling Amount and Time:
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    print("\nAmount and Time Scaled:")
    print(df.head())  # Print the first few rows after scaling

except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}. Make sure you've run the download script.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit(1)