#Import data
# Dataset not included in repo — download 'creditcard.csv' from Kaggle public databases and place in data directory

import pandas as pd
from pathlib import Path

def load_creditcard_data():
    """
    Loads the credit card fraud dataset from the data folder.
    Returns:
        pd.DataFrame: The loaded dataset
    """
    data_path = Path(__file__).resolve().parent.parent / 'data' / 'creditcard.csv'
    try:
        df = pd.read_csv(data_path)
        print("✅ Data loaded successfully.")
        print("First 10 rows of the data frame:")
        print(df.head(10))
        return df
    except FileNotFoundError:
        print("❌ Dataset not found. Please download 'creditcard.csv' from Kaggle and place it in the 'data/' folder.")
        return None


