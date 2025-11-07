def clean_data(df):
    """
    Performs basic cleaning and inspection on the credit card fraud dataset.
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("🔍 Checking for missing values...")
    missing = df.isnull().sum()
    print(missing[missing > 0] if not missing.empty else "✅ No missing values.")

    print("\n📊 Summary statistics:")
    print(df.describe())

    print("\n🧪 Class distribution:")
    print(df['Class'].value_counts())

    return df
