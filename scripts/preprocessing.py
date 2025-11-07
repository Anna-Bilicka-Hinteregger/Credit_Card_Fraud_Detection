def clean_data(df):
    """
    Performs basic cleaning and inspection on the credit card fraud dataset.
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("🔍 Checking for missing values...")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("✅ No missing values found.")
    else:
        print("⚠️ Missing values detected:")
        print(missing)

    print("\n📊 Summary statistics:")
    print(df.describe())

    print("\n🧪 Class distribution:")
    print(df['Class'].value_counts())

    return df
