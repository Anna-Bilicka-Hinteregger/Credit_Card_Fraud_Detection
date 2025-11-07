from scripts.data_loading import load_creditcard_data
from scripts.preprocessing import clean_data, rename_columns
from scripts.smote_split import smote_split
# from src.modeling import train_model  ← add later
# from src.visualization import plot_results  ← add later

def main():
    print("🚀 Starting fraud detection pipeline...\n")

    df = load_creditcard_data()
    if df is not None:
        df_clean = clean_data(rename_columns(df))
        df = df_clean
        # model = train_model(df_clean)
        # plot_results(model)

    X_train, X_test, y_train, y_test = smote_split(df)
    print("\n--- Imbalance Resolved ---")
    print("Resampled training Class Count (Balanced):\n", y_train.value_counts())
    print("Resampled test Class Count (Balanced):\n", y_test.value_counts())

    print("\n✅ Pipeline finished.")

if __name__ == "__main__":
    main()