from scripts.data_loading import load_creditcard_data
from scripts.preprocessing import clean_data, rename_columns
# from src.modeling import train_model  ← add later
# from src.visualization import plot_results  ← add later

def main():
    print("🚀 Starting fraud detection pipeline...\n")

    df = load_creditcard_data()
    if df is not None:
        df_clean = clean_data(rename_columns(df))
        # model = train_model(df_clean)
        # plot_results(model)

    print("\n✅ Pipeline finished.")

if __name__ == "__main__":
    main()