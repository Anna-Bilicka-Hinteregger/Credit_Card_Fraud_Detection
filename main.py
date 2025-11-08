from scripts.data_loading import load_creditcard_data
from scripts.preprocessing import clean_data, rename_columns
from scripts.smote_split import smote_split
from scripts.logistic_regression import logistic_regression
from scripts.random_tree_model import random_tree_model
from scripts.visualization import plot_confusion_matrix, plot_feature_importance

def main():
    print("🚀 Starting fraud detection pipeline...\n")

    #Load the data and clean it
    df = load_creditcard_data()
    if df is not None:
        df_clean = clean_data(rename_columns(df))
        df = df_clean

    #Split the data into training and testing sets, using smote split method
    X_train, X_test, y_train, y_test = smote_split(df)
    print("\n--- Imbalance Resolved ---")
    print("Resampled training Class Count (Balanced):\n", y_train.value_counts())
    print("Resampled test Class Count (Balanced):\n", y_test.value_counts())

    #Logistic Regression model
    report, roc_auc, cm_lr = logistic_regression(X_train, y_train, X_test, y_test)
    print("\n📋 Logistic Regression Report:")
    print(report)
    print("✅ ROC-AUC Score:", roc_auc)

    #Random Forest model
    report, roc_auc, cm_rf, feature_importance = random_tree_model(X_train, y_train, X_test, y_test)
    print("\n📋 Random Forrest Classification Report:")
    print(report)
    print("✅ ROC-AUC Score:", roc_auc)

    #Confusion matrix saved to a file path
    plot_confusion_matrix(cm_lr, title="Logistic Regression Confusion Matrix", save_path="confusion_matrix_logreg.png")
    plot_confusion_matrix(cm_rf, title="Random Forest Confusion Matrix", save_path="confusion_matrix_rf.png")
    #Feature importance
    plot_feature_importance(feature_importance, top_n=10, save_path="feature_importance_rf.png")

    print("\n✅ Pipeline finished.")




if __name__ == "__main__":
    main()