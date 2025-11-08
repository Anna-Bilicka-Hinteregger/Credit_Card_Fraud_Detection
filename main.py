from scripts.data_loading import load_creditcard_data
from scripts.preprocessing import clean_data, rename_columns
from scripts.smote_split import smote_split
from scripts.logistic_regression import logistic_regression
from scripts.random_tree_model import random_tree_model
import seaborn as sns
import matplotlib.pyplot as plt
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
    report, roc_auc, cm = logistic_regression(X_train, y_train, X_test, y_test)
    print("\n📋 Logistic Regression Report:")
    print(report)
    print("✅ ROC-AUC Score:", roc_auc)
    print("\n🧩 Confusion Matrix: \n", cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    #Random Forest model
    report, roc_auc, cm, feature_importance = random_tree_model(X_train, y_train, X_test, y_test)
    print("\n📋 Random Forrest Classification Report:")
    print(report)
    print("✅ ROC-AUC Score:", roc_auc)
    print("\n🧩 Confusion Matrix: \n", cm)
    print("\n🌟 Top Features:")
    print(feature_importance.head(10))

    print("\n✅ Pipeline finished.")


# Logistic Regression Confusion Matrix
plot_confusion_matrix(cm, title="Logistic Regression Confusion Matrix")

# Random Forest Confusion Matrix
plot_confusion_matrix(cm, title="Random Forest Confusion Matrix")

# Feature Importance Plot
plot_feature_importance(feature_importance, top_n=10)

if __name__ == "__main__":
    main()