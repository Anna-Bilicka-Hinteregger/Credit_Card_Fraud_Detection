from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import pandas as pd

def random_tree_model(X_train, y_train, X_test, y_test):
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)

    #Save model
    joblib.dump(model_rf, 'random_forest_model.pkl')
    print("💾 Model saved as 'random_forrest_model.pkl'")

    #Return metrics
    report = classification_report(y_test, y_pred, output_dict=False)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    #Feature importances
    feature_importance = pd.Series(model_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return report, roc_auc, cm, feature_importance

    return report, roc_auc, cm, feature_importance
