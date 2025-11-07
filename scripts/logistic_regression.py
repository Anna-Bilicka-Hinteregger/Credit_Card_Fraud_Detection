from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report, roc_auc_score

def logistic_regression(X_train, y_train, X_test, y_test):
    model_lg = LogisticRegression(solver='liblinear', random_state=42, C=0.1, max_iter=100)
    model_lg.fit(X_train, y_train)

    y_pred = model_lg.predict(X_test)

    # Save model
    joblib.dump(model_lg, 'logistic_regression_model.pkl')

    # Return metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred)

    return report, roc_auc