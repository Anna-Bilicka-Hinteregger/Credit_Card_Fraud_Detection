import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

#Import data
# Dataset not included in repo — download 'creditcard.csv' from Kaggle public databases and place in project root
df = pd.read_csv('creditcard.csv')

#Prints 10 rows
print("First 10 rows of the data frame:")
print(df.head(10))

#Prepare data - x and y split
x = df.drop('Class', axis=1) #Features
y = df['Class'] #Target

#Split data into training and testing sets (70/30)
#Using stratiy=y to secure the fraud cases to split evenly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

#Original distribution for comparison with SMOTE method
print("Original training class distribution:\n", y_train.value_counts())

#Adress class imbalance with SMOTE
smote = SMOTE(random_state=42)

#Applying SMOTE to training data
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print("\n--- Imbalance Resolved ---")
print("Resampled training Class Count (Balanced):\n", y_train_res.value_counts())
print("Resampled test Class Count (Balanced):\n", y_test.value_counts())

#Logistic Regression model
model = LogisticRegression(solver='liblinear', random_state=42, C=0.1, max_iter=100)

#Train the model
model.fit(x_train_res, y_train_res)

#Evaluate
y_pred = model.predict(x_test)

# Save Logistic Regression model
joblib.dump(model, 'logistic_regression_model.pkl')

print("\n--- Model Performance Report ---")
print("\nClassification Report (Key for Fraud Detection):\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix (Raw Counts): \n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

#Probability of positive class, fraud
y_proba = model.predict_proba(x_test)[:, 1]

#Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

#Plot the curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

#Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(x_train_res, y_train_res)
rf_pred = rf_model.predict(x_test)
print(classification_report(y_test, rf_pred))
# Save Random Forest model
joblib.dump(rf_model, 'random_forest_model.pkl')



#ROC curves for both models
lr_proba = model.predict_proba(x_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
roc_auc_lr = auc(fpr_lr, tpr_lr)

rf_proba = rf_model.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='Logistic Regression (AUC = %0.2f)' % roc_auc_lr)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Logistic Regression vs Random Forest')
plt.legend(loc='lower right')
plt.show()

