import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#Import data
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

#Adress class imbalance with SMOTE
smote = SMOTE(random_state=42)

#Applying SMOTE to training data
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print("\n--- Imbalance Resolved ---")
print("Resampled training Class Count (Balanced):\n", y_train_res.value_counts())
print("Resampled test Class Count (Balanced):\n", y_test.value_counts())
