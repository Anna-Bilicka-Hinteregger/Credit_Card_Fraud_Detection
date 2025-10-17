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