from sklearn.model_selection import train_test_split

# Features and target
X = df.drop("class", axis=1)
y = df["class"]

# Train-test split
#Split data into training and testing sets (70/30)
#Using stratiy=y to secure the fraud cases to split evenly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)