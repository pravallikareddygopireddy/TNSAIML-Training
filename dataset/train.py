# Step 1: Import packages
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Step 2: Load the dataset
data = Path("dataset/credit_card_fraud_10k.csv")
df = pd.read_csv(data)


# Step 3: Exploratory Data Analysis (EDA)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())


# Step 4: Split features and target
x = df.drop(["transaction_id","is_fraud"], axis=1)
y = df["is_fraud"]

x = pd.get_dummies(x, dtype=int)

# Step 5: Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Train the classification model
model = LogisticRegression()

model.fit(x_train, y_train)


# Step 7: Make predictions
x_test_pred = model.predict(x_train)

training_acc=accuracy_score(x_test_pred,y_train)
print(training_acc)

joblib.dump(model,"model.joblib")
joblib.dump(list(x.columns),"feature_names.joblib")
print("Model and feature names saved successfully!")



