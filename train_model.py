import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

dataset_path = "dataset"

X = []
y = []

# Load all gesture files
for file in os.listdir(dataset_path):

    if file.endswith(".csv"):

        gesture_name = file.replace(".csv", "")

        data = pd.read_csv(dataset_path + "/" + file, header=None)

        for _, row in data.iterrows():
            X.append(row.tolist())
            y.append(gesture_name)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "gesture_model.pkl")

print("Model saved successfully!")