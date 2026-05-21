import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("Heart_Disease_Prediction.csv")

# Features and target
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Convert target labels
y = y.map({"Presence": 1, "Absence": 0})

# Convert categorical columns
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_model.pkl")

print("Model saved successfully")