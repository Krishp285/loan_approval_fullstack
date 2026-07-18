import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("loan-data.csv")


# Remove null values
clean_data = df.dropna()


# Select Required Columns
clean_data = clean_data[
    [
        "Dependents",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Loan_Status",
    ]
]


# Convert Dependents column
clean_data["Dependents"] = clean_data["Dependents"].replace("3+", "3").astype(int)


# Convert Target Column
clean_data["Loan_Status"] = (
    clean_data["Loan_Status"].astype(str).str.strip().str.lower().map({"y": 1, "n": 0})
)


# Features and Labels
X = clean_data[
    [
        "Dependents",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
]

y = clean_data["Loan_Status"]


# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Model
model = LogisticRegression()

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")


# Save Model
joblib.dump(model, "backend/model.pkl")

# Save Scaler
joblib.dump(scaler, "backend/scaler.pkl")

print("Model Saved Successfully")
