# %% [markdown]
# Loan Application Deep Learning Model (PyTorch Multi-Layer Perceptron)

# %%
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# %% 1. Load Dataset
dataset_path = 'loan_approval_dataset.csv'
df = pd.read_csv(dataset_path)

# Clean column names
df.columns = df.columns.str.strip()

# Calculate total assets value
df['total_assets_value'] = (
    df['residential_assets_value'] +
    df['commercial_assets_value'] +
    df['luxury_assets_value'] +
    df['bank_asset_value']
)

# Select relevant features and target
df = df[['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'total_assets_value', 'loan_status']]

# Process Target Column (Approved -> 1, Rejected -> 0)
df['loan_status'] = df['loan_status'].str.lower().str.strip().map({'approved': 1, 'rejected': 0}).astype(int)

print("Dataset Info:")
print(df.info())
print("\nSample Data:")
print(df.head())

# %% 2. Features and Target Split
X = df[['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'total_assets_value']]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to PyTorch FloatTensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# %% 3. Define Simple Neural Network (Multi-Layer Perceptron)
class SimpleLoanMLP(nn.Module):
    def __init__(self, input_dim=6):
        super(SimpleLoanMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = SimpleLoanMLP(input_dim=6)
print("\nDeep Learning Model Architecture:")
print(model)

# %% 4. Train the Deep Learning Model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == epochs:
        print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

# %% 5. Evaluate Model
model.eval()
with torch.no_grad():
    predictions_prob = model(X_test_tensor)
    predictions = (predictions_prob >= 0.5).float()
    accuracy = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
    print(f"\nAccuracy of Deep Learning PyTorch Model: {accuracy * 100:.2f}%")

# %% 6. Predict on New Application Sample
new_application = pd.DataFrame([{
    'no_of_dependents': 2,
    'income_annum': 5000000,
    'loan_amount': 2000000,
    'loan_term': 36,
    'cibil_score': 750,
    'total_assets_value': 10000000
}])

new_app_scaled = scaler.transform(new_application)
new_app_tensor = torch.tensor(new_app_scaled, dtype=torch.float32)

with torch.no_grad():
    prob = model(new_app_tensor).item()
    pred_status = "Approved" if prob >= 0.5 else "Rejected"

print(f"\nSample Prediction for New Application: {pred_status}")
print(f"Approval Probability: {prob:.4f}")

# %% 7. Save Model State and Scaler
joblib.dump(scaler, "backend/scaler.pkl")
torch.save(model.state_dict(), "backend/model.pth")
print("\nModel weights saved to backend/model.pth")
print("Scaler saved to backend/scaler.pkl")