import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------
# 1. Define Deep Learning PyTorch Neural Network (Simple Loan MLP)
# -------------------------------------------------------------------
class SimpleLoanMLP(nn.Module):
    def __init__(self, input_dim=6):
        super(SimpleLoanMLP, self).__init__()
        # Simple multi-layer perceptron architecture:
        # Input layer -> Hidden Layer 1 (16 neurons) -> ReLU -> Hidden Layer 2 (8 neurons) -> ReLU -> Output (1 neuron) -> Sigmoid
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

def train_and_save():
    # -------------------------------------------------------------------
    # 2. Load and Preprocess Dataset
    # -------------------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv("loan_approval_dataset.csv")

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Calculate total assets value
    df['total_assets_value'] = (
        df['residential_assets_value'] +
        df['commercial_assets_value'] +
        df['luxury_assets_value'] +
        df['bank_asset_value']
    )

    # Select required features and target
    feature_cols = [
        'no_of_dependents',
        'income_annum',
        'loan_amount',
        'loan_term',
        'cibil_score',
        'total_assets_value'
    ]
    
    # Process target variable (Approved -> 1, Rejected -> 0)
    df['loan_status'] = df['loan_status'].str.lower().str.strip().map({'approved': 1, 'rejected': 0}).astype(int)

    X = df[feature_cols]
    y = df['loan_status']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert data to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # -------------------------------------------------------------------
    # 3. Train PyTorch Deep Learning Model
    # -------------------------------------------------------------------
    model = SimpleLoanMLP(input_dim=6)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 100
    print("Training PyTorch Neural Network model...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------
    # 4. Model Evaluation
    # -------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_preds_prob = model(X_test_tensor)
        test_preds = (test_preds_prob >= 0.5).float()
        accuracy = accuracy_score(y_test_tensor.numpy(), test_preds.numpy())
        print(f"Test Accuracy of Deep Learning Model: {accuracy * 100:.2f}%")

    # -------------------------------------------------------------------
    # 5. Save Model and Scaler
    # -------------------------------------------------------------------
    joblib.dump(scaler, "backend/scaler.pkl")
    torch.save(model.state_dict(), "backend/model.pth")
    print("PyTorch model state saved to 'backend/model.pth'")
    print("Scaler saved to 'backend/scaler.pkl'")

if __name__ == "__main__":
    train_and_save()
