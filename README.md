# 🏦 Deep Learning Loan Approval Prediction System

A full-stack Deep Learning web application for loan approval prediction built with **PyTorch**, **FastAPI**, and **Streamlit**.

This project utilizes a **Multi-Layer Perceptron (MLP)** Neural Network written in **PyTorch** to predict whether a loan application will be **Approved** or **Rejected**, along with the exact approval probability score.

---

## 🏗️ Project Architecture

```text
┌─────────────────┐      HTTP POST      ┌─────────────────┐     PyTorch Tensor     ┌────────────────────────┐
│  Streamlit UI   │ ─────────────────>  │ FastAPI Backend │ ───────────────────>   │ Deep Learning PyTorch  │
│ (User Interface)│ <─────────────────  │   REST Server   │ <───────────────────   │  Model (model.pth)     │
└─────────────────┘    JSON Response    └─────────────────┘    Inference (0.0-1.0) └────────────────────────┘
```

---

## ✨ Features

- 🧠 **PyTorch Deep Learning Model**: Custom Multi-Layer Perceptron (MLP) binary classifier.
- ⚡ **FastAPI Backend**: Asynchronous RESTful API serving model inference.
- 🎨 **Streamlit Frontend UI**: Modern, intuitive interface for submitting applicant parameters and viewing results.
- 📊 **Feature Preprocessing & Scaling**: Integrated `StandardScaler` pipeline for asset value feature engineering and tensor conversion.
- 📈 **Probability Score**: Returns calculated approval probability (0.0 to 1.0) along with the final decision status.

---

## 🛠️ Tech Stack

### 1. Deep Learning & Machine Learning
- **PyTorch** (`torch.nn`, `torch.optim`): Model architecture, forward propagation, loss optimization (`BCELoss`), and binary prediction.
- **Scikit-learn**: Feature scaling (`StandardScaler`) and evaluation metrics.
- **Pandas & NumPy**: Data ingestion, manipulation, and asset feature computation.
- **Joblib**: Scaler serialization.

### 2. Backend API
- **FastAPI**: Asynchronous web framework.
- **Uvicorn**: High-performance ASGI server.
- **Pydantic**: Request payload validation.

### 3. Frontend UI
- **Streamlit**: Interactive web dashboard.
- **Requests**: HTTP Client communication with backend API endpoints.

---

## 🧠 Neural Network Model Architecture (`SimpleLoanMLP`)

The neural network is built using PyTorch's `nn.Sequential` with the following layer specification:

```text
Input Features (6) ──> [Linear: 6 -> 16] ──> [ReLU] ──> [Linear: 16 -> 8] ──> [ReLU] ──> [Linear: 8 -> 1] ──> [Sigmoid] ──> Approval Probability
```

| Layer | Type | Input Dim | Output Dim | Activation Function |
| :--- | :--- | :--- | :--- | :--- |
| **Input -> Hidden 1** | `nn.Linear` | 6 | 16 | `nn.ReLU()` |
| **Hidden 1 -> Hidden 2** | `nn.Linear` | 16 | 8 | `nn.ReLU()` |
| **Hidden 2 -> Output** | `nn.Linear` | 8 | 1 | `nn.Sigmoid()` |

### Input Features (6 Parameters):
1. `no_of_dependents`: Number of dependents (integer)
2. `income_annum`: Annual income in currency units (integer)
3. `loan_amount`: Total loan amount requested (integer)
4. `loan_term`: Loan duration in months/years (integer)
5. `cibil_score`: Applicant CIBIL credit score (300 - 900)
6. `total_assets_value`: Combined total valuation of residential, commercial, luxury, and bank assets

---

## 📁 Project Structure

```text
Loan_Application_Model/
│
├── backend/
│   ├── app.py               # FastAPI backend server with PyTorch inference pipeline
│   ├── model.pth            # PyTorch model trained weights (state_dict)
│   ├── scaler.pkl           # Saved StandardScaler model object
│   └── requirements.txt     # Backend Python dependencies
│
├── frontend/
│   ├── streamlit_app.py     # Streamlit interactive UI application
│   └── requirements.txt     # Frontend Python dependencies
│
├── loan_approval_dataset.csv # Primary loan application training dataset
├── train_model.py           # Deep learning model training & serialization script
├── loan.py                  # Standalone training script & test bench
├── main.py                  # Entry utility script
└── README.md                # Detailed project documentation
```

---

## 🚀 Quickstart & Setup Guide

### Step 1 — Prerequisites & Environment Setup

Ensure you have Python 3.9+ installed.

#### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2 — Install Dependencies

#### Install Backend Requirements
```bash
cd backend
pip install -r requirements.txt
```

#### Install Frontend Requirements
```bash
cd ../frontend
pip install -r requirements.txt
```

---

### Step 3 — Train the Deep Learning Model

To train the PyTorch model from scratch and export `model.pth` and `scaler.pkl` to the `backend/` directory, run:

```bash
cd ..
python train_model.py
```

*Expected Output:*
```text
Loading dataset...
Training PyTorch Neural Network model...
Epoch [20/100] - Loss: 0.3120
Epoch [40/100] - Loss: 0.2150
Epoch [60/100] - Loss: 0.1780
Epoch [80/100] - Loss: 0.1560
Epoch [100/100] - Loss: 0.1410
Test Accuracy of Deep Learning Model: ~95.00%
PyTorch model state saved to 'backend/model.pth'
Scaler saved to 'backend/scaler.pkl'
```

---

### Step 4 — Launch the FastAPI Backend Server

Navigate to the `backend` folder and start the server:

```bash
cd backend
uvicorn app:app --reload
```

- **API Base URL**: `http://127.0.0.1:8000`
- **Swagger Interactive Docs**: `http://127.0.0.1:8000/docs`

---

### Step 5 — Launch the Streamlit Frontend Interface

Open a new terminal window, activate the virtual environment, and run:

```bash
cd frontend
streamlit run streamlit_app.py
```

- **Frontend URL**: `http://localhost:8501`

---

## 📡 API Reference & Usage

### 1. Health Check Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**:
```json
{
  "message": "Loan Approval Deep Learning API is running successfully"
}
```

### 2. Predict Loan Status Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

#### Example Request Body
```json
{
  "no_of_dependents": 2,
  "income_annum": 5000000,
  "loan_amount": 2000000,
  "loan_term": 36,
  "cibil_score": 750,
  "total_assets_value": 10000000
}
```

#### Example Response Body
```json
{
  "prediction": "Approved",
  "approval_probability": 0.9542
}
```

---

## 🛠️ Troubleshooting & Notes

- **Model or Scaler File Missing**: If `model.pth` or `scaler.pkl` are not found inside `backend/`, execute `python train_model.py` at the project root level.
- **Port Conflicts**: If port 8000 is occupied, launch FastAPI with a custom port: `uvicorn app:app --reload --port 8001` and update `http://127.0.0.1:8001/predict` in `frontend/streamlit_app.py`.

---

## 🌐 Deploying to Render

This repository is pre-configured for seamless deployment on [Render](https://render.com).

### Method 1: Render Blueprint Deployment (Recommended)

1. Push this project repository to **GitHub** or **GitLab**.
2. Log into your [Render Dashboard](https://dashboard.render.com).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically read [`render.yaml`](file:///d:/SoftCoding_practice/Loan_Application_Model/render.yaml) and create both services:
   - `loan-approval-backend` (FastAPI REST API)
   - `loan-approval-frontend` (Streamlit Web Interface)
6. Click **Apply**. Both services will build and deploy automatically!

---

### Method 2: Manual Web Service Deployment on Render

If you prefer setting up services manually on Render:

#### 1. Deploy the Backend Web Service (`loan-approval-backend`)
- **Service Type**: Web Service
- **Environment**: Python 3
- **Build Command**: `pip install -r backend/requirements.txt && python train_model.py`
- **Start Command**: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.10.0`
- Note down your deployed backend URL (e.g. `https://loan-approval-backend.onrender.com`).

#### 2. Deploy the Frontend Web Service (`loan-approval-frontend`)
- **Service Type**: Web Service
- **Environment**: Python 3
- **Build Command**: `pip install -r frontend/requirements.txt`
- **Start Command**: `streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.10.0`
  - `BACKEND_URL`: `https://loan-approval-backend.onrender.com` (Your backend service URL from step 1)


