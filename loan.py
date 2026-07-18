# %%
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('D:\\SoftCoding_practice\\ml\\loan_approval_dataset.csv')
df.head()
df.info()

# %%
df.columns
df = df[[' no_of_dependents',' income_annum',' loan_amount',' loan_term',' cibil_score', ' residential_assets_value', ' commercial_assets_value',
       ' luxury_assets_value', ' bank_asset_value',' loan_status']]

df.info()
df['total_assets_value'] = df[' residential_assets_value'] + df[' commercial_assets_value'] + df[' luxury_assets_value'] + df[' bank_asset_value']
df.info()
df.head()


# %%

df = df[[' no_of_dependents',' income_annum',' loan_amount',' loan_term',' cibil_score','total_assets_value',' loan_status']]
df.info()
df.head()
df[' loan_status'].value_counts()

# %%
df[' loan_status'] = df[' loan_status'].str.lower().str.strip().map({'approved': 1, 'rejected': 0}).astype(int)
df.info()
df

# %%
X=df[[' no_of_dependents',' income_annum',' loan_amount',' loan_term',' cibil_score', 'total_assets_value']]
Y=df[' loan_status']
X_train , X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=True)
X_train.info()

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy:.2f}")

# %%
new_application = pd.DataFrame([
    {
        ' no_of_dependents': 2,
        ' income_annum': 50000,
        ' loan_amount': 200000,
        ' loan_term': 36,
        ' cibil_score': 750,
        'total_assets_value': 100000
    }])
new_application_scaled = scaler.transform(new_application)
prediction = model.predict(new_application_scaled)
probability = model.predict_proba(new_application_scaled)
print(prediction , probability)
print(f"Prediction for the new loan application: {'Approved' if prediction[0] == 1 else 'Rejected'}")
print(f"Probability of approval: {probability[0][1]:.2f} , Probability of rejection: {probability[0][0]:.2f}")


# Save Model
joblib.dump(model, "backend/model.pkl")

# Save Scaler
joblib.dump(scaler, "backend/scaler.pkl")