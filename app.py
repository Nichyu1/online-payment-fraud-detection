import numpy as np
import pandas as pd
import pickle
import streamlit as st

# --- Images and Icons ---
BANNER_URL = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80"
ILLUSTRATION_URL = "https://cdn.pixabay.com/photo/2017/01/31/13/14/online-2025931_1280.png"  # Payment illustration
SAFE_IMG = "https://cdn-icons-png.flaticon.com/512/190/190411.png"
FRAUD_IMG = "https://cdn-icons-png.flaticon.com/512/564/564619.png"
ANALYTICS_IMG = "https://cdn.pixabay.com/photo/2017/06/10/07/18/analytics-2389153_1280.png"  # Analytics illustration

# --- Sidebar ---
st.sidebar.image(BANNER_URL, use_container_width=True)
st.sidebar.title("About the Project")
st.sidebar.markdown("""
This app uses a machine learning model to detect online payment fraud.\
- **Model:** Random Forest\
- **Features:** Step, Type, Amount, Balances, Flagged\
- **Try single or batch prediction!**
""")
st.sidebar.markdown("---")
st.sidebar.header("Feature Explanations")
st.sidebar.markdown("""
- **Step (Time):** Time step of the transaction
- **Transaction Type:** PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- **Amount:** Transaction amount
- **Old/New Balance (Origin/Dest):** Sender/Receiver balances before/after
- **Is Flagged Fraud?:** Was this transaction flagged as fraud?
""")
st.sidebar.markdown("---")
st.sidebar.header("🎥 What is Payment Fraud?")
st.sidebar.video("https://www.youtube.com/watch?v=2z0kB5MWp1A")  # Example explainer video

# --- Header Banner & Illustration ---
st.image(ILLUSTRATION_URL, use_container_width=True)
st.markdown("""
# 🛡️ Online Payment Fraud Detection
Welcome! Enter transaction details below to check for fraud risk.
""")

# Load model and scaler
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# --- Transaction Types ---
TRANSACTION_TYPES = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
transaction_type_map = {k: v for v, k in enumerate(TRANSACTION_TYPES)}

# --- Single Transaction Prediction ---
st.markdown("## 📝 Single Transaction Prediction")
with st.form(key='fraud_form'):
    col1, col2 = st.columns(2)
    with col1:
        step_time = st.number_input('⏱️ Step (Time)', min_value=0, value=1, help="Time step of the transaction.")
        transaction_type = st.selectbox('💳 Transaction Type', TRANSACTION_TYPES, help="Type of transaction.")
        amount = st.number_input('💰 Amount', min_value=0.0, value=0.0, help="Transaction amount.")
        is_flagged_fraud = st.selectbox('🚩 Is Flagged Fraud?', [0, 1], help="Was this transaction flagged as fraud?")
    with col2:
        old_balance_origin = st.number_input('🏦 Old Balance (Origin)', min_value=0.0, value=0.0, help="Sender's balance before transaction.")
        new_balance_origin = st.number_input('🏦 New Balance (Origin)', min_value=0.0, value=0.0, help="Sender's balance after transaction.")
        old_balance_dest = st.number_input('🏦 Old Balance (Destination)', min_value=0.0, value=0.0, help="Receiver's balance before transaction.")
        new_balance_dest = st.number_input('🏦 New Balance (Destination)', min_value=0.0, value=0.0, help="Receiver's balance after transaction.")
    submit = st.form_submit_button('🔍 Predict')

    # Transaction summary card
    if submit:
        st.markdown("### 📋 Transaction Summary")
        summary_df = pd.DataFrame({
            'Feature': ['Step', 'Type', 'Amount', 'Old Bal Orig', 'New Bal Orig', 'Old Bal Dest', 'New Bal Dest', 'Flagged'],
            'Value': [step_time, transaction_type, amount, old_balance_origin, new_balance_origin, old_balance_dest, new_balance_dest, is_flagged_fraud]
        })
        st.table(summary_df)

        # Prepare input
        transaction_type_num = transaction_type_map.get(transaction_type, 0)
        input_data = (
            step_time,
            transaction_type_num,
            amount,
            old_balance_origin,
            new_balance_origin,
            old_balance_dest,
            new_balance_dest,
            is_flagged_fraud
        )
        input_data_as_np_array = np.asarray(input_data).reshape(1, -1)
        try:
            standard_data = scaler.transform(input_data_as_np_array)
            prediction = model.predict(standard_data)
            proba = model.predict_proba(standard_data)[0]
            st.markdown("---")
            if prediction[0] == 0:
                st.image(SAFE_IMG, width=100)
                st.success(f"""
                ### ✅ The transaction is **not fraudulent**.
                **Model confidence:** {proba[0]*100:.2f}%
                """)
            else:
                st.image(FRAUD_IMG, width=100)
                st.error(f"""
                ### 🚨 The transaction is **fraudulent**!
                **Model confidence:** {proba[1]*100:.2f}%
                """)
        except Exception as e:
            st.error(f"Error: {e}")

# --- Batch Prediction ---
st.markdown("## 📂 Batch Prediction (CSV Upload)")
st.info("Upload a CSV file with the same columns as the model expects. Example: step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFlaggedFraud")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
batch_predict = False
if uploaded_file is not None:
    if st.button("Predict on Uploaded CSV"):
        batch_predict = True
if uploaded_file is not None and batch_predict:
    try:
        df = pd.read_csv(uploaded_file)
        # Map transaction type to numeric
        if 'type' in df.columns:
            df['type'] = df['type'].map(transaction_type_map)
        # Fill missing columns if any
        for col in ['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']:
            if col not in df.columns:
                df[col] = 0
        X = df[['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        df['Prediction'] = np.where(preds==1, 'Fraudulent', 'Not Fraudulent')
        df['Fraud_Probability'] = probs[:,1]
        st.success(f"Predicted {sum(preds==1)} fraudulent out of {len(preds)} transactions.")
        st.dataframe(df.head(10))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")
        # --- Analytics ---
        st.markdown("### 📊 Batch Analytics")
        st.image(ANALYTICS_IMG, width=200)
        st.bar_chart(df['Prediction'].value_counts())
        st.bar_chart(df['type'].value_counts())
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Example Input Section ---
st.markdown("---")
st.markdown("### 🧪 Example Input for Testing")
st.code("""
Step (Time): 1
Transaction Type: TRANSFER
Amount: 181.00
Old Balance (Origin): 181.00
New Balance (Origin): 0.00
Old Balance (Destination): 0.00
New Balance (Destination): 0.00
Is Flagged Fraud?: 0
""", language="yaml")

#st.set_option('server.maxUploadSize', 1024)
