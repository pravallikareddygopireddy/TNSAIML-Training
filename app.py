import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳🔍",
    layout="wide"
)

model=joblib.load("model.joblib")
feature_names=joblib.load("feature_names.joblib")

st.title("💳🔍 Welcome to Credit Card Fraud Detection Application")

#sidebar
st.sidebar.header('Transaction Details')

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0,value=100.0)
transaction_hour = st.sidebar.slider("Transaction Hour",min_value=0,max_value=23,value=12)
merchant_category = st.sidebar.selectbox("Merchant Category", ["Food", "Travel", "Shopping", "Grocery", "Online", "Other"])
foreign_transaction = st.sidebar.selectbox("Foreign Transaction?",[0, 1],format_func=lambda x: "Yes" if x == 1 else "No")
location_mismatch = st.sidebar.selectbox("Location Mismatch?",[0, 1],format_func=lambda x: "Yes" if x == 1 else "No")
device_trust_score = st.sidebar.slider("Device Trust Score",min_value=0.0,max_value=1.0,value=0.8)
velocity_last_24h = st.sidebar.number_input("Transactions in Last 24 Hours",min_value=0,value=1)
cardholder_age = st.sidebar.number_input("Cardholder Age",min_value=18,max_value=120,value=25)

input_data={
        "amount": amount,
        "transaction_hour": transaction_hour,
        "foreign_transaction":foreign_transaction,
        "location_mismatch": location_mismatch,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
        "merchant_category": merchant_category
}
input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(
    columns=feature_names,
    fill_value=0
)
input_df = input_df.reindex(columns=feature_names, fill_value=0)


st.subheader("💳 Transaction Details")
st.dataframe(input_df,use_container_width=True)
prediction = model.predict(input_df)
if st.button("predict",type="primary"):
    prediction=model.predict(input_df)

st.subheader("prediction result")

if prediction[0] == 1:
    st.write("Fraudulent transaction detected!")
else:
    st.write("Legitimate transaction detected.")

