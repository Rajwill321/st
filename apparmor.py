import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


def user_input_features():
    st.subheader("Transaction Details")
    cc_num = st.number_input('Insert Credit Card number')
    tran_amt = st.number_input('Amount in Credit Card Transaction')
    trans_date = st.date_input("Date of Credit Card Transaction (YYYY/MM/DD)", value="today")
    trans_time = st.time_input('Date of Credit Card Transaction', value="now")
    tran_category = st.selectbox('Transaction Category', ['grocery_pos', 'shopping_pos', 'shopping_net', 'gas_transport', 'home', 'kids_pets', 'personal_care', 'food_dining', 'entertainment', 'misc_pos', 'health_fitness', 'misc_net', 'grocery_net', 'travel'])
    
    data = {
        'cc_num': cc_num,
        'tran_amt': tran_amt,
        'date': trans_date,
        'time': trans_time,
        'category': tran_category
    }
    return pd.DataFrame(data, index=[0])


st.title('Credit Card Transaction input for Fraud Detection')

st.session_state.user_data = user_input_features()

if st.button("Predict"):
    st.write(st.session_state.user_data)
