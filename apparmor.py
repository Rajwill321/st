import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
import os
import random
from keras.models import load_model
import tensorflow as tf

# Load the saved model and preprocessors
category_encoder_file = 'category_app_encoder.pkl'
weekday_encoder_file = 'weekday_app_encoder.pkl'
trans_hr_encoder_file = 'trans_hr_app_encoder.pkl'
time_period_encoder_file = 'time_period_app_encoder.pkl'
customer_profile_file  = 'customer_profile.pkl'
customer_usage_file = 'customer_usage.pkl'
transaction_history_file = 'transaction.pkl'
scaler_file = 'standard_scaler_app_encoder.pkl'
woe_category_encoder = joblib.load('woe_category_encoder.pkl')

#Update the following based on the Pickle/ Keras file
trained_model_path = 'baseline_nn_model.keras'

daily_limit = 5000

def load_and_transform_unseen_data(df, col_name, encoder_filename):
    label_encoder, one_hot_encoder = joblib.load(encoder_filename)
    encoded_col = label_encoder.transform(df[col_name])
    encoded_col = encoded_col.reshape(-1, 1)
    one_hot_encoded = one_hot_encoder.transform(encoded_col).astype(int)
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=[f'{col_name}_{cls}' for cls in label_encoder.classes_])
    return one_hot_df

mapping_data = {
    'merchant_cat': ['Shopping POS', 'Grocery POS', 'Misc POS','Shopping Net','Grocery Net','Misc Net','Gas Transport','Home','Kids Pets','Personal Care','Food Dining','Entertainment','Health Fitness','Travel'],
    'model_value': ['shopping_pos', 'grocery_pos', 'misc_pos','shopping_net','grocery_net','misc_net','gas_transport','home','kids_pets','personal_care','food_dining','entertainment','health_fitness','travel']
}

mapping_df = pd.DataFrame(mapping_data)
fatal_error = False

if os.path.exists(trained_model_path):
    print('Loading the traind ML Model')
    loaded_model = load_model('baseline_nn_model.keras')
else:
    print('Please check the path of the trained model')
    fatal_error = True

if os.path.exists(customer_profile_file):
    # Load the DataFrame from the pickle file
    print('Loading customer_df')
    customer_df = pd.read_pickle(customer_profile_file)
else:
    customer_df = pd.DataFrame(columns=['cc_num','category','category_median_amt','category_woe','state_woe'])
    customer_df.to_pickle(customer_profile_file)
    customer_df = pd.read_pickle(customer_profile_file)

if os.path.exists(customer_usage_file):
    # Load the DataFrame from the pickle file
    print('Loading customer_usage_df')
    customer_usage_df = pd.read_pickle(customer_usage_file)
else:
    customer_usage_df = pd.DataFrame(columns=['cc_num','trans_date','accumulated_daily_usage','remaining_daily_limit','median_trans_amt','purchase_power_amt','med_daily','usage_today'])
    customer_usage_df.to_pickle(customer_usage_file)
    customer_usage_df = pd.read_pickle(customer_usage_file)

def map_dropdown_to_model(selected_value):
    default_value = 'shopping_net'  # Default value if not found
    model_value = mapping_df[mapping_df['merchant_cat'] == selected_value]['model_value'].values
    return model_value[0] if len(model_value) > 0 else default_value

def predict_fraud(input_data):
    global customer_df
    global customer_usage_df
    global daily_limit

    if fatal_error:
        return f'Error: ML Model Not found'
    try:
        # Perform any necessary data preprocessing on 'input_data' (e.g., date parsing)
        print('Input received')
        print(input_data)
        cc_num = input_data['cc_num'].iloc[0]
        amt = input_data['amt'].iloc[0]
        trans_date = input_data['trans_date'].iloc[0]
        trans_cat = input_data['category'].iloc[0]

        input_data['trans_time'] = pd.to_datetime(input_data['trans_time'], format='%H:%M:%S')
        input_data['trans_hr']   = input_data['trans_time'].dt.hour
        trans_hr_df = load_and_transform_unseen_data(input_data, 'trans_hr', trans_hr_encoder_file)
        input_data = pd.concat([input_data,trans_hr_df],axis=1)
        
        input_data['time_period'] = pd.cut(input_data['trans_hr'], bins=[0, 6, 12, 18, 22, 24], 
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'], right=False, ordered=False)
        time_period_df = load_and_transform_unseen_data(input_data, 'time_period', time_period_encoder_file)
        input_data = pd.concat([input_data,time_period_df],axis=1)

        input_data['trans_date'] = pd.to_datetime(input_data['trans_date'], format='%Y-%m-%d')
        input_data['weekday'] = input_data['trans_date'].dt.weekday
        weekday_df = load_and_transform_unseen_data(input_data, 'weekday', weekday_encoder_file)
        input_data = pd.concat([input_data,weekday_df],axis=1)

        cat_df = load_and_transform_unseen_data(input_data, 'category', category_encoder_file)
        input_data = pd.concat([input_data,cat_df],axis=1)

        #print("Post Enriching Date & Time Data")
        #print(input_data)
        mask = customer_df['cc_num'] == cc_num
        if mask.any():
            print('Record Found')
        else:
            #subset_columns = ['cc_num','category','category_median_amt','category_woe','state_woe','purchase_power_amt','median_trans_amt','med_daily']
            print("Record not found")
            random_index = random.randint(0, len(customer_df) - 1)
            random_row = customer_df.iloc[random_index]
            var_state_woe = random_row['state_woe']
            default_amt = amt
            print("Default Amount")
            print(default_amt)
            print(type(default_amt))
            for category in mapping_df['model_value']:
                category_df = customer_df[customer_df['category'] == category]
                random_row = category_df.sample(n=1)
                random_category_woe = random_row['category_woe'].values[0]
                random_category_median_amt = random_row['category_median_amt'].values[0]
                #if (input_data['category'] == category).any():
                new_row = {'cc_num': cc_num, 
                           'category': category,
                           'category_median_amt': random_category_median_amt,
                           'category_woe': random_category_woe,
                           'state_woe':var_state_woe,
                           'purchase_power_amt': default_amt * 1.00,
                           'median_trans_amt':default_amt * 1.00,
                           'med_daily':default_amt * 1.00 }
                new_row_df = pd.DataFrame([new_row])
                print("New Record being added")
                print(new_row_df)
                print(new_row_df.info())
                customer_df = pd.concat([customer_df, new_row_df], ignore_index=True)
                customer_df.to_pickle(customer_profile_file)
                customer_df = pd.read_pickle(customer_profile_file)

        cc_cond = (customer_df['cc_num'] == cc_num)  
        cat_cond = (customer_df['category'] == trans_cat) 
        combined_condition = cc_cond & cat_cond
        selected_rows = customer_df[combined_condition]
        if not selected_rows.empty:
            var_category_median_amt = selected_rows['category_median_amt'].iloc[0]
            var_category_woe = selected_rows['category_woe'].iloc[0]
        
        # Check if the combination of cc_num and trans_date already exists in the DataFrame
        mask = (customer_usage_df['cc_num'] == cc_num) & (customer_usage_df['trans_date'] == trans_date)
        if mask.any():
            print("Card & Date Combination Found")
            var_accumulated_daily_usage = customer_usage_df.loc[mask, 'accumulated_daily_usage'].values[0] + amt
            var_usage_today = customer_usage_df.loc[mask, 'usage_today'].values[0] + 1
            customer_usage_df.loc[mask, 'accumulated_daily_usage'] += amt
            customer_usage_df.loc[mask, 'usage_today'] += 1
        else:
        #tran_subset_columns = ['cc_num','trans_date','accumulated_daily_usage','usage_today']
            print("Card & Date Combination Not Found")
            var_accumulated_daily_usage = amt
            var_usage_today = 1
            new_daily_record = {'cc_num' : cc_num,
                                'trans_date': trans_date,
                                 'accumulated_daily_usage' : var_accumulated_daily_usage,
                                 'usage_today' : var_usage_today }
            new_daily_record_df = pd.DataFrame([new_daily_record])
            customer_usage_df = pd.concat([customer_usage_df, new_daily_record_df], ignore_index=True)
        
        customer_usage_df.to_pickle(customer_usage_file)
        customer_usage_df = pd.read_pickle(customer_usage_file)
        
        mask = customer_df['cc_num'] == cc_num
        if mask.any():
            input_data.at[0, 'med_daily'] = customer_df.loc[mask, 'med_daily'].values[0]
            input_data.at[0, 'state_woe'] = customer_df.loc[mask, 'state_woe'].values[0]
            input_data.at[0, 'median_trans_amt'] = customer_df.loc[mask, 'median_trans_amt'].values[0]
            input_data.at[0, 'purchase_power_amt'] = customer_df.loc[mask, 'purchase_power_amt'].values[0]
            #print("Post Enriching Customer Data")
            #print(input_data.info())

        input_data['usage_today'] = var_usage_today
        input_data['category_woe'] = var_category_woe
        input_data['category_median_amt'] = var_category_median_amt
        input_data['daily_limit'] = 5000
        input_data['accumulated_daily_usage'] = var_accumulated_daily_usage
        input_data['remaining_daily_limit'] = 5000 - var_accumulated_daily_usage
        input_data['over_median_amt'] = (input_data['amt'] > input_data['median_trans_amt']).astype(int)
        input_data['over_purchase_power'] = (input_data['amt'] > input_data['purchase_power_amt']).astype(int)
        
        print(input_data)
        #Scale the input Data
        app_scaler = joblib.load(scaler_file)
        columns_to_scale = ['amt','accumulated_daily_usage','remaining_daily_limit','purchase_power_amt','med_daily','category_median_amt','median_trans_amt','usage_today']
        scaled_data = app_scaler.transform(input_data[columns_to_scale])
        input_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale)
        input_data_scaled = pd.concat([input_scaled, input_data.drop(columns=columns_to_scale)], axis=1)
        #print(input_data_scaled)

        trained_feature_order = ['amt','accumulated_daily_usage','remaining_daily_limit','category_median_amt','median_trans_amt','purchase_power_amt','med_daily','over_median_amt','over_purchase_power','usage_today','trans_hr_0','trans_hr_1','trans_hr_2','trans_hr_3','trans_hr_4','trans_hr_5','trans_hr_6','trans_hr_7','trans_hr_8','trans_hr_9','trans_hr_10','trans_hr_11','trans_hr_12','trans_hr_13','trans_hr_14','trans_hr_15','trans_hr_16','trans_hr_17','trans_hr_18','trans_hr_19','trans_hr_20','trans_hr_21','trans_hr_22','trans_hr_23','time_period_Afternoon','time_period_Evening','time_period_Morning','time_period_Night','weekday_0','weekday_1','weekday_2','weekday_3','weekday_4','weekday_5','weekday_6','category_entertainment','category_food_dining','category_gas_transport','category_grocery_net','category_grocery_pos','category_health_fitness','category_home','category_kids_pets','category_misc_net','category_misc_pos','category_personal_care','category_shopping_net','category_shopping_pos','category_travel','state_woe','category_woe']
        input_data_for_model = input_data_scaled[trained_feature_order]

        # Use your trained machine learning model to make predictions
        # Replace 'model' with your actual trained model
        input_tensor = tf.convert_to_tensor(input_data_for_model)

        prediction = loaded_model.predict(input_tensor)
        print(prediction)
        prediction = (prediction > 0.5).astype(int)

        # Format the prediction result
        if prediction == 1:
            return 'Flagged as Fraud'
        else:
            return 'Not Flagged as Fraud'
    except Exception as e:
        # Handle any errors or exceptions gracefully
        return f'Error: {str(e)}'

    

def user_input_features():
    st.subheader("Transaction Details")
    cc_num = st.number_input('Insert Credit Card number')
    amt = st.number_input('Amount in Credit Card Transaction')
    trans_date = st.date_input("Date of Credit Card Transaction (YYYY/MM/DD)", value="today")
    trans_time = st.time_input('Date of Credit Card Transaction', value="now")
    tran_category = st.selectbox('Transaction Category', ['misc_net', 'grocery_pos', 'shopping_pos', 'shopping_net', 'gas_transport', 'home', 'kids_pets', 'personal_care', 'food_dining', 'entertainment', 'misc_pos', 'health_fitness', 'grocery_net', 'travel'])
    category = map_dropdown_to_model(tran_category)
    trans_date_time = pd.to_datetime(str(trans_date) + ' ' + str(trans_time))
    # Create a DataFrame with the user input
    input_data = pd.DataFrame({'cc_num': [cc_num],
                               'amt': [amt],
                               'trans_date': [trans_date],
                               'trans_time': [trans_time],
                               'category': [category],
                               'trans_date_time': [trans_date_time]})
    return input_data


st.title('Credit Card Transaction input for Fraud Detection')
#input_for_predict = tf.convert_to_tensor(user_input_features())
#st.session_state.user_data = input_for_predict
st.session_state.user_data = user_input_features()

if st.button("Predict"):
    prediction = predict_fraud(st.session_state.user_data)
    #st.write(st.session_state.user_data)
    st.write(prediction)