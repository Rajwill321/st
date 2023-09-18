import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model and preprocessors
model = tf.keras.models.load_model('mushroom_model.h5')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.title('Mushroom Edibility Classifier')

def user_input_features():
    st.subheader("Cap Features")
    cap_shape = st.selectbox('Cap Shape', ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
    cap_surface = st.selectbox('Cap Surface', ['fibrous', 'grooves', 'scaly', 'smooth'])
    cap_color = st.selectbox('Cap Color', ['brown', 'yellow', 'white', 'gray', 'red', 'pink', 'cinnamon', 'purple', 'buff', 'green'])

    bruises = st.selectbox('Bruises', ['yes', 'no'])
    odor = st.selectbox('Odor', ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'])

    st.subheader("Gill Features")
    gill_attachment = st.selectbox('Gill Attachment', ['attached', 'descending', 'free', 'notched'])
    gill_spacing = st.selectbox('Gill Spacing', ['close', 'crowded'])
    gill_size = st.selectbox('Gill Size', ['broad', 'narrow'])
    gill_color = st.selectbox('Gill Color', ['black', 'brown', 'gray', 'pink', 'white', 'chocolate', 'purple', 'red', 'buff', 'green', 'yellow', 'orange'])

    st.subheader("Stalk Features")
    stalk_shape = st.selectbox('Stalk Shape', ['enlarging', 'tapering'])
    stalk_root = st.selectbox('Stalk Root', ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'missing'])
    stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
    stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
    stalk_color_above_ring = st.selectbox('Stalk Color Above Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
    stalk_color_below_ring = st.selectbox('Stalk Color Below Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])

    st.subheader("Veil and Ring Features")
    veil_type = st.selectbox('Veil Type', ['partial', 'universal'])
    veil_color = st.selectbox('Veil Color', ['brown', 'orange', 'white', 'yellow'])
    ring_number = st.selectbox('Ring Number', ['none', 'one', 'two'])
    ring_type = st.selectbox('Ring Type', ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'])
    
    st.subheader("Miscellaneous Features")
    spore_print_color = st.selectbox('Spore Print Color', ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])
    population = st.selectbox('Population', ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
    habitat = st.selectbox('Habitat', ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])

    data = {
        'cap_shape': cap_shape,
        'cap_surface': cap_surface,
        'cap_color': cap_color,
        'bruises': bruises,
        'odor': odor,
        'gill_attachment': gill_attachment,
        'gill_spacing': gill_spacing,
        'gill_size': gill_size,
        'gill_color': gill_color,
        'stalk_shape': stalk_shape,
        'stalk_root': stalk_root,
        'stalk_surface_above_ring': stalk_surface_above_ring,
        'stalk_surface_below_ring': stalk_surface_below_ring,
        'stalk_color_above_ring': stalk_color_above_ring,
        'stalk_color_below_ring': stalk_color_below_ring,
        'veil_type': veil_type,
        'veil_color': veil_color,
        'ring_number': ring_number,
        'ring_type': ring_type,
        'spore_print_color': spore_print_color,
        'population': population,
        'habitat': habitat
    }

    return pd.DataFrame(data, index=[0])

user_data = user_input_features()

user_data_encoded = pd.get_dummies(user_data, drop_first=True)
missing_cols = set(X.columns) - set(user_data_encoded.columns)
for col in missing_cols:
    user_data_encoded[col] = 0
user_data_encoded = user_data_encoded[X.columns]
user_data_scaled = scaler.transform(user_data_encoded)

prediction = model.predict(user_data_scaled)
result = encoder.inverse_transform([int(prediction[0])])

st.write(f"The mushroom is predicted to be {result[0]}.")
