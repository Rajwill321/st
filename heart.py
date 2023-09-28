#Importing the libraries
import numpy as np
import pickle
import streamlit as st
import gzip

# Path to the compressed file
compressed_file_path = 'heart_trained_model.sav.gz'

with gzip.open(compressed_file_path, 'rb') as compressed_file:
    #loading the saved model
    loaded_model = pickle.load(compressed_file)

#Creating a function for Prediction
def heartdisease_prediction (input_data):
	# changing the input data to a numpy array
	numpy_data= np.asarray (input_data)
	#Reshaping the numpy array as we are predicting for only on instance
	input_reshaped = numpy_data.reshape (1,-1)
	prediction = loaded_model.predict (input_reshaped)
	if (prediction[0] == 0):
		st.success ('The person does not have heart disease')
	else:
		st.warning ('The person have heart disease')


#Adding title to the page
st.title ('Heart disease prediction Web App')

#Getting the input data from the user
age = st.text_input ('Age in Years')
sex = st.text_input ('Sex : 1 – male, 0 – female')
cp = st.text_input ('Chest pain type')
trestbps = st.text_input ('Resting blood pressure in mm Hg')
chol = st.text_input ('Serum cholesterol in mg/dl')
fbs = st.text_input ('Fasting blood sugar > 120 mg/dl : 1 – true, 0 – false')
restecg = st.text_input ('Resting electrocardiographic results')
thalach = st.text_input ('Maximum heart rate achieved')
exang = st.text_input ('Exercise induced angina: 1 – yes, 0 – no')
oldpeak = st.text_input ('ST depression')
slope = st.text_input ('Slope')
ca = st.text_input ('Number of major vessels (0-3)')
thal = st.text_input ('Thal')

# code for Prediction
diagnosis = ' '
# creating a button for Prediction
if st.button ('Heart Disease Test Result'):
	diagnosis=heartdisease_prediction ([int(age),int(sex),int(cp),int(trestbps),int(chol),int(fbs),int(restecg),int(thalach),int(exang),int(oldpeak),int(slope),int(ca),int(thal)])
