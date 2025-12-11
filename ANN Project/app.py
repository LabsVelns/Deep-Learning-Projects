import streamlit as st
from keras.models import load_model
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = load_model('ann_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoder 
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the one-hot encoder
with open('OneHotGeo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)
    
st.title('Customer Churn Prediction')
# Input fields
input_data = {}

# User input
geography =        st.selectbox('Geography', onehot_encoder.categories_[0])
gender =           st.selectbox('Gender', label_encoder.classes_)
age =              st.slider('Age', 18, 92)
balance =          st.number_input('Balance')
credit_score =     st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure =           st.slider('Tenure', 0, 10)
num_of_products =  st.slider('Number of Products', 1, 4)
has_cr_card =      st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([geo_encoded_df, input_data.reset_index(drop=True)], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
