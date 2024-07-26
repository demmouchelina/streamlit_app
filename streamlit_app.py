import streamlit as st
import requests

st.title('Random Forest Classifier')

# Input features
feature1 = st.number_input('Feature 1')
feature2 = st.number_input('Feature 2')
# Add inputs for all features required by the model

if st.button('Predict'):
    features = [feature1, feature2]  # Add all feature inputs
    response = requests.post('http://127.0.0.1:5000/predict', json={'features': features})
    prediction = response.json()['prediction']
    st.write(f'Prediction: {prediction}')
