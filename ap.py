import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# Load the model
model = joblib.load('our_joblib_model')

# Title
st.title("Insurance Cost Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Convert input to model format
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_map = {"southwest": 1, "southeast": 2, "northwest": 3, "northeast": 4}
region = region_map[region]

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    prediction = model.predict(input_data)[0]
    st.write(f"The predicted insurance cost is: ${prediction:.2f}")
