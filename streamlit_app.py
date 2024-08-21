import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Load the pre-fitted scaler and models
scaler = joblib.load('scaler.pkl')
knn_model = joblib.load('knn_model.pkl')
lg_model = joblib.load('lg_model.pkl')

st.title("Diabetes Classifier")
url = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
st.write("### Dataset Used: [%s](%s)" % ("Pima Indians Diabetes Database", url))

col1, col2, col3, col4 = st.columns(4)

with col1:
    pregnancies = st.number_input("Pregnancies", step=1, min_value=0)
    insulin = st.number_input("Insulin", step=1, min_value=0)

with col2:
    glucose = st.number_input("Glucose", step=1, min_value=0)
    bmi = st.number_input("BMI", step=0.1, min_value=0.1, format="%0.1f")

with col3:
    blood_pressure = st.number_input("Blood Pressure", step=1, min_value=0)
    dpf = st.number_input("Diabetes Pedigree Function", step=0.01, min_value=0.00, max_value=1.00)

with col4:
    skin_thickness = st.number_input("Skin Thickness", step=1, min_value=0)
    age = st.number_input("Age", step=1, min_value=21)

user_data = {
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
}

user_df = pd.DataFrame(user_data)

classifier = st.selectbox("Model", ("KNN", "Logistic Regression"))


def predict(classifier, user_df):
    user_array = scaler.transform(user_df)
    if classifier == "KNN":
        result = knn_model.predict(user_array)[0]
    elif classifier == "Logistic Regression":
        result = lg_model.predict(user_array)[0]
    return result


if st.button("Predict"):
    result = predict(classifier, user_df)
    if result == 0:
        st.write("Not Diagnosed with Diabetes")
    elif result == 1:
        st.write("Diagnosed with Diabetes")
    else:
        st.write("uhh")
else:
    st.write("The result will appear here")
