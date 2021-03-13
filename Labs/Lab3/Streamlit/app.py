import streamlit as st
import os 
import pandas as pd


st.title('Heart Disease Diagnosis Assistant')
st.markdown('This application is meant to **_assist_ _doctors_ _in_ diagnosing**, if a patient has a **_Heart_ _Disease_ _or_ not** using few details about their health')


df = pd.read_csv('heart.csv')

if st.checkbox('Show me Training Data'):
	st.dataframe(df)

st.markdown('Please **Enter _the_ _below_ details** to know the results -')

age = st.text_input(label='Age')

gender_ls = ['Male', 'Female']
sex = st.selectbox('Gender', gender_ls)

cp_ls = ['Typical Angina', 'Atypical Angina', 'Non-anginal pain', 'Asymptomatic']
cp = st.multiselect('Chest pain Type', cp_ls)

restbp = st.slider('Resting Blood Pressure', 0, 220, 120)

chol = st.slider('Serum Cholestoral in mg/dl', 0, 600, 150)

fbs_ls = ['fasting blood sugar > 120 mg/dl', 'fasting blood sugar < 120 mg/dl']
fbs = st.selectbox('Fasting Blood Sugar (>126 mg/dL signals diabetes)', fbs_ls)



if st.button('Check Diagnosis'):

	st.header('A Machine Learning Model would predict this') 