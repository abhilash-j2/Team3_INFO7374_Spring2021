import streamlit as st
import pandas as pd

st.write("""
	# My first app
	Hello *world!*
	""")

df = pd.read_csv("gender_submission.csv")
my_chart = st.line_chart(df)

number = st.slider("Pick a number", 0, 10)
date = st.date_input("Pick a date")


# Radio buttons
pets = ["dogs", "cats", "owls"]
st.radio("Pick a pet", pets)