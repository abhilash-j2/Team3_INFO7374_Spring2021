import streamlit as st
import pandas as pd
import numpy as np

# NOTE : you can also pass a URL to streamlit run? This is great when combined with Github Gists. For example: 
# streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py

# Add title 
st.title("Streamlit Application")

# st.write()
# You can pass almost anything to st.write(): text, data, Matplotlib figures, Altair charts, and more.

# Writing a dataframe using st.write()
st.write("### Creating a table")
st.write(pd.DataFrame({
	'column_1' :[1,2,3,4,5,6,7,8,9,10,11],
	'column_2' :['a','b','c','d','e','f','g','h','i','j','k']
	}))

# NOTE : There are other data specific functions like st.dataframe() and st.table() that you can also use for displaying data.


# Magic Commands in streamlit
# You can also write to your app without calling any Streamlit methods. 
# Streamlit supports “magic commands,” which means you don’t have to use st.write() at all

"""
### Using Magic Command
Here's our first attempt at using data to create a table:
"""

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df


"""
### Line Chart
"""
chart_data = pd.DataFrame(
	np.random.randn(20, 3),
	columns = ["a", "b", "c"])

st.line_chart(chart_data)


"""
### Plot Map of San Francisco, with some random points
"""

map_data = pd.DataFrame(
	np.random.randn(1000,2) / [50,50] + [37.6, -122.4], 
	columns = ['lat', 'lon'])

st.map(map_data)

"""
### Adding Widgets for Interactivity

#### Checkboxes for Show/Hiding Data
"""
if st.checkbox('Show Dataframe'):
	st.line_chart(chart_data)


"""
#### Use Selectbox for optioons

Use st.selectbox to choose from a series.
"""


left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")