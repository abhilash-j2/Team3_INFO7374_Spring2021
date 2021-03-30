import streamlit as st
import pandas as pd
import pickle

# st.beta_set_page_config(layout="wide")
st.title('Multi Touch Attribution')
# st.markdown('This application is meant to **_assist_ _doctors_ _in_ diagnosing**, if a patient has a **_Heart_ _Disease_ _or_ not** using few details about their health')
 
@st.cache
def load_first_few(filepath):
    data = pd.read_feather(filepath).head(20)
    return data 

data_head = load_first_few("pickle_files/df5_head.feather")
# data_head.to_feather("pickle_files/df5_head.feather")

st.markdown("## Understanding the data ")
if st.checkbox("Show preprocessed dataset"):
    st.write(data_head)

st.markdown("## Journey lengths")
col1, col2 = st.beta_columns(2)
with col1:
    st.image("images/journey_lengths.png")
with col2:
    st.image("images/unique_journey.png")

st.markdown("## Journey stats")

@st.cache
def load_journey_stats():
    with open("pickle_files/journey_stats_agg.pkl","rb") as f:
        data = pickle.load(f)
    return data

journey_data = load_journey_stats()
st.table(journey_data)

st.markdown("## Exploration of Attribution Techniques")

techniques_ls = ['Logistic Regression (keras)', 'Last Touch Attribution','First Touch Attribution','Time Decaying Attribution','U-Shaped Attribution', "Linear Attribution"]
attribution_technique = st.selectbox("Attribution method", techniques_ls, index=0)


def showcase_technique(image_file, code_file):
    main_area.image("images/" + str(image_file), use_column_width=True)
    with open("codefiles/" + str(code_file),"r") as f:
        code_area.code(f.read())

code_area, main_area = st.beta_columns(2)
if attribution_technique == "Last Touch Attribution":
    showcase_technique("LTA.png","LTA.py")
elif attribution_technique == 'Logistic Regression (keras)':
    showcase_technique("keras_logreg.png","Logreg_keras.py")
elif attribution_technique == 'Time Decaying Attribution':
    showcase_technique("TDA.png","TDA.py")
elif  attribution_technique == 'U-Shaped Attribution':
    showcase_technique("Ushape.png","Ushape.py")
elif attribution_technique == "Linear Attribution":
    showcase_technique("LA.png","LA.py")
elif attribution_technique == 'First Touch Attribution':
    showcase_technique("FTA.png","FTA.py")


st.markdown("## Simulation Results")

simulation_df = pd.read_pickle("pickle_files/simulation_results.pkl")
#simulation_df.index=simulation_df["pitches"]

st.dataframe(simulation_df)
# cols_req = st.multiselect("Columns",techniques_ls)
st.markdown("<br/> <br/>",unsafe_allow_html=True)
st.line_chart(simulation_df)

st.markdown("""
## References
- https://blog.griddynamics.com/cross-channel-marketing-spend-optimization-deep-learning/
- https://github.com/ikatsov/tensor-house/blob/master/promotions/channel-attribution-lstm.ipynb
- https://pixelme.me/blog/attribution-models
""")