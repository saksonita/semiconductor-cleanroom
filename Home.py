# Import library
import streamlit as st
import pandas as pd
import time
import pandas_profiling

from streamlit_pandas_profiling import st_profile_report

## Page configuration

st.set_page_config(
    page_title="cleanroom_analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",

)

## Mainpage
st.title('Semiconductor Cleanroom Particle Data Analysis', anchor="title")
st.header(':bar_chart: About the Data')

uploaded_file = st.file_uploader("Upload the CSV file")
    
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    dataframe = dataframe.drop('Unnamed: 0',1)
    dataframe = dataframe.set_index('time')
        # Initialization the session state to store data frame
    
    pr = dataframe.profile_report()
    st.dataframe(dataframe.head(21))
    st_profile_report(pr)

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = dataframe

