# Import library
import streamlit as st
import numpy as np
import scipy.stats as stats

# import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd


# from Home import dataframe


st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.subheader('Data Preprocessing')

##### Remove Outliers
st.subheader('A. Remove Outliers')
data=st.session_state["dataframe"]

fig = px.box(data,y=data.columns)
# Use x instead of y argument for horizontal plot
# fig.add_trace(go.Box(data['particle05_zone100']))

st.write ('Before Remove Outliers')
st.plotly_chart(fig,use_container_width=True)

st.write('Data Quantity is',data.shape)




z_scores = stats.zscore(data['particle05_zone100'])
# calculate z-scores of `df`

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
new_df = data[filtered_entries]
st.write ('After Remove Outliers')
fig1 = px.box(new_df,y=data.columns)


st.plotly_chart(fig1,use_container_width=True)

st.write('The data quantity after remove outliers of Particle 0.5 in Zone 100 is ',new_df.shape)

#### Data Value Over Time
st.write('Data Value Overtime')

fig2 = px.line(new_df,y='particle05_zone100')

st.plotly_chart(fig2,use_container_width=True)

## Function missing Value

def missing_statistics(df):    
    statitics = pd.DataFrame(df.isnull().sum()).reset_index()
    statitics.columns=['COLUMN NAME',"MISSING VALUES"]
    statitics['TOTAL ROWS'] = df.shape[0]
    statitics['% MISSING'] = round((statitics['MISSING VALUES']/statitics['TOTAL ROWS'])*100,2)
    return statitics

st.subheader('B. Resampling and Handling Missing Values')
new_df.index = pd.to_datetime(new_df.index)
sampled=new_df.resample('min').mean().interpolate(method='linear')

st.table(missing_statistics(sampled))

if 'sampled' not in st.session_state:
    st.session_state['sampled'] = sampled



