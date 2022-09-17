# Import library
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.svm import OneClassSVM


## Sidebar

st.title('Anomaly Detection')

##### Remove Outliers
sampled=st.session_state["sampled"]
sampled = sampled.reset_index()

cols = ['humidity_zone100', 'particle05_zone100',
       'particle10_zone100', 'temperature_zone100', 'temperature_y',
       'wind_speed', 'humidity_y', 'steam_pressure', 'dew_point_temperature',
       'local_barometric', 'pm10Value', 'dp']


# Take useful feature and standardize them 

outliers_fraction = 0.01
data = sampled[cols]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
# train one class SVM 
model_svm =  OneClassSVM(nu=0.95 * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
data = pd.DataFrame(np_scaled)
model_svm.fit(data)
# add the data to the main  
sampled['anomaly'] = pd.Series(model_svm.predict(data))
sampled['anomaly'] = sampled['anomaly'].map( {1: 0, -1: 1} )
st.write(sampled['anomaly'].value_counts())

## Plot Anomaly

st.write ("CL-Central Limit (Mean) = 121")
st.write ("ULC-Upper Limit Control (Warning Line) = 340")

a = sampled.loc[sampled['anomaly'] == 1, ['time', 'particle05_zone100']] #anomaly
upper_only=a[a['particle05_zone100']>340]
fig2 = px.line(sampled, x='time', y='particle05_zone100',
              hover_data={'time': "|%B %d, %Y %H:%M:%S"},
              title='Anamoly')
fig2.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")
fig1 = px.scatter(upper_only, x="time", y="particle05_zone100", color="particle05_zone100")

fig3 = go.Figure(data=fig1.data + fig2.data)
fig3.add_hline(y=121, line_dash="dot",
              annotation_text="CL", 
              annotation_position="bottom right",
              annotation_font_size=20,
              annotation_font_color="blue"
             )
fig3.add_hline(y=340, line_dash="dot",
              annotation_text="ULC", 
              annotation_position="bottom right",
              annotation_font_size=20,
              annotation_font_color="red"
             )
st.plotly_chart(fig3,use_container_width=True)

if 'sampled' not in st.session_state:
    st.session_state['sampled'] = sampled