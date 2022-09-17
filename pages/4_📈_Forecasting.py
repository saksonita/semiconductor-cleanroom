# Import library
import streamlit as st
import scipy.stats as stats
import pandas as pd
import numpy as np
from pmdarima.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.express as px



import warnings
warnings.filterwarnings('ignore')

from scipy import stats

import tensorflow as tf


st.title ('Forecasting Particle 05 in Zone 100')


sampled=st.session_state["sampled"]
# sampled = sampled.set_index('time')

col = ['particle05_zone100']

## Train Test Divided 

train, test = train_test_split(sampled, train_size=0.8)

df_train = train[col]
df_test = test[col]

### Scaled
scaler = MinMaxScaler().fit(df_train)

train_scaled = scaler.transform(df_train)
test_scaled = scaler.transform(df_test)


### Prepared For model 

# Th input shape should be [samples, time steps, features]
def create_dataset (X, look_back = 1):
    Xs, ys = [], []
    
    for i in range(len(X)-look_back):
        v = X[i:i+look_back]
        Xs.append(v)
        ys.append(X[i+look_back])
        
    return np.array(Xs), np.array(ys)

X_train, y_train = create_dataset(train_scaled,30)
X_test, y_test = create_dataset(test_scaled,30)

print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape) 
print('y_test.shape: ', y_test.shape)
 


### Load Model

# Recreate the exact same model, including its weights and the optimizer
new_model_gru = tf.keras.models.load_model('models/model_gru.h5')

# Transform data back to original data space
y_test = scaler.inverse_transform(y_test)
y_train = scaler.inverse_transform(y_train)

# Make prediction
def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction

prediction_gru = prediction(new_model_gru)
print(prediction_gru)



# Calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()

    print(model_name + ':')
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    print('')


#### Plot using Plotly 

range_future = len(prediction_gru)

test_data = pd.DataFrame(y_test)
prediction = pd.DataFrame(prediction_gru)
df_train = df_train.reset_index()
new_join = pd.concat([df_train['time'],test_data, prediction], axis=1)
new_join.columns =['Date','Real Data','Prediction']


print(new_join.dropna())

all_data = new_join.dropna()

fig1 = px.line(all_data, x='Date', y=['Real Data','Prediction'],
              hover_data={'Date': "|%B %d, %Y %H:%M:%S"},
              title='Forecasting Results')



st.plotly_chart(fig1,use_container_width=True)



