import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model

model = tf.keras.models.load_model('regression_model.h5')

## Load encoder and scaler

with open('label_encoder_gender.pkl', 'rb') as file:
  label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
  onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
  scaler = pickle.load(file)

## Streamlit app

st.title('Estimated salary prediction')

## User input

