## Importing all the libraries first

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

## Load the imdb datasets and the word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## Load the pretrained model with the relu activation function

model=load_model('rnn/simple_rnn_imdb.h5')

## Creating our helper functions

## Helper functions

## Function to decode reviews

def decode_reviews(encoded_review):
  return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

## Function to preprocess use input

def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

## Prediction function presenting

def predict_sentiment(review):
  preprocessed_input=preprocess_text(review)
  prediction = model.predict(preprocessed_input)

  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

  return sentiment, prediction[0][0]

import streamlit as st

## Create the streamlit app

st.title('Moview review sentiment analysis with answer values as Positive or Negative')
st.write('Enter the review of a movie to classify it as Positive or Negative')

## User Input

user_input = st.text_area('Movie Review')

if st.button('Classify')