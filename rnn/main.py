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

model=load_model('./rnn/simple_rnn_imdb.h5')

## Creating our helper functions

## Helper functions

## Function to decode reviews

def decode_reviews(encoded_review):
  return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

## Function to preprocess use input

import re

def preprocess_text(text):

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.lower().split()

    encoded_review = []

    for word in words:

        if word not in word_index:
            encoded_review.append(2)
            continue

        index = word_index[word] + 3

        if index < 10000:
            encoded_review.append(index)
        else:
            encoded_review.append(2)

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

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

if st.button('Classify'):
  preprocessed_input = preprocess_text(user_input)

  ## making the prediction

  prediction = model.predict(preprocessed_input)
  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

  ## Display the result

  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]}')

else:
  st.write('Please enter a movie review.')
