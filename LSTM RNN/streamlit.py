import streamlit as st
import pickle 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the LSTM model

model=load_model('LSTM RNN/next_word_lstm.h5')

## Load the tokenizer

with open('LSTM RNN/tokenizer.pickle', 'rb') as handle:
  tokenizer=pickle.load(handle)

## Funciton to predict the next word

def predict_next_word(model, tokenizer, text, max_sequences_len):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequences_len:
    token_list=token_list[-(max_sequences_len-1):] ## Ensure the sequence length matches the max_sequences_len
  token_list=pad_sequences([token_list], maxlen=max_sequences_len-1, padding='pre')
  predicted=model.predict(token_list, verbose=0)
  predicted_word_index = np.argmax(predicted, axis=1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

## streamlit app 

st.title("Next Word Prediction with LSTM and Early Stopping")
input_text=st.text_input("Enter the sequence of the Words", "Ana de armas is my")
if st.button("Predict the next word"):
  max_sequence_len=model.input_shape[1]+1 ## Retrieve max sequence length from tokenizer 
  next_word=predict_next_word(model, tokenizer, input_text, max_sequence_len)
  st.write(f"Next word: {next_word}")