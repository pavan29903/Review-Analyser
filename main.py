import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}


model = load_model('simpleRnn_model.h5')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment:")


user_input = st.text_area("")


if st.button("Predict"):
    preprocessed_input = preprocess_text(user_input)
    
    prediction = model.predict(preprocessed_input)
    
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {prediction[0][0]:.4f}')

else:
    st.write("Please enter a review and click 'Predict' to see the sentiment analysis.")


