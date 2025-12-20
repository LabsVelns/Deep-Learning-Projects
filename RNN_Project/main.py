import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence


@st.cache_resource # Cache the model loading
def load_sentiment_model():
    return load_model("rnn_imdb_model.keras", compile=False)

model = load_sentiment_model()
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review): # Numbers to text
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.4 else 'Negative'
    return sentiment, prediction[0][0]

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to analyze its sentiment.")
user_input = st.text_area("Enter a movie review:", height=200)

if st.button("Analyze Sentiment"):
    sentiment, confidence = predict_sentiment(user_input)
    st.write(f"Sentiment: **{sentiment}**")
    st.write(f"Confidence: **{confidence:.2f}**")
else:
    st.write("Please enter a valid movie review.")