import pickle
from keras.preprocessing.sequence import pad_sequences
import streamlit as st
from keras.models import load_model


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


model = load_model('next_word_predictor_lstm.h5')

def predic_next_word(model, text, tokenizer, max_seq_len):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_seq_len-1, padding='pre')
    pred = model.predict(padded, verbose=0)
    predicted_word_index = pred.argmax(axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

st.title("Next Word Predictor")
st.header("Enter a phrase to predict the next word")
user_input = st.text_input("Enter a phrase:")
if st.button('Predict'):
    max_seq_len = model.input_shape[1] + 1
    next_word = predic_next_word(model, user_input, tokenizer, max_seq_len)
    st.write(f'The predicted next word is: {next_word}')