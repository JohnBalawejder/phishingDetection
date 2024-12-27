import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model("detect_phish.h5", compile=False)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_input(url, tokenizer, max_length=150):
    sequences = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(sequences, maxlen=max_length, padding="post")
    return padded

st.title("Phishing URL Detector")
st.write("Enter a URL below to determine if it is legitimate or a phishing attempt.")
user_input = st.text_input("Enter URL", placeholder="e.g., example.com")

if st.button("Predict"):
    if user_input.strip(): 
        processed_input = preprocess_input(user_input, tokenizer)

        prediction = model.predict(processed_input)
        prediction = prediction[0][0] 

        if prediction > 0.5:
            st.error(f"The URL is likely a phishing attempt! ⚠️ (Confidence: {prediction:.2%})")
        else:
            st.success(f"The URL appears to be legitimate. ✅ (Confidence: {1 - prediction:.2%})")
    else:
        st.warning("Please enter a URL to test.")