import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Email spam recognizer", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('../model_building/model.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model \n Error: {e}")
    st.stop()

st.title("Email spam recognizer")
st.write("Enter an email below to check if it is spam or no")

user_input = st.text_area("Email text", height=158, placeholder="Enter text from email here...")

if st.button("Check Category"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        input_data = tf.constant([user_input], dtype=tf.string)

        prediction_score = model.predict(input_data)[0][0]

        st.divider()

        if prediction_score >= 0.5:
            st.error(f"SPAM")
            st.caption(f"Confidence: {prediction_score:.2%}")
        else:
            st.success(f"NOT SPAM")
            st.caption(f"Confidence: {(1 - prediction_score):.2%}")