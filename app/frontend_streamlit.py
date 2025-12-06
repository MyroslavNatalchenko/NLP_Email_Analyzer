import streamlit as st
import requests

st.set_page_config(page_title="Email spam recognizer", layout="centered")

st.title("Email spam recognizer")
st.write("Enter an email below to check if it is spam or no")

user_input = st.text_area("Email text", height=158, placeholder="Enter text from email here...")

API_URL = "http://127.0.0.1:8000/predict"

if st.button("Check Category"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            response = requests.post(API_URL, json={"text": user_input})

            if response.status_code == 200:
                result = response.json()
                is_spam = result["is_spam"]
                confidence = result["confidence"]

                st.divider()

                if is_spam:
                    st.error(f"SPAM")
                else:
                    st.success(f"NOT SPAM")

                st.caption(f"Confidence: {confidence:.2%}")
            else:
                st.error("Error connecting to the backend.")

        except Exception as e:
            st.error(f"Connection failed: {e}")