import streamlit as st
import joblib
from utils import clean_text  

# Load model and vectorizer
model = joblib.load("fake_news_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit App UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title(" Fake News Detection App")
st.markdown("Enter a news article text below to check if it's **Real** or **Fake**:")

# Text input
user_input = st.text_area("Paste your news content here:", height=200)

if st.button(" Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to make prediction.")
    else:
        # Clean and vectorize input
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0][prediction] * 100
        
        # Display result
        if prediction == 1:
            st.success(f" Real News (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"Fake News (Confidence: {confidence:.2f}%)")
