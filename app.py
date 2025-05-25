import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to clean input
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

# Sentiment emoji map
sentiment_map = {
    0: ("Negative", "😠"),
    1: ("Neutral", "😐"),
    2: ("Positive", "😊")
}

# Page config
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="✈️", layout="centered")

# UI layout
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown("Analyze the sentiment of airline-related tweets using a trained ML model.")

tweet = st.text_area("✍️ Enter a tweet:")

if st.button("Analyze"):
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vec)[0]
    
    label, emoji = sentiment_map[pred]
    st.markdown(f"### Sentiment: **{label}** {emoji}")
