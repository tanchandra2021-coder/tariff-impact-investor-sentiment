import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load model + tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Tariff Social Media Impact on Investor Sentiment")
text = st.text_area("Paste a tariff-related tweet or social media post:")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Preprocess
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]

        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_idx = np.argmax(probs)
        sentiment = sentiment_labels[sentiment_idx]

        # Map probabilities into "impact scores"
        confidence = float(probs[sentiment_idx])
        if sentiment == "Positive":
            impact = f"↑ Optimism in investors (+{round(confidence*10,2)} sentiment points)"
        elif sentiment == "Negative":
            impact = f"↓ Concern among investors (-{round(confidence*10,2)} sentiment points)"
        else:
            impact = f"~ Stable reaction (±{round((1-confidence)*5,2)} sentiment points)"

        # Display
        st.subheader("Sentiment Probabilities:")
        for label, p in zip(sentiment_labels, probs):
            st.write(f"{label}: {p:.4f}")

        st.subheader("Predicted Investor Sentiment Impact:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Impact:** {impact}")
