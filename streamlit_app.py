import streamlit as st
from datetime import datetime
import pandas as pd

from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,  # Force CPU
    revision="main",  # Avoids meta tensor bugs in some versions
    torch_dtype="float32"  # Forces safe tensor loading
)


# Streamlit UI
st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("🧠 HuggingFace Sentiment Classifier")
st.write("Enter some text and see how the model feels about it!")

# Input box
user_input = st.text_area("Your Text Here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]
            label = result['label']
            confidence = result['score']

            # Display result
            st.markdown(f"### Sentiment: `{label}`")
            st.markdown(f"**Confidence:** `{confidence:.2f}`")

            # Save to CSV
            log_data = {
                "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "text": [user_input],
                "label": [label],
                "confidence": [round(confidence, 4)]
            }

            df = pd.DataFrame(log_data)

            # Append or create CSV
            try:
                old_df = pd.read_csv("sentiment_log.csv")
                df = pd.concat([old_df, df], ignore_index=True)
            except FileNotFoundError:
                pass

            df.to_csv("sentiment_log.csv", index=False)
            st.success("✅ Logged to sentiment_log.csv")
