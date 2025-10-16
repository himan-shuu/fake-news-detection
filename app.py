import os
import requests
import zipfile
import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer

st.title("📰 Fake News Detection App")
st.write("Detect fake news using BERT + BiLSTM model.")

# ----------------------------
# Download model & tokenizer if not exists
# ----------------------------
MODEL_URL = "https://github.com/himan-shuu/fake-news-detection/raw/main/fake_news_model.h5"
TOKENIZER_ZIP_URL = "https://github.com/himan-shuu/fake-news-detection/archive/refs/heads/main.zip"

if not os.path.exists("fake_news_model.h5"):
    r = requests.get(MODEL_URL)
    with open("fake_news_model.h5", "wb") as f:
        f.write(r.content)

if not os.path.exists("fake_news_tokenizer"):
    r = requests.get(TOKENIZER_ZIP_URL)
    with open("repo.zip", "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile("repo.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("fake-news-detection-main/fake_news_tokenizer", "fake_news_tokenizer")

# ----------------------------
# Load model and tokenizer
# ----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("fake_news_model.h5", compile=False)
    tokenizer = BertTokenizer.from_pretrained("fake_news_tokenizer")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ----------------------------
# Streamlit UI
# ----------------------------
user_input = st.text_area("Enter a news headline or article text:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        inputs = tokenizer(
            [user_input],
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128
        )
        preds = model(inputs)
        probs = tf.nn.softmax(preds.logits if hasattr(preds, "logits") else preds, axis=-1)
        pred_class = int(tf.argmax(probs, axis=1))

        label = "🟢 Real News" if pred_class == 1 else "🔴 Fake News"
        st.subheader(label)
        st.write(f"**Confidence:** {float(tf.reduce_max(probs) * 100):.2f}%")
