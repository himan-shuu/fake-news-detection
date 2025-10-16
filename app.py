import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

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
st.title("📰 Fake News Detection App")
st.write("This app uses a fine-tuned BERT + BiLSTM model to detect fake news.")

user_input = st.text_area("Enter a news headline or article text:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer(
            [user_input],
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Model prediction
        preds = model(inputs)
        probs = tf.nn.softmax(preds.logits if hasattr(preds, "logits") else preds, axis=-1)
        pred_class = int(tf.argmax(probs, axis=1))

        label = "🟢 Real News" if pred_class == 1 else "🔴 Fake News"
        st.subheader(label)
        st.write(f"**Confidence:** {float(tf.reduce_max(probs) * 100):.2f}%")
