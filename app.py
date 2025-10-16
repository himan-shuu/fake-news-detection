import os
import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer

st.title("📰 Fake News Detection App")
st.write("Detect fake news using BERT + BiLSTM model.")

# ----------------------------
# Load model and tokenizer
# ----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    # Load the Keras model directly from folder
    model = tf.keras.models.load_model("fake_news_model.keras", compile=False)
    
    # Load the tokenizer folder
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
