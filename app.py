import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import os

# --- Configuration Constants ---
# Use the .keras file as it is the newer, more robust format.
MODEL_PATH = 'fake_news_model_final.keras'
TOKENIZER_DIR = './' # Since the tokenizer files are in the root directory
MAX_LEN = 128 # Based on common BERT max length and tokenizer.json snippet

# --- Model and Tokenizer Loading ---
# We use st.cache_resource to load the heavy assets (model and tokenizer) only once.

@st.cache_resource
def load_assets():
    """Loads the BERT tokenizer and the Keras model."""
    try:
        # 1. Load Tokenizer using the folder path
        # It needs the files: vocab.txt, tokenizer.json, tokenizer_config.json, special_tokens_map.json
        st.info("Loading BERT Tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)

        # 2. Load Keras Model
        st.info(f"Loading Keras Model from {MODEL_PATH}...")
        
        # NOTE: If your model uses custom layers not registered in Keras, you may need 
        # to define them here in the `custom_objects` dict.
        # However, the .keras format often handles most internal layer definitions automatically.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        return tokenizer, model
    except FileNotFoundError as e:
        st.error(f"Asset loading failed: One or more required files were not found.")
        st.error(f"Ensure all files (.keras, .txt, .json) are in the same directory as this app.py.")
        st.error(f"Missing file or directory issue: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model/tokenizer loading: {e}")
        st.error("This is often due to dependency/version mismatch (e.g., TensorFlow/Keras version).")
        st.stop()


# --- Text Preprocessing Function ---
def preprocess_text(text, tokenizer, max_len):
    """Encodes the input text into BERT's required input format."""
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'  # Returns TensorFlow tensors
    )
    # BERT-based models typically require input_ids and attention_mask
    return [encoded['input_ids'], encoded['attention_mask']]


# --- Prediction Function ---
def predict_fake_news(text, tokenizer, model):
    """Runs the model prediction and formats the output."""
    if not text.strip():
        return "Please enter an article to classify.", None

    # Preprocess the input text
    inputs = preprocess_text(text, tokenizer, MAX_LEN)
    
    # Predict
    # The output is likely a probability (e.g., [0.95])
    prediction = model.predict(inputs)
    
    # Assuming Binary Classification (0=Real, 1=Fake) and sigmoid activation
    fake_prob = prediction[0][0]
    
    # Determine the class based on a 0.5 threshold
    if fake_prob > 0.5:
        label = "FAKE"
        confidence = fake_prob
    else:
        label = "REAL"
        confidence = 1.0 - fake_prob
        
    return label, confidence


# --- Streamlit UI Layout ---
def main():
    st.set_page_config(
        page_title="BERT-BiLSTM Fake News Detector",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for aesthetics
    st.markdown("""
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    .main-header {
        font-size: 2.5em;
        font-weight: 700;
        color: #1e3a8a; /* Deep Blue */
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.2em;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2em;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    .fake-result {
        background-color: #fee2e2; /* Light Red */
        border: 2px solid #ef4444; /* Red Border */
    }
    .real-result {
        background-color: #d1fae5; /* Light Green */
        border: 2px solid #10b981; /* Green Border */
    }
    .result-text {
        font-size: 2em;
        font-weight: bold;
    }
    .confidence-text {
        font-size: 1em;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📰 Fake News Detector 🕵️‍♀️</div>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">BERT + BiLSTM Model for Article Verification</p>', unsafe_allow_html=True)

    # --- Load Assets ---
    tokenizer, model = load_assets()

    # --- Input Area ---
    article_text = st.text_area(
        "Paste the news article text here:",
        height=300,
        placeholder="E.g., 'A shocking new study claims eating chocolate can reverse aging, but experts are skeptical.'"
    )

    # --- Predict Button ---
    if st.button("Classify Article", use_container_width=True, type="primary"):
        with st.spinner('Analyzing content...'):
            label, confidence = predict_fake_news(article_text, tokenizer, model)

        if confidence is not None:
            
            confidence_percent = f"{confidence * 100:.2f}%"
            
            if label == "FAKE":
                st.markdown(
                    f"""
                    <div class="result-box fake-result">
                        <div class="result-text" style="color: #ef4444;">🚨 FAKE NEWS ALERT 🚨</div>
                        <div class="confidence-text">Confidence: {confidence_percent}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box real-result">
                        <div class="result-text" style="color: #10b981;">✅ REAL NEWS</div>
                        <div class="confidence-text">Confidence: {confidence_percent}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            # Displays the error message from the predict function (e.g., if text is empty)
            st.warning(label)

if __name__ == "__main__":
    # Ensure TensorFlow is configured to run eagerly for debugging simplicity, if needed
    # tf.config.run_functions_eagerly(True) 
    main()
