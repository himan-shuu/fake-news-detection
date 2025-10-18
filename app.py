import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import os

# --- Configuration Constants ---
# Using the .h5 file for file accessibility.
MODEL_PATH = 'fake_news_model_final.h5' 
TOKENIZER_DIR = './' # Since the tokenizer files are in the root directory
MAX_LEN = 128 # Based on tokenizer.json snippet

# --- Custom Object Fix (Needed for model loading) ---
def bert_encode_stub(input_tensor):
    """
    Stub function for the 'bert_encode' custom object used in the Keras model.
    """
    return input_tensor


# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_assets():
    """Loads the BERT tokenizer, the Keras model, and its expected input names."""
    try:
        # 1. Load Tokenizer
        st.info("Loading BERT Tokenizer...")
        # CRITICAL FIX: Use the standard from_pretrained method on the directory
        # This is more robust for loading the full configuration (vocab.txt + JSON files)
        tokenizer = BertTokenizer.from_pretrained(
            TOKENIZER_DIR, 
            do_lower_case=True,
            # trust_remote_code=True is added for increased compatibility
            trust_remote_code=True 
        )

        # 2. Load Keras Model with the custom object fix
        st.info(f"Loading Keras Model from {MODEL_PATH}...")
        
        # We pass the custom stub for the Lambda layer
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={"bert_encode": bert_encode_stub}, 
            compile=False
        )
        
        # We'll return a placeholder to maintain structure.
        input_names = ['input_ids', 'attention_mask'] # Placeholder
        
        # Return all necessary components
        return tokenizer, model, input_names
    except FileNotFoundError as e:
        st.error(f"Asset loading failed: One or more required files were not found.")
        st.error(f"Missing file or directory issue: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model/tokenizer loading: {e}")
        st.error("This is likely due to dependency mismatch. Try installing specific versions: pip install tensorflow==2.15.0 transformers==4.38.0")
        st.stop()


# --- Text Preprocessing Function ---
def preprocess_text(text, tokenizer, max_len):
    """Encodes the input text into BERT's required input dictionary format."""
    # The tokenizer always returns 'input_ids' and 'attention_mask'
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'  # Returns TensorFlow tensors
    )
    
    # Return the two tensors as a tuple
    return encoded['input_ids'], encoded['attention_mask']


# --- Prediction Function ---
# CRITICAL CHANGE: The inputs are now passed as a list of tensors.
def predict_fake_news(text, tokenizer, model, input_names):
    """Runs the model prediction and formats the output."""
    if not text.strip():
        return "Please enter an article to classify.", None

    # Preprocess the input text (returns the two tensors)
    input_ids, attention_mask = preprocess_text(text, tokenizer, MAX_LEN)
    
    # CRITICAL FIX: Pass inputs as a list (positional matching)
    inputs_list = [input_ids, attention_mask]
    
    # Predict - This list format is the most robust for multi-input H5 models
    prediction = model.predict(inputs_list) 
    
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
    # CRITICAL FIX: Store CSS in a variable to avoid multiline string issues near st.markdown
    css_content = """
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
        transition: all 0.3s ease-in-out;
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
    """
    st.markdown(css_content, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📰 BERT-BiLSTM Fake News Detector 🕵️‍♀️</div>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Powered by Hugging Face Tokenizer and Keras</p>', unsafe_allow_html=True)

    # --- Load Assets ---
    # Unpack the new input_names variable (it's now a placeholder)
    tokenizer, model, input_names = load_assets()

    # --- Input Area ---
    article_text = st.text_area(
        "Paste the news article text here:",
        height=300,
        placeholder="E.g., 'A shocking new study claims eating chocolate can reverse aging, but experts are skeptical.'"
    )

    # --- Predict Button ---
    if st.button("Classify Article", use_container_width=True, type="primary"):
        with st.spinner('Analyzing content...'):
            # Pass input_names (placeholder, but keeps signature consistent)
            label, confidence = predict_fake_news(article_text, tokenizer, model, input_names)

        if confidence is not None:
            
            confidence_percent = f"{confidence * 100:.2f}%"
            
            if label == "FAKE":
                st.markdown(
                    f"""
                    <div class="result-box fake-result">
                        <div class="result-text" style="color: #ef4444;">🚨 FAKE NEWS ALERT 🚨</div>
                        <div class="confidence-text">Confidence: {confidence_percent} that the article is Fake.</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.balloons()
            else:
                st.markdown(
                    f"""
                    <div class="result-box real-result">
                        <div class="result-text" style="color: #10b981;">✅ REAL NEWS</div>
                        <div class="confidence-text">Confidence: {confidence_percent} that the article is Real.</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            # Displays the error message from the predict function (e.g., if text is empty)
            st.warning(label)

if __name__ == "__main__":
    main()
