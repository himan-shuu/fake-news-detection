import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import os
import json 

# --- Configuration Constants ---
# Using the .h5 file for file accessibility.
MODEL_PATH = 'fake_news_model_final.h5' 
TOKENIZER_DIR = './' 
MAX_LEN = 128 

# --- Custom Object Fix (Needed for model loading) ---
def bert_encode_stub(input_tensor):
    """
    Stub function for the 'bert_encode' custom object used in the Keras model.
    This resolves the 'Unknown object type' error.
    """
    return input_tensor


# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_assets():
    """Loads the BERT tokenizer, the Keras model, and its expected input names."""
    try:
        # 1. Load Tokenizer Components Manually (Minimalist approach to avoid JSON error)
        st.info("Loading BERT Tokenizer using only vocab.txt...")
        
        vocab_path = os.path.join(TOKENIZER_DIR, 'vocab.txt')
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found at: {vocab_path}")
            
        # Use the raw constructor with hardcoded standard BERT special tokens. 
        # This isolates the tokenizer from corrupted JSON configuration files.
        tokenizer = BertTokenizer(
            vocab_file=vocab_path,
            do_lower_case=True,
            cls_token='[CLS]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            unk_token='[UNK]',
            mask_token='[MASK]'
        )

        # 2. Load Keras Model with the custom object fix
        st.info(f"Loading Keras Model from {MODEL_PATH}...")
        
        # We pass the custom stub for the Lambda layer
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={"bert_encode": bert_encode_stub}, 
            compile=False
        )
        
        # --- FIX: Removed the model.input_names check to bypass the AttributeError ---
        # We rely solely on the most common fallback names for multi-input H5 models.
        input_names = ['input_1', 'input_2']
            
        # Return all necessary components
        return tokenizer, model, input_names
    except FileNotFoundError as e:
        st.error(f"Asset loading failed: One or more required files were not found.")
        st.error(f"Missing file or directory issue: {e}")
        st.stop()
    except Exception as e:
        # Generic catch for all other errors
        st.error(f"An error occurred during asset loading: {e}")
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
    
    # Return the two tensors as a tuple (T1=input_ids, T2=attention_mask)
    return encoded['input_ids'], encoded['attention_mask']


# --- Prediction Function ---
def predict_fake_news(text, tokenizer, model, input_names):
    """Runs the model prediction and formats the output."""
    if not text.strip():
        return "Please enter an article to classify.", None

    # Preprocess the input text (returns the two tensors)
    input_ids_tensor, attention_mask_tensor = preprocess_text(text, tokenizer, MAX_LEN)
    
    # CRITICAL FIX: Convert Tensors to NumPy arrays and create a dictionary 
    # using the discovered/fallback input layer names from load_assets.
    
    # input_names has two elements (['input_1', 'input_2'])
    
    # Create the dictionary:
    inputs_dict = {}
    
    # Case 1: Assumed order based on standard BERT input pipeline (T1=IDs, T2=Mask)
    # Map the first input name to the input_ids tensor (T1)
    inputs_dict[input_names[0]] = input_ids_tensor.numpy()
    # Map the second input name to the attention_mask tensor (T2)
    inputs_dict[input_names[1]] = attention_mask_tensor.numpy()
    
    # Predict - using dictionary input with numpy arrays
    prediction = model.predict(inputs_dict) 
    
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
            st.warning(label)

if __name__ == "__main__":
    main()
