import streamlit as st
import numpy as np
import joblib
import json
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page settings
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

# Load model, tokenizer, metadata
model = load_model("model__gru.h5")
tokenizer = joblib.load("tokenizer.pkl")

with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

MAX_LEN = metadata["MAX_LEN"]

# Prediction function
def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prob = model.predict(padded)[0][0]
    pred = 1 if prob >= 0.5 else 0
    return pred, prob  # prob = REAL probability

# Initialize session state
if "do_predict" not in st.session_state:
    st.session_state.do_predict = False

# Custom CSS for dark background image and readable text
st.markdown("""
    <style>
        /* Background image with overlay */
        .stApp {
            background: 
                linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                url("https://cdn.builtin.com/cdn-cgi/image/f=auto,fit=cover,w=1200,h=635,q=80/sites/www.builtin.com/files/2024-10/use-ai-battle-misinformation.png");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Header */
        .title-banner {
            font-size: 42px;
            font-weight: 900;
            text-align: center;
            padding: 30px;
            border-radius: 15px;
            background: rgba(255,255,255,0.2);
            color: #fff;
            margin-bottom: 25px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }

        .sub-text {
            text-align: center;
            font-size: 20px;
            margin-bottom: 25px;
            color: #f0f0f0;
        }

        /* Textarea */
        textarea {
            border-radius: 12px !important;
            border: 2px solid #fff !important;
            background-color: rgba(255,255,255,0.95);
            color: #000;
        }

        /* Remove focus outline/box-shadow on textarea and inputs */
        textarea:focus, textarea:active, .stTextArea textarea:focus {
            outline: none !important;
            box-shadow: none !important;
            border-color: #ffffff !important;
        }

        /* Button sizing - target Streamlit button */
        div.stButton > button {
            width: 180px;
            height: 50px;
            background-color: #4a90e2;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            border: none !important;
        }
        div.stButton > button:hover {
            background-color: #357ab8;
            color: white;
        }
        /* remove focus outline from button */
        div.stButton > button:focus,
        div.stButton > button:active {
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }

        /* Result boxes */
        .result-box {
            padding: 22px;
            border-radius: 12px;
            font-size: 22px;
            text-align: center;
            font-weight: 600;
            margin-top: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }

        .real {
            background-color: rgba(40,167,69,0.95);
            color: #fff;
        }

        .fake {
            background-color: rgba(220,53,69,0.95);
            color: #fff;
        }

        /* Progress bars customization */
        /* 1st bar = REAL, 2nd bar = FAKE */
        .stProgress > div > div > div > div[role="progressbar"]:nth-child(1) {
            background: linear-gradient(90deg, #28a745, #28a745) !important; /* green */
        }
        .stProgress > div > div > div > div[role="progressbar"]:nth-child(2) {
            background: linear-gradient(90deg, #dc3545, #dc3545) !important; /* red */
        }

        /* Confidence Scores heading and text */
        h3, .stMarkdown, .stText {
            color: #ffffff !important;
        }

        /* Spinner text */
        div.stSpinner > div > div {
            color: #ffffff !important;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            color: #f0f0f0;
        }

        /* center a block if needed */
        .center-block {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title-banner">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Enter a news article or headline below to check if it is REAL or FAKE.</p>', unsafe_allow_html=True)

# Text input
text = st.text_area(
    "News Text:",
    key="text_input",
    height=200,
    placeholder="Type or paste news content here..."
)

# --- Centered button using columns (reliable) ---
col1, col2, col3 = st.columns([1, 0.6, 1])  # middle column will be narrower so button appears centered
with col2:
    # remove use_container_width to keep button fixed width
    if st.button("Predict"):
        if text.strip():
            st.session_state.do_predict = True

# Prediction logic
if st.session_state.do_predict and st.session_state.text_input.strip():
    with st.spinner("Analyzing... Please wait."):
        time.sleep(1.5)
        prediction, real_prob = predict(st.session_state.text_input)
        fake_prob = 1 - real_prob
        real_percent = real_prob * 100
        fake_percent = fake_prob * 100

    # Display prediction
    if prediction == 1:
        st.markdown(
            f'<div class="result-box real">✅ This news is <b>REAL</b></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box fake">🚫 This news is <b>FAKE</b></div>',
            unsafe_allow_html=True
        )

    # Confidence scores
    st.write("### Confidence Scores")
    st.write(f"REAL: {real_percent:.2f}%")
    st.progress(int(real_percent))
    st.write(f"FAKE: {fake_percent:.2f}%")
    st.progress(int(fake_percent))

# Footer
st.markdown('<p class="footer">Developed using Streamlit</p>', unsafe_allow_html=True)
