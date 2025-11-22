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
    layout="centered"
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

# Custom CSS

st.markdown("""
    <style>
        .title-banner {
            font-size: 40px;
            font-weight: 900;
            text-align: center;
            padding: 25px;
            border-radius: 12px;
            background: linear-gradient(90deg, #005c97, #363795);
            color: white;
            margin-bottom: 25px;
        }
        .sub-text {
            text-align: center;
            font-size: 18px;
            margin-bottom: 25px;
            color: #444;
        }
        textarea {
            border-radius: 12px !important;
        }
        .result-box {
            padding: 22px;
            border-radius: 12px;
            font-size: 22px;
            text-align: center;
            font-weight: 600;
            margin-top: 25px;
        }
        .real {
            background-color: #d4edda;
            color: #155724;
            border-left: 8px solid #28a745;
        }
        .fake {
            background-color: #f8d7da;
            color: #721c24;
            border-left: 8px solid #dc3545;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
            color: #777;
        }
    </style>
""", unsafe_allow_html=True)

# Header

st.markdown('<div class="title-banner">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Enter a news article or headline below to check if it is likely REAL or FAKE.</p>', unsafe_allow_html=True)

# --------------------------
# Text input (bound to session state)
# --------------------------
text = st.text_area(
    "News Text:",
    key="text_input",  # directly bound to session state
    height=200,
    placeholder="Type or paste news content here..."
)

# Predict button

if st.button("Predict", use_container_width=True):
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

    # Display prediction first
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

    # Show confidence scores
    st.write("### Confidence Scores")
    st.write(f"REAL: {real_percent:.2f}%")
    st.progress(int(real_percent))
    st.write(f"FAKE: {fake_percent:.2f}%")
    st.progress(int(fake_percent))

# Footer
st.markdown('<p class="footer">Developed using Streamlit</p>', unsafe_allow_html=True)
