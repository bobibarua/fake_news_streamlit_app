# Fake News Detection Streamlit App

## 📌 Project Overview

This repository contains the **Streamlit web application** for the Fake News Detection system, built using the **GRU deep learning model** from the main ML/DL project. Users can input news text and receive an instant prediction of whether the news is *Fake* or *Real*.

This repo focuses exclusively on **deployment and inference**, while the training, evaluation, and experimentation are maintained in a separate repository.

---

## 🗂️ Files Included

* `.gitignore`
* `README.md`
* `app.py`
* `model__gru.h5`
* `model_metadata.json`
* `requirements.txt`
* `runtime.txt`
* `tokenizer.pkl`

---

## 🚀 Live Demo

The app is deployed on Streamlit Cloud and can be accessed here:
[https://fake-news-system-app.streamlit.app/](https://fake-news-system-app.streamlit.app/)

---

## How to Run Locally

Follow these steps to run the Streamlit app locally:

1. **Clone the repository**

```bash
git clone https://github.com/bobibarua/fake_news_streamlit_app.git
cd fake_news_streamlit_app
```

2. **(Optional) Create a virtual environment**

```bash
python -m venv venv          # Create virtual environment
source venv/bin/activate     # Activate (Linux/Mac)
venv\Scripts\activate      # Activate (Windows)
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

5. **Open in browser**
   After running the above command, Streamlit will provide a local URL (usually `http://localhost:8501`) where you can access the app.

> All required files (`model__gru.h5`, `tokenizer.pkl`, `model_metadata.json`) are already included in the repo, so no additional setup is needed.

---

## 🔗 Related Repository

For the complete project, including training, evaluation, and notebooks, visit:
[Fake News Detection ML/DL Repository](https://github.com/bobibarua/fake-news-detection-ml-dl)

---

## 👤 Author

**Bobi Barua**
GitHub: [https://github.com/bobibarua](https://github.com/bobibarua)

---

⭐ If you find this project useful, consider giving it a star!

