# Email Spam Recognizer

Full-stack ML app designed to classify emails as "Spam" or "Not Spam" [Ham]. This project demonstrates data preprocessing and model training with TensorFlow/Keras, deployment using: FastAPI for backend and Streamlit for frontend

![demo](https://github.com/MyroslavNatalchenko/NLP_Email_Analyzer/blob/main/img/demonstration_of_web_page.png)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model Evaluation](#model-evaluation)
4. [Usage](#usage)
    - [1. Download libraries](#1-train-the-model)
    - [2. Start the Backend](#2-start-the-backend)
    - [3. Start the Frontend](#3-start-the-frontend)
5. [Project Structure](#project-structure)
6. [Tech Stack](#tech-stack)

---

## Project Overview

1.  **Model Training:** Deep learning model (Embedding + GlobalAveragePooling1D) is trained on the email dataset from Kaggle
2.  **Backend (API):** FastAPI server loads the trained model and exposes a REST endpoint (`/predict`) to handle classification requests
3.  **Frontend (UI):** Streamlit interface accepts user input, sends it to the backend, and displays the result (Spam/Not Spam) along with a confidence score.

## Model Evaluation

The model performance is evaluated using the following metrics:

  * **Classification report:** The overall correctness of the model.
```
              precision    recall  f1-score   support

         Ham       0.97      0.99      0.98       380
        Spam       0.97      0.92      0.94       120

    accuracy                           0.97       500
   macro avg       0.97      0.95      0.96       500
weighted avg       0.97      0.97      0.97       500
```
  * **Confusion Matrix:** visualizes False Positives and False Negatives.
  * **ROC Curve:** Measures the model's ability to distinguish between classes.

![cm_roc](https://github.com/MyroslavNatalchenko/NLP_Email_Analyzer/blob/main/img/cm_roc.png)

## Usage

### 1\. Download all needable libraries for your environment

```bash
pip install -r requirements.txt
```

### 2\. Start the Backend

Open a new terminal, navigate to the `app` folder, and start the FastAPI server.

```bash
cd app
uvicorn backend_fast_api:app --reload
```

### 3\. Start the Frontend

Open another new terminal, navigate to the `app` folder, and launch the Streamlit interface.

```bash
cd app
streamlit run frontend_streamlit.py
```

## Project Structure

```text
├── README.md                   # Project documentation
├── app/
│   ├── backend_fast_api.py     # FastAPI Backend
│   └── frontend_streamlit.py   # Streamlit Frontend
├── dataset/
│   └── emails.csv              # Dataset
├── model_building/
│   ├── model.keras             # Saved model
│   └── train_model.py          # Model training script
└── requirements.txt            # Dependencies for project
````

## Tech Stack

  * **Language:** Python 3.10+
  * **Machine Learning:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
  * **Backend:** FastAPI, Uvicorn
  * **Frontend:** Streamlit
  * **Visualization:** Matplotlib, Seaborn
