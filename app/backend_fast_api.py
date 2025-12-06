from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

try:
    model = tf.keras.models.load_model('../model_building/model.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")


class EmailRequest(BaseModel):
    text: str


def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'^subject: ', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.post("/predict")
def predict_spam(email: EmailRequest):
    processed_text = clean_text(email.text)

    input_data = tf.constant([processed_text], dtype=tf.string)

    prediction_score = float(model.predict(input_data)[0][0])

    is_spam = prediction_score >= 0.5
    confidence = prediction_score if is_spam else 1 - prediction_score

    return {
        "is_spam": is_spam,
        "confidence": confidence,
        "raw_score": prediction_score
    }