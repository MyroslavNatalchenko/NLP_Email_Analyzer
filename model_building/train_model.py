import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

def load_data():
    df = pd.read_csv("../dataset/emails.csv")

    df = df.sample(n=5000, random_state=42)

    X = df['Text'].astype(str).values
    y = df['Spam'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=500,
                                                        train_size=4500,
                                                        random_state=42,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

def create_vectorizer(X_train):
    vectorize_layer = layers.TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorize_layer.adapt(X_train)

    return vectorize_layer

def build_model(vectorize_layer):
    model = models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        layers.Embedding(input_dim=10000, output_dim=16),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

def save_model(model, model_path="model.keras"):
    model.save(model_path)
    print(f"Model saved to {model_path}")

def main():
    X_train, X_test, y_train, y_test = load_data()
    vectorize_layer = create_vectorizer(X_train)
    model = build_model(vectorize_layer)

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test))

    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == '__main__':
    main()