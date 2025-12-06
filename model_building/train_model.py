import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Disable GPU if needed (as per original code)
tf.config.set_visible_devices([], 'GPU')


def clean_data(text):
    """
    Cleans text by converting to lowercase, removing 'Subject:' prefix,
    stripping punctuation, and normalizing whitespace.
    """
    text = str(text).lower()
    text = re.sub(r'^subject: ', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data():
    """
    Loads dataset from CSV, applies cleaning
    """
    df = pd.read_csv("../dataset/emails.csv")
    df['Text'] = df['Text'].apply(clean_data)

    # Sampling 5000 rows as per requirements
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
    """
    Creates and adapts Vectorization layer for text into integer sequences
    """
    vectorize_layer = layers.TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=200
    )

    vectorize_layer.adapt(X_train)

    return vectorize_layer


def build_model(vectorize_layer):
    """
    Constructs a Sequential neural network model
    """
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
    """
    Evaluates the model on test data
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.3f}")

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_model(model, model_path="model.keras"):
    """
    Saves the trained model to a file
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")


def main():
    X_train, X_test, y_train, y_test = load_data()
    vectorize_layer = create_vectorizer(X_train)
    model = build_model(vectorize_layer)

    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=16,
        validation_data=(X_test, y_test))

    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == '__main__':
    main()