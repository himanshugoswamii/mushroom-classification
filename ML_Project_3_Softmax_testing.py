# proj3_classification_test.py
import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASSES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma',
           'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img

def load_test_data(test_csv):
    df = pd.read_csv(test_csv)

    images, labels = [], []
    for _, row in df.iterrows():
        full_path = os.path.join(os.path.dirname(test_csv), row['image_path'])
        img = preprocess_image(full_path)
        images.append(img)
        labels.append(row['label'])

    X = np.array(images)
    le = LabelEncoder()
    y = le.fit(CLASSES).transform(labels)
    y = tf.convert_to_tensor(y)
    return X, y



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model_classification.h5', help='Path to trained classification model')
    parser.add_argument('--test_csv', type=str, default='mushrooms_test.csv', help='CSV with image paths and labels')
    args = parser.parse_args()

    print("🔍 Loading model...")
    model = tf.keras.models.load_model(args.model)

    print("📥 Loading test data...")
    X_test, y_test = load_test_data(args.test_csv)

    print("🧪 Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("✅ Test Accuracy: {:.2f}%".format(acc * 100))

if _name_ == "_main_":
    main()