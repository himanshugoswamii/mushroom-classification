import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASSES = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma',
           'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    return img

def load_test_data(test_csv_path):
    df = pd.read_csv(test_csv_path)
    images, labels = [], []
    for _, row in df.iterrows():
        full_path = os.path.join(os.path.dirname(test_csv_path), row['image_path'])
        images.append(preprocess_image(full_path))
        labels.append(row['label'])
    X = np.array(images)
    y = LabelEncoder().fit(CLASSES).transform(labels)
    return X, y

X_test, y_true = load_test_data("sample_test_data(1) (1)/mushrooms_test.csv")

full_model = tf.keras.models.load_model("softmax_model.keras", compile=False)
extractor = tf.keras.Sequential(full_model.layers[:-2])  

features = extractor.predict(X_test)

svm = joblib.load("svm_model.pkl")
y_pred = svm.predict(features)
acc = accuracy_score(y_true, y_pred)
print(f"\n SVM Test Accuracy: {acc*100:.2f}%")

le = LabelEncoder()
le.fit(CLASSES)
true_labels = le.inverse_transform(y_true)
pred_labels = le.inverse_transform(y_pred)

print("\n🧾 Prediction Breakdown:")
for i in range(len(y_true)):
    print(f"Image {i+1}: True = {true_labels[i]:<10} | Predicted = {pred_labels[i]}")
