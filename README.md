# AI_Fake_Image_Detector_Trees
AI project to detect fake vs real images — privacy-safe demo
AI project to detect real vs fake images in a privacy-safe and visually appealing way.  
The project can work with any type of images (animals, cars, paintings, etc.).
AI_Fake_Image_Detector/
└─ src/
# AI_Fake_Image_Detector_Trees.py

"""
AI Fake Image Detector using Decision Trees
Author: Your Name
Description: This is a simple AI project that detects whether an image is fake or real using a decision tree classifier.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to load images and convert them to feature vectors
def load_images(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if not os.path.isdir(label_folder):
            continue
        for file in os.listdir(label_folder):
            img_path = os.path.join(label_folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img.flatten())
                labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_images("dataset")  # ضع هنا اسم مجلد الصور: "real" و "fake" بداخله

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
