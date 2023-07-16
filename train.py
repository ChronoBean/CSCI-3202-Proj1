import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# Importing the necessary libraries

Datadirectory = "Users/benjaminrush/Downloads/test1/train/"
Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "suprise"]

# Reading and preprocessing the images

img_size = 224
training_Data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_Data()

random.shuffle(training_Data)

X = []
y = []

for features, label in training_Data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / 255.0

# Training the model

model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = tf.keras.layers.Dense(128)(base_output)
final_output = tf.keras.layers.Activation('relu')(final_output)
final_output = tf.keras.layers.Dense(64)(final_output)
final_output = tf.keras.layers.Activation('relu')(final_output)
final_output = tf.keras.layers.Dense(7, activation='softmax')(final_output)

new_model = tf.keras.Model(inputs=base_input, outputs=final_output)

new_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

new_model.fit(X, y, epochs=25)

new_model.save('Final_model_95p07.h5')

# Testing with live webcam

import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]

        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image / 255.0

        Predictions = new_model.predict(final_image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if np.argmax(Predictions) == 0:
            status = "Angry"
        elif np.argmax(Predictions) == 1:
            status = "Disgust"
        elif np.argmax(Predictions) == 2:
            status = "Fear"
        elif np.argmax(Predictions) == 3:
            status = "Happy"
        elif np.argmax(Predictions) == 4:
            status = "Sad"
        elif np.argmax(Predictions) == 5:
            status = "Surprise"
        else:
            status = "Neutral"

        x1, y1, w1, h1 = 0, 0, 175, 75
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    cv2.imshow('Face Emotion Recognition', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
