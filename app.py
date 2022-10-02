import cv2
import streamlit as st
import tensorflow as tf
import numpy as np

trained_model = tf.keras.models.load_model('facial_emotions_model.h5')
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']


st.title('Emotion Detection')

# run = st.checkbox('Run')

frame_window = st.image([])
frame_window2 = st.image([])

cam = cv2.VideoCapture(0)

# pred = 'Detecting'

a = st.empty()
face_cascade = cv2.CascadeClassifier('haarcascade.xml')
# a.text_area(value='detecting', label='')
while True:
    ret, frame = cam.read()
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window2.image(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            frame = frame[y:y+h, x:x+h]





        if frame.any():
            resized = cv2.resize(frame, (48, 48))
            resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            reshaped_array = np.reshape(resized_gray, (1, 48, 48, 1))
            result=trained_model.predict(reshaped_array)
            y_pred=np.argmax(result[0])

            pred = classes[y_pred]
            # print('The person facial emotion is:',classes[y_pred])
            a.write(pred)
        # a = st.empty()


        frame_window.image(resized_gray)



