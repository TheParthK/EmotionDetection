from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np

trained_model = tf.keras.models.load_model('facial_emotions_model.h5')
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
cascade = cv2.CascadeClassifier("haarcascade.xml")
a = st.empty()

class VideoProcessor:
    def recv(self, f):
        frame = f.to_ndarray(format='bgr24')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame = frame[y:y+h, x:x+w]
        
        resized = cv2.resize(frame, (48, 48))
        resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        reshaped_array = np.reshape(resized_gray, (1, 48, 48, 1))
        result = trained_model.predict(reshaped_array)
        y_pred = np.argmax(result[0])
        pred = classes[y_pred]
        a.write(pred)
#         return av.VideoFrame.from_ndarray(frame, format='bgr24')
	  return av.VideoFrame.from_ndarray(frame)



webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)
