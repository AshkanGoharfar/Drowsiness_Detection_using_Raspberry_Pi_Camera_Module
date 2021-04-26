import io
import picamera
import cv2
import numpy
from Operators import *
import time
import tensorflow as tf
import numpy as np


def run_cam():
    counter = 0
    model = tf.keras.models.load_model('LSTM_saved_model.h5')
    model.summary()
    sequence_of_distances = []
    while True:
        stream = io.BytesIO()

        # Get the picture (low resolution, so it should be quite fast)
        # Here you can also specify other parameters (e.g.:rotate the image)
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 240)
            camera.capture(stream, format='jpeg')

        buff = numpy.frombuffer(stream.getvalue(), dtype=numpy.uint8)
        image = cv2.imdecode(buff, 1)

        # Load a cascade file for detecting faces
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Draw a rectangle around every found face
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 4)

        distances = np.array(read_DROZY_videos(image))
        sequence_of_distances.append(distances)
        if counter % 10 == 0:
            Y_pred = model.predict(np.array(sequence_of_distances).reshape([-1, 10, 12]))
            sequence_of_distances = []
            y_pred = np.argmax(Y_pred, axis=1)
            print('y_pred is : ', y_pred)

        counter += 1
        time.sleep(1)
