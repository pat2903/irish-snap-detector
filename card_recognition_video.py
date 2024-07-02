import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from constants import cards, create_model
import cv2
import time

CONFIDENCE_THRESHOLD = 0.7

# disable progress bar, so it's easier to read output
tf.keras.utils.disable_interactive_logging()

model = create_model()

model = tf.keras.models.load_model('model_4_keras.keras')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# more info
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open webcam")
    exit()


while True:
    # capture the video frame buy frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break 

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_card = cards[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    if confidence >= CONFIDENCE_THRESHOLD:
        predicted_card = cards[predicted_class_index]
        cv2.putText(frame, f"Card: {predicted_card}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No card detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()