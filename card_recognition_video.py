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

######## CARD PREPROCESSING #########

# Isolate card based of colour and shape
def detect_card_outline(frame):
    """
    Detect playing cards in a given frame

    This is done by:
    1. Using HSV (Hue, saturation, value) colour space for better colour-based segmentation 
    2. Focus on white; I assume that cards are white, which they usually are.
    3. Use morphological operatons (shape based) to clean up edges and remove noise.
    4. Filter contours; I detect only card-like shapes.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range for white colour
    # remember, working in HSV now
    white_lower_bound = np.array([0, 0, 200])
    white_upper_bound = np.array([180, 30, 255])

    # create binary mask of white (i.e. card) regions
    mask = cv2.inRange(hsv, white_lower_bound, white_upper_bound)

    # clean up  mask using morphological operations
    # https://www.geeksforgeeks.org/python-opencv-morphological-operations/
    # like a matrix
    kernel = np.ones((5,5), np.uint8)
    # fill regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # remove noise i.e. black regions
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # now, i filter the countours
    # to identify card-like shapes
    # cards trained are like 224x224 = ~50,000
    # going to assume the pixel area could be 10% of that, so ~5000
    res = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # quadrilateral logic:
        # https://stackoverflow.com/questions/62274412/cv2-approxpolydp-cv2-arclength-how-these-works
        if area > 5000:
            shape_perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*shape_perimeter, True)
            if len(approx) == 4:
                res.append(approx)
    return res


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

    card_contours = detect_card_outline(frame)

    for contour in card_contours:
        # going to use to mark the card boundary
        x,y,w,h = cv2.boundingRect(contour)

        card_img = frame[y:y+h, x:x+w]
        card_img = cv2.resize(card_img, (224, 224))
        card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
        card_img = Image.fromarray(card_img)
        img_array = np.array(card_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_card = cards[predicted_class_index]

            cv2.putText(frame, f"{predicted_card}", (x, y-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y+h+35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()