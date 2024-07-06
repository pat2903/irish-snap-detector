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
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # based off brightness, decide whether to use 
    # white mask or combined thresh
    # white mask works best when the background pixels are darker
    # combined_thresh works best when the background pixels are lighter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalized_gray = gray.astype(float) / np.max(gray)
    average_brightness = np.mean(normalized_gray)
    final_mask = None
    if average_brightness < 0.5:
        # detect card by assuming it is white and looking
        # for white pixels
        white_lower_bound = np.array([0, 0, 200])
        white_upper_bound = np.array([180, 30, 255])
        final_mask = cv2.inRange(hsv, white_lower_bound, white_upper_bound)  
    else:
        # approach used when the background is white
        # taken from https://stackoverflow.com/questions/51400374/image-card-detection-using-python
        _, thresh_H = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_S = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_mask = cv2.bitwise_or(thresh_H, thresh_S)

    # clean the mask
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(final_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # quadrilateral logic:
    # inspired by
    # https://stackoverflow.com/questions/62274412/cv2-approxpolydp-cv2-arclength-how-these-works
    res = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # cards trained are like 224x224 = ~50,000
        # going to assume the pixel area could be 10% of that, so ~5000
        if area > 5000:
            shape_perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03*shape_perimeter, True)
            # detection sometimes a bit wonky, so i'll give it a bit of leeway 
            if 4 <= len(approx) <= 6:
                res.append(approx)
    return res


# more info
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

# idx 1 to connect to continuity camera
cap = cv2.VideoCapture(1)

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

    cv2.imshow('Card Detector', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything done, release the capture
cap.release()
cv2.destroyAllWindows()