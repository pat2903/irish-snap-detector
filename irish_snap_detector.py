import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from constants import cards, create_model
import cv2
from card_recognition_video import detect_card_outline
from irish_snap_rules import IrishSnap
from card import Card

CONFIDENCE_THRESHOLD = 0.9

# disable progress bar, so it's easier to read output
tf.keras.utils.disable_interactive_logging()

class CardDetector():
    def __init__(self, webcam: int):
        self.model = create_model()
        self.model = tf.keras.models.load_model('model_4_keras.keras')
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        self.cap = cv2.VideoCapture(webcam)
        self.irish_snap = IrishSnap()
        self.last_detected_card = None

    def update_game_state(self, predicted_card: str) -> bool:
        card = Card(predicted_card)
        if card.__eq__(self.last_detected_card):
            return False
        else:
            is_snap = self.irish_snap.play_card(predicted_card)
            self.last_detected_card = card
            return is_snap


    def run(self):
        if not self.cap.isOpened():
            print("Webcam cannot be accessed.")
            exit()
        
        # freeze the frame when snap is detected

        frozen_frame = None
        while True:
            if frozen_frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break 

                card_contours = detect_card_outline(frame)

                for contour in card_contours:
                    x,y,w,h = cv2.boundingRect(contour)

                    card_img = frame[y:y+h, x:x+w]
                    card_img = cv2.resize(card_img, (224, 224))
                    card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
                    card_img = Image.fromarray(card_img)
                    img_array = np.array(card_img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    predictions = self.model.predict(img_array)
                    predicted_class_index = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_index]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_card = cards[predicted_class_index]
                        is_snap = self.update_game_state(predicted_card)

                        cv2.putText(frame, f"{predicted_card}", (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

                        if is_snap:
                            cv2.putText(frame, "SNAP!", (x-100, y+300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 30)
                            frozen_frame = frame.copy()
            else:
                frame = frozen_frame

            cv2.imshow('Irish Snap Detector!', frame)

            # press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # when everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()