import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from constants import cards, create_model
import cv2

model = create_model()

model = tf.keras.models.load_model('model_4_keras.keras')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# note to self:
# EfficientNet models expect pixels in the range of 0 to 255 
# so do NOT scale images for this model
img_path = 'ace_diamond.jpg'
img = Image.open(img_path).resize((224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# make prediction
predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions[0])

predicted_card = cards[predicted_class_index]

confidence = predictions[0][predicted_class_index]

print(f"Predicted card: {predicted_card}")
print(f"Confidence: {confidence:.2f}")