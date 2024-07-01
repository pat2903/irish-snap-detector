import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

cards = [
    "ace of clubs",
    "ace of diamonds",
    "ace of hearts",
    "ace of spades",
    "eight of clubs",
    "eight of diamonds",
    "eight of hearts",
    "eight of spades",
    "five of clubs",
    "five of diamonds",
    "five of hearts",
    "five of spades",
    "four of clubs",
    "four of diamonds",
    "four of hearts",
    "four of spades",
    "jack of clubs",
    "jack of diamonds",
    "jack of hearts",
    "jack of spades",
    "joker",
    "king of clubs",
    "king of diamonds",
    "king of hearts",
    "king of spades",
    "nine of clubs",
    "nine of diamonds",
    "nine of hearts",
    "nine of spades",
    "queen of clubs",
    "queen of diamonds",
    "queen of hearts",
    "queen of spades",
    "seven of clubs",
    "seven of diamonds",
    "seven of hearts",
    "seven of spades",
    "six of clubs",
    "six of diamonds",
    "six of hearts",
    "six of spades",
    "ten of clubs",
    "ten of diamonds",
    "ten of hearts",
    "ten of spades",
    "three of clubs",
    "three of diamonds",
    "three of hearts",
    "three of spades",
    "two of clubs",
    "two of diamonds",
    "two of hearts",
    "two of spades"
]

def create_model():
    IMG_SIZE = 224
    NUM_CLASSES = 53
    V2S_Model = EfficientNetV2S(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights=None)
    
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = V2S_Model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Concatenate()([x, x1])
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
model = create_model()

# Load the entire model (not just weights)
model = tf.keras.models.load_model('model_4_keras.keras')

# Compile the model
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

# Make prediction
predictions = model.predict(img_array)

# Get the predicted class index
predicted_class_index = np.argmax(predictions[0])

# Get the predicted class name
predicted_card = cards[predicted_class_index]

# Get the confidence score
confidence = predictions[0][predicted_class_index]

# Print the results
print(f"Predicted card: {predicted_card}")
print(f"Confidence: {confidence:.2f}")