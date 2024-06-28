import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
from tensorflow.python.keras.layers import EfficientNetB0
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# First tried CNN, but I had trouble with test accuracy
# Now going to try a pre-trained model like EfficentNet 

# paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'
validate_dir = 'dataset/valid'

# in general, batch size 32 is recommended
# https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
BATCH_SIZE = 32

# next, augment the data
# essentially, "mess" with the image to simulate real-world inconsistencies
# un-augmented images also used, in addition to these
# also, normalising using 255 to fit the range 0-1

# 0.2 for shift range seems to be the consensus
train_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range = 15,
    horizontal_flip = True,
    fill_mode = 'nearest',
    shear_range = 0.1,
    zoom_range = 0.2,
)

validate_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)


# load images from directory

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    # turns off neurons during training
    # this helps with overfitting
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# compiling the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )

# train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10,
    validation_data=validate_generator,
    validation_steps=validate_generator.samples // BATCH_SIZE
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")