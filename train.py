import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import save_model

# paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'
validate_dir = 'dataset/valid'

# in general, batch size 32 is recommended
# https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
BATCH_SIZE = 32
IMG_SIZE = 224  

# next, augment the data
# essentially, "mess" with the image to simulate real-world inconsistencies
# un-augmented images also used, in addition to these

train_datagen = ImageDataGenerator(
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range = 15,
    horizontal_flip = True,
    fill_mode = 'nearest',
    shear_range = 0.1,
    zoom_range = 0.2,
    brightness_range=[0.8,1.2]
)

validate_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()


# load images from directory

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"Number of classes: {train_generator.num_classes}")
for class_name, class_index in train_generator.class_indices.items():
    print(f"{class_index}: {class_name}")

# load with pre-trained ImageNet
# exclude top layers for transfer learning
B3_model = EfficientNetB3(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

# unfreeze the last 30 layers
# allow model to adapt to dataset
for layer in B3_model.layers[-30:]:
    layer.trainable = True

# dropout is used to reduce overfitting issues
model = tf.keras.Sequential([
    B3_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# compiling the model
# start with initial lower learning rate, to prevent large updates
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks for training
# early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5)

# train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=50,
    validation_data=validate_generator,
    validation_steps=validate_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr]
)

# evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

save_model(model, 'model_3_keras.keras')
save_model(model, 'model_3_h5.h5')
model.save_weights('model_3.weights.h5')