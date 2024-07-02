import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import save_model

# eval
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import seaborn as sns

# paths
train_dir = 'dataset/train'
test_dir = 'dataset/test'
validate_dir = 'dataset/valid'

# in general, batch size 32 is recommended
# https://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 100 # in practice, we won't reach this because of early stopping
LEARNING_RATE = 1e-4
NUM_CLASSES = 53

# learning rate scheduler
# see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
def lr_scheduler(epoch):
    initial_lr = LEARNING_RATE
    if epoch < 10:
        return initial_lr
    elif epoch < 30:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

####### SETTING UP DATA #######

# next, augment the data
# essentially, "mess" with the image to simulate real-world inconsistencies
# un-augmented images also used, in addition to these

train_datagen = ImageDataGenerator(
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range = 90,
    horizontal_flip = True,
    vertical_flip=True,
    fill_mode = 'nearest',
    shear_range = 0.2,
    zoom_range = 0.2,
    brightness_range=[0.7,1.2],
    channel_shift_range=50
)

validate_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()


# load images from directory

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)



####### CREATE MODEL #######

# load with pre-trained ImageNet
V2S_Model = EfficientNetV2S(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet')

for layer in V2S_Model.layers[-40:]:
    layer.trainable = True

# dropout is used to reduce overfitting issues
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = V2S_Model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x1 = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x1)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Concatenate()([x, x1])
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# callbacks for training
# early stopping to prevent overfitting
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_scheduler)

# compiling the model
# start with initial lower learning rate, to prevent large updates
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# train the model
history = model.fit(
    train_generator,
    steps_per_epoch=None,  
    epochs=EPOCHS,
    validation_data=validate_generator,
    validation_steps=None,
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)

# evaluate the model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")
 
# save the model in various formats, for testing purposes
save_model(model, 'model_4_keras.keras')
save_model(model, 'model_4_h5.h5')
#model.save_weights('model_4.weights.h5')

# further testing

# generate predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(class_report)

# for diagnostics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')