import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.src.legacy.preprocessing.image import ImageDataGenerator

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

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

validate_datagen = ImageDataGenerator(rescale=1.0/255.0)

# load images from directory

train_generator = train_datagen.flow_from_directory(
    train_dir,
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

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(224,224),
    batch_size=BATCH_SIZE
    class_mode='categorical'
)