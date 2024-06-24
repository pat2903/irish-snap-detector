import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_img(path, img_shape=224, channels=3):
    """
    Reads in an image and returns a processed image tensor (default shape 224, 224, 3)
    """

    img = tf.io.read_file(path)
    # decode it into a tensor
    img = tf.image.decode_image(img, channels)
    img = tf.image.resize(img, [img_shape, img_shape])
    # normalise the image for compatibility
    img = img / 255
    
    return img

# test code
# note: .HEIC files do not work
path = "ace_diamond.jpg"
preped_img = load_and_process_img(path)
plt.imshow(preped_img)
plt.show()