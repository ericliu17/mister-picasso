'''
From neural style transfer example with Keras.
'''

from scipy.misc import imread, imresize
import numpy as np


# open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_width, img_height):
    img = imresize(imread(image_path), (img_width, img_height))
    img = img[:, :, ::-1].astype('float64')
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0)


# convert a tensor into a valid image
def deprocess_image(x):
    x = x.transpose((1, 2, 0))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')
