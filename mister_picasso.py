'''
Based on Neural style transfer example with Keras.
# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from __future__ import print_function
import argparse
from keras import backend as K
from img_processing import preprocess_image, deprocess_image
from model import model
from loss import calc_loss_grad
from evaluator import Evaluator
from optimization import optimizer

# parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
# parser.add_argument('base_img_path', metavar='base', type=str,
#                     help='Path to the image to transform.')
# parser.add_argument('style_img_path', metavar='ref', type=str,
#                     help='Path to the style reference image.')
# parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                     help='Prefix for the saved results.')
#
# args = parser.parse_args()
# base_img_path = args.base_img_path
# style_img_path = args.style_img_path
# result_prefix = args.result_prefix
# weights_path = 'vgg16_weights.h5'

base_img_path = 'images/EricSquare.jpg'
style_img_path = 'images/ambroise.jpg'
result_prefix = 'result'
weights_path = 'vgg16_weights.h5'

# dimensions of the generated picture
img_width = 128
img_height = 128
assert img_height == img_width, \
       'Due to the use of the Gram matrix, width and height must match.'

# get tensor representations of images
base_img = K.variable(preprocess_image(base_img_path,
                                       img_width,
                                       img_height))
style_img = K.variable(preprocess_image(style_img_path,
                                        img_width,
                                        img_height))
combo_img = K.placeholder((1, 3, img_width, img_height))
# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_img, style_img, combo_img],
                             axis=0)

print('Loading model with VGG16 network weights...')
model = model(weights_path, input_tensor, img_width, img_height)
loss, grads = calc_loss_grad(model, combo_img, img_width, img_height)
evaluate = Evaluator(loss, grads, combo_img, img_width, img_height)
optimizer(evaluate, img_width, img_height, result_prefix)
