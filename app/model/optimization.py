'''
From neural style transfer example with Keras.
'''

from __future__ import print_function
import time
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from img_processing import deprocess_image


def optimizer(evaluator, img_width, img_height, result_prefix, iterations=10):
    # initial state
    x = np.random.uniform(0, 255, (1, 3, img_width, img_height))
    x[0, 0, :, :] -= 103.939
    x[0, 1, :, :] -= 116.779
    x[0, 2, :, :] -= 123.68

    for i in xrange(iterations):
        print('Start of iteration', i + 1)
        start_time = time.time()
        # run scipy-based optimization (L-BFGS) over the pixels of the
        # generated image so as to minimize the neural loss
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        if i + 1 == iterations:
            img = deprocess_image(x.copy().reshape((3, img_width, img_height)))
            fname = result_prefix + '.png'
            imsave(fname, img)
            print('Image saved as', fname)
        end_time = time.time()
        print('Iteration {} completed in {}s'.format(i, end_time - start_time))
        print()
