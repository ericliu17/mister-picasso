from __future__ import print_function
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
from img_processing import deprocess_image
from scipy.misc import imsave


def optimizer(evaluator, img_width, img_height, result_prefix, iterations=11):
    # initial state
    x = np.random.uniform(0, 255, (1, 3, img_width, img_height))
    x[0, 0, :, :] -= 103.939
    x[0, 1, :, :] -= 116.779
    x[0, 2, :, :] -= 123.68

    for i in xrange(iterations):
        print('Start of iteration', i + 1)
        start_time = time.time()
        # run scipy-based optimization (L-BFGS) over the pixels of the
        # generated image so as to minimize the neural style loss
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        img = deprocess_image(x.copy().reshape((3, img_width, img_height)))
        fname = result_prefix + '_at_iteration_{}.png'.format(i)
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration {} completed in {}s'.format(i, end_time - start_time))
        print()
