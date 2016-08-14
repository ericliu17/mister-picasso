import numpy as np
from keras import backend as K
from theano.tensor.nnet.neighbours import images2neibs


def make_patches(x, patch_size, patch_stride):
    # Break image into patches
    x = K.expand_dims(x, 0)
    patches = images2neibs(x, (patch_size, patch_size),
                              (patch_stride, patch_stride), mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (K.shape(x)[1],
                                  K.shape(patches)[0] / K.shape(x)[1],
                                  patch_size,
                                  patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    patches_norm = K.sqrt(K.sum(K.square(patches),
                                axis=(1,2,3),
                                keepdims=True))

    return patches, patches_norm


def find_matches(a, a_norm, b):
    '''For each patch in A, find the best matching patch in B'''
    # we want cross-correlation here so flip the kernels
    convs = K.conv2d(a, b[:, :, ::-1, ::-1], border_mode='valid')
    argmax = K.argmax(convs / a_norm, axis=1)
    return argmax
