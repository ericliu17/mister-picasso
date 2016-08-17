from keras import backend as K
from theano.tensor.nnet.neighbours import images2neibs
from theano.tensor.nnet import conv2d
import time


def spatial_loss(style, combo, shape=2):
    # start = time.time()
    style_patches, style_norm = make_patches(style, shape)
    # end = time.time()
    # print end - start

    # start = time.time()
    combo_patches, combo_norm = make_patches(combo, shape)
    # end = time.time()
    # print end - start

    # start = time.time()
    patch_ids = find_matches(combo_patches, combo_norm,
                             style_patches / style_norm)
    # end = time.time()
    # print end - start

    best_patches = K.reshape(style_patches[patch_ids], K.shape(combo_patches))

    return K.sum(K.square(best_patches - combo_patches)) / shape ** 2


def make_patches(x, shape):
    x = K.expand_dims(x, 0)

    patches = images2neibs(x, (shape, shape))
    patches = K.reshape(patches, (K.shape(x)[1],
                                  K.shape(patches)[0] / K.shape(x)[1],
                                  shape, shape))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(1,2,3),
                                keepdims=True))

    return patches, patches_norm


def find_matches(combo, combo_norm, style_normalized):
    convs = conv2d(combo, style_normalized)
    return K.argmax(convs / combo_norm, axis=1)


# def to_numpy(tensor):
#     return K.eval(tensor)
