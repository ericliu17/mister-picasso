'''
From neural style transfer example with Keras,
except for spatial loss.
'''

from keras import backend as K
import sys
sys.path.insert(0, '../model/')
from spatial import spatial_loss


# weights of the different loss components
total_variation_weight = 1.
content_weight = 0.2
style_weight = 1.
spatial_weight = 0.02
last_layer = 'conv_4_2'


# compute the neural style loss
def calc_loss_grad(model, combo_img, img_width, img_height):
    # get the symbolic outputs of each "key" layer
    outputs_dict = dict([(layer.name, layer.output) \
                        for layer in model.layers \
                        if layer.name[:5] == 'conv_'])

    # combine these loss functions into a single scalar
    loss = K.variable(0.)

    # content loss
    # use last non-fully-connected layer
    layer_feats = outputs_dict[last_layer]
    base_feats = layer_feats[0, :, :, :]
    combo_feats = layer_feats[2, :, :, :]
    cl = content_loss(base_feats, combo_feats)
    loss += content_weight * cl

    # style loss and spatial loss
    for name in outputs_dict:
        layer_feats = outputs_dict[name]
        style_feats = layer_feats[1, :, :, :]
        combo_feats = layer_feats[2, :, :, :]
        spatial = spatial_loss(style_feats, combo_feats)
        sl = style_loss(style_feats, combo_feats, img_width, img_height)
        loss += (spatial_weight / len(outputs_dict)) * spatial
        loss += (style_weight / len(outputs_dict)) * sl

    # single scalar loss
    loss += total_variation_weight * \
            total_variation_loss(combo_img, img_width, img_height)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combo_img)

    return loss, grads


def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    return K.dot(features, K.transpose(features))


def style_loss(style, combo, img_width, img_height):
    S = gram_matrix(style)
    C = gram_matrix(combo)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / \
           (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combo):
    return K.sum(K.square(base - combo))


def total_variation_loss(x, img_width, img_height):
    assert K.ndim(x) == 4
    a = K.square(x[:, :, :img_width - 1, :img_height - 1] - \
                 x[:, :, 1:, :img_height - 1])
    b = K.square(x[:, :, :img_width - 1, :img_height - 1] - \
                 x[:, :, :img_width - 1, 1:])
    return K.sum(K.pow(a + b, 1.25))
