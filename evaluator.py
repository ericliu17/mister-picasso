import numpy as np
from keras import backend as K


# helper class to retrieve loss and gradients for scipy optimizer
class Evaluator(object):
    def __init__(self, loss, grads, combo_img, img_width, img_height):
        self.loss_value = None
        self.grads_values = None
        self.img_width = img_width
        self.img_height = img_height
        self.f_outputs = self.set_f_outputs(combo_img, loss, grads)

    def set_f_outputs(self, combo_img, loss, grads):
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)
        return K.function([combo_img], outputs)

    def loss(self, x):
        assert self.loss_value is None
        self.loss_value, self.grad_values = self.eval_loss_and_grads(x)
        return self.loss_value

    # only for retrieving grad values
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value, self.grad_values = None, None
        return grad_values

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, 3, self.img_width, self.img_height))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values
