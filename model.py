from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
from PIL import Image


class lstmModel(object):

    def __init__(self, size, step, data):
        self.size = size
        self.maxlen = self.size[0]
        self.step = step
        self.data = data
        self.colors = None
        self.color_idx = None
        self.idx_color = None
        self.palettes = []
        self.next_colors = []


    def vectorize(self):
        print('Getting colors...')
        self.colors = set(self.data)
        print('Total colors:', len(self.colors))
        self.color_idx = dict((c, i) for i, c in enumerate(self.colors))
        self.idx_color = dict((i, c) for i, c in enumerate(self.colors))

        for i in range(0, len(self.data) - self.maxlen, self.step):
            self.palettes.append(self.data[i: i + self.maxlen])
            self.next_colors.append(self.data[i + self.maxlen])
        print('nb sequences:', len(self.palettes))

        print('Vectorization...')
        X = np.zeros((len(self.palettes), self.maxlen, len(self.colors)),
                     dtype=np.bool)
        y = np.zeros((len(self.palettes), len(self.colors)), dtype=np.bool)

        for i, palette in enumerate(self.palettes):
            for j, color in enumerate(palette):
                X[i, j, self.color_idx[color]] = 1
            y[i, self.color_idx[self.next_colors[i]]] = 1

        return X, y


    def model(self, X, y, diversities):
        # build the model: 2 stacked LSTM
        print('Build model...')
        model = Sequential()
        # model.add(LSTM(128, input_shape=(self.maxlen, len(self.colors))))
        # model.add(Dense(len(self.colors)))
        # model.add(Activation('softmax'))
        model.add(LSTM(len(self.colors)*2, input_shape=(self.maxlen,
                                                        len(self.colors))))
        # model.add(Dropout(0.2))
        # model.add(LSTM(512, 512, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(len(self.colors)))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # train the model, output generated text after each iteration
        for iteration in range(1, 60):
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(X, y, batch_size=128, nb_epoch=1)
            self.generate(X, diversities, iteration)


    def generate(self, X, diversities, iteration):
        start_idx = random.randint(0, len(self.data) - self.maxlen - 1)

        for diversity in diversities:
            print()
            print('----- diversity:', diversity)

            generated = np.array([])
            seed = self.data[start_idx: start_idx + self.maxlen]

            for i in range(self.maxlen**2):
                x = np.zeros((1, self.maxlen, len(self.colors)))
                for j, color in enumerate(seed):
                    x[0, j, self.color_idx[color]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_idx = self.sample(preds, diversity)
                next_color = self.idx_color[next_idx]

                seed = np.append(seed[1:], next_color)
                generated = np.append(generated, next_color)

            generated = generated.reshape(self.size[0], self.size[1], 3)
            filename = 'output_{}_{}.png'.format(iteration, diversity)
            print('Saving output image as {}'.format(filename))
            Image.fromarray(generated).save(filename)


    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
