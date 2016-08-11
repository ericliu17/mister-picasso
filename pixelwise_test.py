from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
from PIL import Image


class LSTMModel(object):

    def __init__(self, step, data):
        self.n, self.w, self.h, self.d = data.shape
        self.step = step
        self.data = data
        self.colors = None
        self.color_idx = None
        self.idx_color = None
        self.palettes = []
        self.next_colors = []
        self.model = None


    def lstm_model(self, diversities, iterations=50):
        X, y = self.vectorize()
        print('Build model...')
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=X.shape[1:],
                            return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(128, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(y.shape[1]))
        self.model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer)

        # train the model, output generated lines after each iteration
        for iteration in xrange(0, iterations):
            print()
            print('-' * 50)
            print('Iteration', iteration + 1)
            self.model.fit(X, y, batch_size=128, nb_epoch=1)

            if (iteration + 1) % 10 == 0:
                self.generate(diversities, iteration + 1)


    def vectorize(self):
        self.data = np.reshape(self.data, (self.n * self.w * self.h, self.d))
        # Map RGB values to tuple
        self.data = map(tuple, self.data)

        print('Getting colors...')
        self.colors = set(self.data)
        print('Total colors:', len(self.colors))

        self.color_idx = dict((c, i) for i, c in enumerate(self.colors))
        self.idx_color = dict((i, c) for i, c in enumerate(self.colors))

        for i in xrange(0, len(self.data) - self.w, self.step):
            self.palettes.append(self.data[i: i + self.w])
            self.next_colors.append(self.data[i + self.w])
        print('nb sequences:', len(self.palettes))

        print('Vectorization...')
        X = np.zeros((len(self.palettes), self.w, len(self.colors)),
                     dtype=np.bool)
        y = np.zeros((len(self.palettes), len(self.colors)), dtype=np.bool)

        for i, palette in enumerate(self.palettes):
            for j, color in enumerate(palette):
                X[i, j, self.color_idx[color]] = 1
            y[i, self.color_idx[self.next_colors[i]]] = 1

        return X, y


    def generate(self, diversities, iteration):
        start_idx = random.randint(0, len(self.data) - self.w - 1)

        for diversity in diversities:
            print()
            print('Diversity:', diversity)

            generated = []
            seed = self.data[start_idx : start_idx + self.w]

            for i in xrange(self.w * self.h):
                if (i + 1) % self.h == 0:
                    line = (i + 1) / self.h
                    print('Generating line {} of {}'.format(line, self.h))

                x = np.zeros((1, self.w, len(self.colors)))
                for j, color in enumerate(seed):
                    x[0, j, self.color_idx[color]] = 1.

                preds = self.model.predict(x, verbose=0)[0]
                next_idx = self.sample(preds, diversity)
                next_color = self.idx_color[next_idx]

                seed = seed[1:] + [next_color]
                generated.append(next_color)

            generated = np.array(generated).astype(np.uint8)
            generated = generated.reshape(self.w, self.h, self.d)
            filename = 'output_{}_{}.png'.format(iteration, diversity)
            print('Saving output image as {}'.format(filename))
            Image.fromarray(generated).save(filename)


    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
