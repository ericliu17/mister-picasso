from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
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
        self.palettes = []
        self.next_colors = []
        self.model = None


    def lstm_model(self, diversities, iterations=50, generations=10):
        X, y = self.vectorize()
        print('Build model...')
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(None, X.shape[2]),
                            return_sequences=True))
        self.model.add(TimeDistributed(Dense(y.shape[2])))
        self.model.add(Activation('relu'))

        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

        # train the model, output generated lines after each iteration
        for iteration in xrange(0, iterations):
            print()
            print('-' * 50)
            print('Iteration', iteration + 1)
            self.model.fit(X, y, batch_size=128, nb_epoch=10000)

            if (iteration + 1) % generations == 0:
                self.generate(diversities, iteration + 1)


    def vectorize(self):
        self.data = np.reshape(self.data, (self.n * self.w * self.h, self.d))
        # Select only column 0 values (column 1 all 255)
        self.data = self.data[:,0]
        self.data = np.reshape(self.data, (self.w, self.h))

        for i in xrange(self.h - 1):
            self.palettes.append(self.data[i])
            self.next_colors.append(self.data[i + 1])
        print('nb sequences:', len(self.palettes))

        X, y = np.array(self.palettes), np.array(self.next_colors)
        X, y = X[None, :, :], y[None, :, :]

        return X, y


    def generate(self, diversities, iteration):
        start_idx = random.randint(0, self.data.shape[0] - 1)

        for diversity in diversities:
            print()
            print('Diversity:', diversity)

            generated = []
            seed = self.data[start_idx]
            seed = seed[None, None, :]

            for i in xrange(self.h):
                if (i + 1) % self.h == 0:
                    line = (i + 1) / self.h
                    print('Generating line {} of {}'.format(line, self.h))

                preds = self.model.predict(seed, verbose=0)[0]
                # next_idx = self.sample(preds, diversity)
                next_idx = np.argmax(preds)
                next_color = self.data[next_idx]

                seed = next_color[None, None, :]
                generated.append(next_color)

            generated = np.array(generated).astype(np.uint8)
            # add back 255
            generated = generated.reshape(self.n * self.w * self.h, 1)
            add = np.full((self.n * self.w * self.h, 1), 255, dtype = np.uint8)
            generated = np.concatenate((generated, add), axis=1)
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
