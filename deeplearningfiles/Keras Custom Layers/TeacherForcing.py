import numpy as np

import theano

from keras.engine.topology import Layer
from keras.layers import Merge
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.engine import InputSpec

from KerasClassifiers import showModel


class TeacherForcingInputLayer(Layer):
    def __init__(self, output_dim, lastLayerOutputVariable, **kwargs):
        self.output_dim = output_dim
        self.input_dim = None
        self.lastLayerOutput = lastLayerOutputVariable

        self.stateful = True
        self.uses_learning_phase = True

        super(TeacherForcingInputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (bs x input_dim)
        self.input_dim = input_shape[1]

    def call(self, x, mask=None):
        xOut = K.in_train_phase(x, self.lastLayerOutput)
        # xOut = x
        return xOut


class TeacherForcingOutputLayer(Layer):
    def __init__(self, output_dim, lastLayerOutputVariable, **kwargs):
        self.output_dim = output_dim
        self.input_dim = None
        self.lastLayerOutput = lastLayerOutputVariable
        self.updates = []
        self.stateful = True
        super(TeacherForcingOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (bs x input_dim)
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[1]

        if self.stateful:
            self.reset_states()

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')

        K.set_value(self.lastLayerOutput, np.zeros((input_shape[0], self.output_dim)))

    def call(self, x, mask=None):
        self.updates = [(self.lastLayerOutput, x)]
        return x


def runMain():
    input_dim = 1
    n_samples = 1000
    final_output_dim = 2
    batch_size = 1
    n_epochs = 10

    # generate dummy data
    np.random.seed(0)
    x_train = np.random.random((n_samples, input_dim))
    y_train = np.repeat(np.arange(1, n_samples + 1)[:, None], final_output_dim, axis=1)
    y_train_last = np.zeros_like(y_train)
    y_train_last[1:] = y_train[:-1]
    # y_train_last = y_train

    final_model = Sequential()

    left_branch = Sequential(name="Input Pass Through Sequential")
    left_branch.add(Dense(output_dim=input_dim,
                          activation='linear',
                          weights=[np.eye(input_dim), np.zeros((input_dim,))],
                          trainable=False,
                          batch_input_shape=(batch_size, input_dim),
                          name="Input Pass Through Layer"))

    right_branch = Sequential(name="Last Output Sequential")
    lastInitialVal = np.zeros((batch_size, final_output_dim), dtype=np.float32)
    lastOutputLayerVariable = K.variable(value=lastInitialVal, name="lastOutputLayer")
    teacherLayer = TeacherForcingInputLayer(final_output_dim,
                                            lastLayerOutputVariable=lastOutputLayerVariable,
                                            batch_input_shape=(batch_size, final_output_dim),
                                            name="Last Output Layer")
    right_branch.add(teacherLayer)

    merged = Merge([left_branch, right_branch], mode='concat', name="Merge Input and Last Output")
    final_model.add(merged)

    # final_model.add(Dense(50, activation='tanh', name="Dense Layer 1"))
    # final_model.add(Dense(50, activation='tanh', name="Dense Layer 2"))
    final_model.add(Dense(final_output_dim, activation='linear', name="Dense Layer Final"))

    final_model.add(TeacherForcingOutputLayer(final_output_dim, lastLayerOutputVariable=lastOutputLayerVariable))

    final_model.compile(optimizer='rmsprop',
                        loss='mse',
                        metrics=['mse'])

    # showModel(final_model)

    final_model.fit([x_train, y_train_last], y_train, nb_epoch=n_epochs, batch_size=batch_size)

    final_model.reset_states()
    score = final_model.evaluate([x_train, y_train_last], y_train, batch_size=batch_size, verbose=1)
    print("\nThe final cheat score is {0}".format(score))

    final_model.reset_states()
    score = final_model.evaluate([x_train, np.zeros_like(y_train_last)], y_train, batch_size=batch_size, verbose=1)
    print("\nThe final real score is {0}".format(score))

    final_model.reset_states()
    predictions = final_model.predict([x_train, np.zeros_like(y_train_last)], batch_size=batch_size, verbose=1)
    print("\nThe predictions {0}".format(predictions[:10]))

    # getOutputFunction = K.function(inputs=[left_branch.input, right_branch.input, K.learning_phase()],
    #                                outputs=[final_model.layers[-1].output, lastOutputLayerVariable],
    #                                updates=final_model.updates)
    # K.set_value(lastOutputLayerVariable, lastInitialVal)
    # for i in range(10):
    #     outputer = getOutputFunction([x_train[i:i + 1, :], y_train_last[i:i + 1, :], 1])
    #     print "Run number {i} train output:{outputer} lastVar:{lastVar}".format(i=i, outputer=outputer[0][0, 0], lastVar=outputer[1][0, 0])
    # K.set_value(lastOutputLayerVariable, lastInitialVal)
    # for i in range(10):
    #     outputer = getOutputFunction([x_train[i:i + 1, :], np.zeros_like(y_train_last[i:i + 1, :]), 0])
    #     print "Run number {i} test output:{outputer} lastVar:{lastVar}".format(i=i, outputer=outputer[0][0, 0], lastVar=outputer[1][0, 0])


if __name__ == '__main__':
    runMain()
