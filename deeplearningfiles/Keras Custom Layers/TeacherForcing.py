import numpy as np

import theano

import keras.backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.models import Sequential
from keras.layers import Dense, Merge, TimeDistributed, InputLayer

from KerasClassifiers import showModel


class TeacherForcingTopLevel(Merge):
    levelCounter = 0

    def __init__(self, input_dim, batch_size, final_output_dim, makeTimeDistributed=False, time_steps=1, **kwargs):
        self.output_dim = input_dim + final_output_dim
        self.batch_size = batch_size
        self.makeTimeDistributed = makeTimeDistributed
        if "name" in kwargs:
            myName = kwargs["name"]
        else:
            # not sure this ever runs...
            myName = "Teacher Forcing Top Level {0}".format(TeacherForcingTopLevel.levelCounter)
            TeacherForcingTopLevel.levelCounter += 1

        if makeTimeDistributed:
            batch_input_shape_Main = (batch_size, time_steps, input_dim)
            batch_input_shape_Teacher = (batch_size, time_steps, final_output_dim)
            batch_input_shape_LastEst = (batch_size, time_steps, final_output_dim)
        else:
            batch_input_shape_Main = (batch_size, input_dim)
            batch_input_shape_Teacher = (batch_size, final_output_dim)
            batch_input_shape_LastEst = (batch_size, final_output_dim)

        left_branch = InputLayer(batch_input_shape=batch_input_shape_Main, name='Main Input')

        lastInitialVal = np.zeros(batch_input_shape_LastEst, dtype=np.float32)
        self.lastOutputLayerVariable = K.variable(value=lastInitialVal, name="lastOutputLayer")
        self.teacherLayer = TeacherForcingInputLayer(final_output_dim,
                                                     lastLayerOutputVariable=self.lastOutputLayerVariable,
                                                     batch_input_shape=batch_input_shape_Teacher,
                                                     name="Last Output Layer of " + myName)
        right_branch_layer = self.teacherLayer
        self.teacherLayer.create_input_layer(self.teacherLayer.batch_input_shape, name="Teacher Input")

        super(TeacherForcingTopLevel, self).__init__([left_branch, right_branch_layer], mode='concat', **kwargs)


class TeacherForcingInputLayer(Layer):
    def __init__(self, output_dim, lastLayerOutputVariable, **kwargs):
        self.output_dim = output_dim
        self.input_dim = None
        self.lastLayerOutput = lastLayerOutputVariable

        self.stateful = True
        self.uses_learning_phase = True

        super(TeacherForcingInputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (bs x ... x input_dim)
        self.input_dim = input_shape[-1]

    def call(self, x, mask=None):
        xOut = K.in_train_phase(x, self.lastLayerOutput)
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

        K.set_value(self.lastLayerOutput, np.zeros((list(input_shape)[:-1] + [self.output_dim])))

    def call(self, x, mask=None):
        self.updates = [(self.lastLayerOutput, x)]
        return x


def runMain():
    input_dim = 10
    n_samples = 200
    final_output_dim = 2
    batch_size = 1  # i can't get a batch size of more then one b/c we would need to be batch stateful...?
    n_epochs = 10

    # generate dummy data that is related to the last sample
    np.random.seed(0)
    x_train = np.random.random((n_samples, input_dim))
    y_train = np.repeat(np.arange(1, n_samples + 1)[:, None], final_output_dim, axis=1)
    y_train_last = np.zeros_like(y_train)
    y_train_last[1:] = y_train[:-1]

    # make our final model
    final_model = Sequential()

    # make the top layer which takes our actual input and our true last output at train time or the estimated last output at test time
    merged = TeacherForcingTopLevel(input_dim=input_dim, batch_size=batch_size, final_output_dim=final_output_dim, name="Teacher Forcing Top Level")
    final_model.add(merged)

    # make actual layers of ANN
    final_model.add(Dense(final_output_dim, activation='linear', name="Dense Layer Final"))

    # make the last layer catch the final output to give back to the top layer in test time
    final_model.add(TeacherForcingOutputLayer(final_output_dim, lastLayerOutputVariable=merged.lastOutputLayerVariable))

    # compile our model
    final_model.compile(optimizer='rmsprop',
                        loss='mse',
                        metrics=['mse'])

    showModel(final_model)

    # train our model using the actual last values
    final_model.fit([x_train, y_train_last], y_train, nb_epoch=n_epochs, batch_size=batch_size)

    # we must reset the state before doing anything else
    final_model.reset_states()
    # get our score when we use the estimated outputs and feed in the true last which it should ignore
    scoreCheating = final_model.evaluate([x_train, y_train_last], y_train, batch_size=batch_size, verbose=1)
    print("\nThe final cheat score is {0}".format(scoreCheating))

    final_model.reset_states()
    score = final_model.evaluate([x_train, np.zeros_like(y_train_last)], y_train, batch_size=batch_size, verbose=1)
    print("\nThe final real score is {0}".format(score))

    assert np.all(np.isclose(score, scoreCheating)), "The ANN used the true last values and it shouldn't"

    # check our actual predictions
    final_model.reset_states()
    predictions = final_model.predict([x_train, np.zeros_like(y_train_last)], batch_size=batch_size, verbose=1)
    samplesToShow = 10
    print("\nThe predictions \n{0}".format(predictions[:samplesToShow]))
    print("\nThe truth \n{0}".format(y_train[:samplesToShow]))

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


def testTimeDistributed():
    input_dim = 10
    final_output_dim = 2
    batch_size = 3
    time_steps = 200
    n_epochs = 10

    # generate dummy data that is related to the last sample
    np.random.seed(0)
    x_train = np.random.random((batch_size, time_steps, input_dim))
    y_train = np.repeat(np.arange(1, time_steps + 1)[:, None], final_output_dim, axis=1)
    y_train = np.repeat(y_train[None, :, :], batch_size, axis=0)
    y_train_last = np.zeros_like(y_train)
    y_train_last[:, 1:] = y_train[:, :-1]

    # make our final model
    final_model = Sequential()

    # make the top layer which takes our actual input and our true last output at train time or the estimated last output at test time
    merged = TeacherForcingTopLevel(input_dim=input_dim,
                                    batch_size=batch_size,
                                    final_output_dim=final_output_dim,
                                    makeTimeDistributed=True,
                                    time_steps=time_steps,
                                    name="Teacher Forcing Top Level")
    final_model.add(merged)

    # make actual layers of ANN
    thisDense = Dense(final_output_dim, activation='linear', name="Dense Layer Final")
    thisDenseTD = TimeDistributed(thisDense, name="TD {0}".format(thisDense.name))
    final_model.add(thisDenseTD)

    # make the last layer catch the final output to give back to the top layer in test time
    TFOL = TeacherForcingOutputLayer(final_output_dim, lastLayerOutputVariable=merged.lastOutputLayerVariable, name="Teacher Forcing Output Layer")
    final_model.add(TFOL)

    # compile our model
    final_model.compile(optimizer='rmsprop',
                        loss='mse',
                        metrics=['mse'])

    # showModel(final_model)

    # train our model using the actual last values
    final_model.fit([x_train, y_train_last], y_train, nb_epoch=n_epochs, batch_size=batch_size)

    weights = final_model.get_weights()

    # do this again but set timesteps to 1
    # make our final model
    final_model = Sequential()

    # make the top layer which takes our actual input and our true last output at train time or the estimated last output at test time
    merged = TeacherForcingTopLevel(input_dim=input_dim,
                                    batch_size=batch_size,
                                    final_output_dim=final_output_dim,
                                    makeTimeDistributed=True,
                                    time_steps=1,
                                    name="Teacher Forcing Top Level")
    final_model.add(merged)

    # make actual layers of ANN
    thisDense = Dense(final_output_dim, activation='linear', name="Dense Layer Final")
    thisDenseTD = TimeDistributed(thisDense, name="TD {0}".format(thisDense.name))
    final_model.add(thisDenseTD)

    # make the last layer catch the final output to give back to the top layer in test time
    TFOL = TeacherForcingOutputLayer(final_output_dim, lastLayerOutputVariable=merged.lastOutputLayerVariable, name="Teacher Forcing Output Layer")
    final_model.add(TFOL)

    final_model.set_weights(weights=weights)
    # compile our model
    final_model.compile(optimizer='rmsprop',
                        loss='mse',
                        metrics=['mse'])

    # check our actual predictions
    final_model.reset_states()
    predictions = np.zeros((batch_size, time_steps, final_output_dim))
    for i in range(time_steps):
        predictions[:, i:i + 1, :] = final_model.predict([x_train[:, i:i + 1, :], np.zeros_like(y_train_last[:, i:i + 1, :])],
                                                         batch_size=batch_size,
                                                         verbose=1)

    samplesToShow = 10
    print("\nThe predictions \n{0}".format(predictions[:samplesToShow]))
    print("\nThe truth \n{0}".format(y_train[:samplesToShow]))

    # getOutputFunction = K.function(inputs=merged.input + [K.learning_phase()],
    #                                outputs=[final_model.layers[-1].output, merged.teacherLayer.lastLayerOutput],
    #                                updates=final_model.updates)
    # final_model.reset_states()
    # steps_to_show = 10
    # for i in range(time_steps):
    #     outputer = getOutputFunction([x_train[:, i:i + 1], y_train_last[:, i:i + 1], 1])
    #     print "Run number {i} train output:{outputer} lastVar:{lastVar}".format(i=i, outputer=outputer[0][0, :steps_to_show, :],
    #                                                                             lastVar=outputer[1][0, 0])
    # final_model.reset_states()
    # for i in range(time_steps):
    #     outputer = getOutputFunction([x_train[:, i:i + 1], np.zeros_like(y_train_last[:, i:i + 1]), 0])
    #     print "Run number {i} test output:{outputer} lastVar:{lastVar}".format(i=i, outputer=outputer[0][0, :steps_to_show, :],
    #                                                                            lastVar=outputer[1][0, 0])


if __name__ == '__main__':
    # runMain()
    testTimeDistributed()
