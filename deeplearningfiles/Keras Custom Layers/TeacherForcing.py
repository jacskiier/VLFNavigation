import numpy as np

import keras.backend as K
from keras.engine.topology import Layer
from keras.engine import InputSpec, Input
from keras.models import Sequential, Model
from keras.layers import Dense, Merge, TimeDistributed, InputLayer

from KerasClassifiers import showModel


class TeacherForcingTopLevel(Merge):
    levelCounter = 0

    def __init__(self, input_dim=None, batch_size=None, final_output_dim=None, makeTimeDistributed=False, time_steps=1, **kwargs):
        assert input_dim is not None, "Must give an input_dim"
        assert batch_size is not None, "Must give batch_size"
        assert final_output_dim is not None, "Must give final_output_dim"
        self.output_dim = input_dim + final_output_dim
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.final_output_dim = final_output_dim
        self.time_steps = time_steps
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

        self.left_branch = left_branch
        self.right_branch = right_branch_layer
        super(TeacherForcingTopLevel, self).__init__([left_branch, right_branch_layer], mode='concat', **kwargs)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'batch_size': self.batch_size,
                  'final_output_dim': self.final_output_dim,
                  'time_steps': self.time_steps,
                  'makeTimeDistributed': self.makeTimeDistributed,
                  'input_dim': self.input_dim}
        base_config = super(TeacherForcingTopLevel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TeacherForcingInputLayer(Layer):
    def __init__(self, output_dim=None, lastLayerOutputVariable=None, **kwargs):
        assert output_dim is not None, "Must give output_dim"
        self.output_dim = output_dim
        self.input_dim = None
        self.lastLayerOutput = lastLayerOutputVariable

        self.stateful = True
        self.uses_learning_phase = True

        super(TeacherForcingInputLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(TeacherForcingInputLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # input_shape = (bs x ... x input_dim)
        self.input_dim = input_shape[-1]

    def call(self, x, mask=None):
        assert self.lastLayerOutput is not None, "lastLayerOutput was never setup"
        xOut = K.in_train_phase(x, self.lastLayerOutput)
        return xOut


class TeacherForcingOutputLayer(Layer):
    def __init__(self, output_dim=None, lastLayerOutputVariable=None, **kwargs):
        assert output_dim is not None, "Must give output_dim"
        self.output_dim = output_dim
        self.input_dim = None
        self.lastLayerOutput = lastLayerOutputVariable
        self.updates = []
        self.stateful = True
        super(TeacherForcingOutputLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(TeacherForcingOutputLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        if self.lastLayerOutput is not None:
            K.set_value(self.lastLayerOutput, np.zeros((list(input_shape)[:-1] + [self.output_dim])))

    def call(self, x, mask=None):
        assert self.lastLayerOutput is not None, "lastLayerOutput was never setup"
        self.updates = [(self.lastLayerOutput, x)]
        return x


class TeacherForcingModel(Model):
    def __init__(self, original_model, **kwargs):
        """
        The init function for the teacher forcing wrapper

        :param original_model: the original model that you want to wrap teacher learning around
        :type original_model: Model

        :param kwargs: extra arguments
        :type kwargs: dict
        """
        self.original_model = original_model
        output_shape = original_model.get_output_shape_at(0)
        self.final_output_dim = output_shape[-1]

        input_shape = original_model.get_input_shape_at(0)
        self.input_dim = input_shape[-1] - self.final_output_dim
        self.batch_size = input_shape[0]

        if len(input_shape) > 2:
            self.makeTimeDistributed = True
            time_steps = input_shape[1]
        else:
            self.makeTimeDistributed = False
            time_steps = 1

        TFTL = TeacherForcingTopLevel(input_dim=self.input_dim,
                                      batch_size=self.batch_size,
                                      final_output_dim=self.final_output_dim,
                                      makeTimeDistributed=self.makeTimeDistributed,
                                      time_steps=time_steps,
                                      name="Teacher Forcing Top Level")
        TFTLTensor = TFTL.get_output_at(0)

        originalOutTensor = original_model(TFTLTensor)

        # make the last layer catch the final output to give back to the top layer in test time
        TFOL = TeacherForcingOutputLayer(self.final_output_dim, lastLayerOutputVariable=TFTL.lastOutputLayerVariable,
                                         name="Teacher Forcing Output Layer")
        output = TFOL(originalOutTensor)

        # build our one timestep model
        self.oneTimestepModel = None
        self.buildOneTimestepModel()

        super(TeacherForcingModel, self).__init__(input=[TFTL.left_branch.input, TFTL.right_branch.input], output=[output], **kwargs)

    def buildOneTimestepModel(self):
        # rebuild this model BUT make timesteps be 1
        TFTL = TeacherForcingTopLevel(input_dim=self.input_dim,
                                      batch_size=self.batch_size,
                                      final_output_dim=self.final_output_dim,
                                      makeTimeDistributed=self.makeTimeDistributed,
                                      time_steps=1,
                                      name="Teacher Forcing Top Level")
        TFTLTensor = TFTL.get_output_at(0)
        originalOutTensor = self.original_model(TFTLTensor)
        # make the last layer catch the final output to give back to the top layer in test time
        TFOL = TeacherForcingOutputLayer(self.final_output_dim, lastLayerOutputVariable=TFTL.lastOutputLayerVariable,
                                         name="Teacher Forcing Output Layer")
        output = TFOL(originalOutTensor)
        self.oneTimestepModel = Model(input=[TFTL.left_branch.input, TFTL.right_branch.input], output=[output])

    def setOneTimestepModelWeights(self):
        # set the weights of this model to the new one timestep model
        weights = self.get_weights()
        self.oneTimestepModel.set_weights(weights=weights)

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        if self.makeTimeDistributed is False:
            batch_size = 1
            return super(TeacherForcingModel, self).evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose)
        else:
            kerasRowMultiplier = x.shape[0] / batch_size
            time_steps = x.shape[1]
            yShape = (batch_size, 1, self.final_output_dim)

            self.setOneTimestepModelWeights()
            predictions = np.zeros((batch_size * kerasRowMultiplier, time_steps, self.final_output_dim))
            for k in range(kerasRowMultiplier):
                for i in range(time_steps):
                    batchSlice = slice(k * batch_size, (k + 1) * batch_size)
                    predictions[batchSlice, i:i + 1, :] = self.oneTimestepModel.evaluate(
                        x=[x[batchSlice, i:i + 1, :], np.zeros(yShape)],
                        y=y,
                        batch_size=batch_size,
                        verbose=0)
            return predictions

    def predict(self, x, batch_size=32, verbose=0):
        if self.makeTimeDistributed is False:
            batch_size = 1
            return super(TeacherForcingModel, self).predict(x, batch_size=batch_size, verbose=verbose)
        else:
            kerasRowMultiplier = x.shape[0] / batch_size
            time_steps = x.shape[1]
            yShape = (batch_size, 1, self.final_output_dim)

            self.setOneTimestepModelWeights()
            predictions = np.zeros((batch_size * kerasRowMultiplier, time_steps, self.final_output_dim))
            for k in range(kerasRowMultiplier):
                for i in range(time_steps):
                    batchSlice = slice(k * batch_size, (k + 1) * batch_size)
                    predictions[batchSlice, i:i + 1, :] = self.oneTimestepModel.predict(
                        x=[x[batchSlice, i:i + 1, :], np.zeros(yShape)],
                        batch_size=batch_size,
                        verbose=0)
            return predictions


def runMain():
    input_dim = 10
    n_samples = 300
    final_output_dim = 2
    batch_size = 1  # i can't get a batch size of more then one b/c we would need to be batch stateful...?
    n_epochs = 20

    # generate dummy data that is related to the last sample
    np.random.seed(0)
    x_train = np.random.random((n_samples, input_dim))
    y_train = np.repeat(np.arange(1, n_samples + 1)[:, None], final_output_dim, axis=1)
    y_train_last = np.zeros_like(y_train)
    y_train_last[1:] = y_train[:-1]

    # make actual layers of ANN in original model
    original_model = Sequential(name="Original Model")
    original_model.add(
        Dense(final_output_dim, batch_input_shape=(batch_size, input_dim + final_output_dim), activation='linear', name="Dense Layer Final"))

    # Wrap the teacher forcing model around the original model
    final_model = TeacherForcingModel(original_model=original_model)

    # compile our model
    final_model.compile(optimizer='rmsprop',
                        loss='mse',
                        metrics=['mse'])

    showModel(final_model)

    # train our model using the actual last values
    final_model.fit([x_train, y_train_last], y_train, nb_epoch=n_epochs, batch_size=batch_size, verbose=2)

    # we must reset the state before doing anything else
    final_model.reset_states()
    # get our score when we use the estimated outputs and feed in the true last which it should ignore
    scoreCheating = final_model.evaluate([x_train, y_train_last], y_train, batch_size=batch_size, verbose=2)
    print("\nThe final cheat score is {0}".format(scoreCheating))

    final_model.reset_states()
    score = final_model.evaluate([x_train, np.zeros_like(y_train_last)], y_train, batch_size=batch_size, verbose=2)
    print("\nThe final real score is {0}".format(score))

    assert np.all(np.isclose(score, scoreCheating)), "The ANN used the true last values and it shouldn't"

    # check our actual predictions
    final_model.reset_states()
    predictions = final_model.predict([x_train, np.zeros_like(y_train_last)], batch_size=batch_size, verbose=2)
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
    batch_size = 7
    time_steps = 100
    n_epochs = 1000
    kerasRowMultiplier = 5

    # generate dummy data that is related to the last sample
    np.random.seed(0)
    x_train = np.random.random((batch_size, kerasRowMultiplier * time_steps, input_dim))
    y_train = np.repeat(np.arange(1, kerasRowMultiplier * time_steps + 1)[:, None], final_output_dim, axis=1)
    y_train = np.repeat(y_train[None, :, :], batch_size, axis=0)
    y_train_last = np.zeros_like(y_train)
    y_train_last[:, 1:] = y_train[:, :-1]

    x_train = np.reshape(x_train, (batch_size, kerasRowMultiplier, time_steps, input_dim), order='C')
    x_train = np.reshape(x_train, (batch_size * kerasRowMultiplier, time_steps, input_dim), order='F')
    y_train = np.reshape(y_train, (batch_size, kerasRowMultiplier, time_steps, final_output_dim), order='C')
    y_train = np.reshape(y_train, (batch_size * kerasRowMultiplier, time_steps, final_output_dim), order='F')
    y_train_last = np.reshape(y_train_last, (batch_size, kerasRowMultiplier, time_steps, final_output_dim), order='C')
    y_train_last = np.reshape(y_train_last, (batch_size * kerasRowMultiplier, time_steps, final_output_dim), order='F')

    # make original model
    original_model = Sequential()
    thisDense = Dense(output_dim=final_output_dim,
                      activation='linear',
                      name="Dense Layer Final")
    thisDenseTD = TimeDistributed(layer=thisDense,
                                  batch_input_shape=(batch_size, time_steps, input_dim + final_output_dim),
                                  name="TD {0}".format(thisDense.name))
    original_model.add(thisDenseTD)

    # Wrap the teacher forcing model around the original model
    final_model = TeacherForcingModel(original_model=original_model)

    # compile our model
    final_model.compile(optimizer='rmsprop',
                        loss='mse',
                        metrics=['mse'])

    # showModel(final_model)

    # train our model using the actual last values
    final_model.fit([x_train, y_train_last], y_train, nb_epoch=n_epochs, batch_size=batch_size, verbose=2)

    # predictions = predictOneByOne(model=final_model, x=x_train, batch_size=batch_size, final_output_dim=final_output_dim)
    predictions = final_model.predict(x=x_train, batch_size=batch_size, verbose=2)

    batchesToShow = kerasRowMultiplier * batch_size
    batch_skip = batch_size
    samplesToShow = 10
    for batchToShow in range(0, batchesToShow, batch_skip):
        print("\nBatch {0}".format(batchToShow))
        print("\nThe predictions \n{0}".format(predictions[batchToShow, :samplesToShow]))
        print("\nThe truth \n{0}".format(y_train[batchToShow, :samplesToShow]))

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
