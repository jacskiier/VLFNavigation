import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import SGD
from ClassificationUtils import load_data
import pandas as pd
import os
import CreateUtils
import KerasClassifiers

if __name__ == '__main__':
    thingToDo = 4
    if thingToDo == 0:
        # for a single-input model with 2 classes (binary):

        model = Sequential()
        model.add(Dense(1, input_dim=784, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # generate dummy data
        import numpy as np

        data = np.random.random((1000, 784))
        labels = np.random.randint(2, size=(1000, 1))

        # train the model, iterating on the data in batches
        # of 32 samples
        model.fit(data, labels, nb_epoch=10, batch_size=32)
    elif thingToDo == 1:
        (X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.pkl",
                                                              nb_words=None,
                                                              skip_top=0,
                                                              maxlen=None)

        max_features = 1000
        maxlen = None
        model = Sequential()
        model.add(Embedding(max_features, 256, input_length=maxlen))
        model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, batch_size=16, nb_epoch=10)
        score = model.evaluate(X_test, y_test, batch_size=16)
        print("The final score is {0}".format(score))

    elif thingToDo == 2:
        datasetFileName = CreateUtils.getDatasetFile( featureSetName="MFCCFeaturesLargeWindowLowFreq",datasetName= "threeClassProblemSequence")
        datasets, inputs, outputs, max_batch_size = load_data(datasetFileName, makeSharedData=False)
        X_train = datasets[0][0]
        y_trainLabel = datasets[0][1]
        y_train = np.zeros((X_train.shape[0], np.max(y_trainLabel) + 1))
        for rower in np.arange(X_train.shape[0]):
            y_train[rower, y_trainLabel[rower]] = 1

        X_test = datasets[2][0]
        y_testLabel = datasets[2][1]
        y_test = np.zeros((X_test.shape[0], np.max(y_testLabel) + 1))
        for rower in np.arange(X_test.shape[0]):
            y_test[rower, y_testLabel[rower]] = 1

        model = Sequential(name="Input Layer 0")
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(64, input_dim=50 * 15 * 4, init='uniform', name="Layer 1"))
        model.add(Activation('tanh', name="tanh Layer 1"))
        model.add(Dense(64, init='uniform', name="Layer 2"))
        model.add(Activation('tanh', name="tanh Layer 2"))
        model.add(Dropout(0.5, name="Dropout Layer 2"))
        model.add(Dense(3, init='uniform', name="Layer 3"))
        model.add(Activation('softmax', name="softmax Layer 3"))

        KerasClassifiers.showModel(model)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        print("Start fitting")
        model.fit(X_train, y_train,
                  nb_epoch=5,
                  batch_size=100)
        score = model.evaluate(X_test, y_test, batch_size=100)
        print("\n")
        print("model metrics {0}".format(model.metrics_names))
        print("Score is:{0}".format(score))
        predictions = model.predict_proba(X_test)
        print(predictions)

        from keras.utils.visualize_util import plot

        athing1 = plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    elif thingToDo == 3:
        datasetFileName = CreateUtils.getDatasetFile(featureSetName="MFCCFeaturesLargeWindowLowFreq", datasetName="threeClassProblemSequence")
        timesteps = 1000
        data_dim = 100
        datasets, inputs, outputs, max_batch_size = load_data(datasetFileName, makeSharedData=False)

        X_train = datasets[0][0]
        X_train = np.reshape(X_train, (X_train.shape[0], timesteps, X_train.shape[1] / timesteps))
        y_trainLabel = datasets[0][1]
        y_train = np.zeros((X_train.shape[0], np.max(y_trainLabel) + 1))
        for rower in np.arange(X_train.shape[0]):
            y_train[rower, y_trainLabel[rower, 0]] = 1

        X_valid = datasets[1][0]
        X_valid = np.reshape(X_valid, (X_valid.shape[0], timesteps, X_valid.shape[1] / timesteps))
        y_validLabel = datasets[2][1]
        y_valid = np.zeros((X_valid.shape[0], np.max(y_validLabel) + 1))
        for rower in np.arange(X_valid.shape[0]):
            y_valid[rower, y_validLabel[rower, 0]] = 1

        X_test = datasets[2][0]
        X_test = np.reshape(X_test, (X_test.shape[0], timesteps, X_test.shape[1] / timesteps))
        y_testLabel = datasets[2][1]
        y_test = np.zeros((X_test.shape[0], np.max(y_testLabel) + 1))
        for rower in np.arange(X_test.shape[0]):
            y_test[rower, y_testLabel[rower, 0]] = 1

        model = Sequential()
        model.add(LSTM(input_shape=(timesteps, data_dim), output_dim=128, activation='sigmoid',
                       inner_activation='hard_sigmoid', name="LSTM Layer 1"))
        model.add(Dropout(0.5, name="Dropout Layer 1"))
        model.add(Dense(3, name="Dense Layer 2"))
        model.add(Activation('softmax', name="softmax Layer 2"))

        KerasClassifiers.showModel(model)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.fit(X_train, y_train,
                  batch_size=16,
                  nb_epoch=10,
                  validation_data=(X_valid, y_valid))

        score = model.evaluate(X_test, y_test, batch_size=16)
        print("The final score is {0}".format(score))
    elif thingToDo == 4:
        rootDataFolder = CreateUtils.getRootDataFolder()
        loadWeightsFilePath = os.path.join(CreateUtils.getExperimentFolder(
            'PatchShortTallAllFreq',
            'bikeneighborhoodSequenceOneFileNormCTDM',
            'LSTM',
            'RegressionAllClasses2LPlus2MLPStatefulKalmanTrainDQ2AutoBatchDropRlrPWeightRMSPropTD'),
            'bestLoss_modelWeights.h5')
        filePatherArg = loadWeightsFilePath
        with pd.HDFStore(filePatherArg, 'r') as datasetStore:
            print datasetStore.root
            print datasetStore.root.model_weights

        import h5py

        f = h5py.File(filePatherArg, mode='r')
        model_weights = f['model_weights']
        for (layer_name, layer) in model_weights.iteritems():
            print layer_name
            weight_names = [n.decode('utf8') for n in layer.attrs['weight_names']]
            if len(weight_names):
                for (weightName, weightValue) in layer.iteritems():
                    print weightName
                    arr = np.asarray(weightValue)
                    print arr
