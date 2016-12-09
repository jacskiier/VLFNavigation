import sklearn.ensemble
from sklearn import gaussian_process
import os
import RegressionUtils
import cPickle

import matplotlib.pylab as plt
import numpy as np


def makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters, datasetParameters, classifierParameters, valueMethod=0,
                           useLabels=True, whichSetName='valid', showFigures=True):
    """
    Make staticstics for a model using the features, datset, and classifier given whose model is already made

    :type experimentsFolder: str
    :param experimentsFolder: Location of the pre made model and where the statistics will be saved

    :type statisticsStoreFolder: str
    :param statisticsStoreFolder: Location of the folder to store the statistics being generated

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type valueMethod: int
    :param valueMethod: Values type to be for classification and thresholding, 0 = use highest probability, 1 = use ration of higest prob to second, 2 = use difference of highest prob to second

    :type useLabels: bool
    :param useLabels: If labels should be used in charts True or just the class number False

    :type whichSet: int
    :param whichSet: Which of the sets to do the statistics on training=0 validation=1 testing=2

    :type showFigures: bool
    :param showFigures: If you want to see the figures now instead of viewing them on disk later (still saves them no matter what)
    """

    setNames = ['train', 'valid', 'test']
    whichSet = setNames.index(whichSetName)

    if os.path.exists(os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    rogueClassesMaster = classifierParameters['rogueClasses']

    # get predicted values
    print("Getting predicted values for {0}".format(classifierParameters['classifierType']))
    datasets, inputs, outputs, max_batch_size = RegressionUtils.load_data(datasetFile, rogueClasses=(), makeSharedData=False)
    modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'best_model.pkl')
    print ("opening model {0}".format(modelStoreFilePathFullTemp))
    rfcs = cPickle.load(open(modelStoreFilePathFullTemp))
    X_test = datasets[whichSet][0]

    appendY = classifierParameters['appendY'] if 'appendY' in classifierParameters else False
    y_test = datasets[whichSet][1]

    # sigma = 100
    # pingpongs = 1
    # if appendY:
    #     pingpongs = 9
    # y_est = np.zeros(y_test.shape)
    # for currentDim in range(y_test.ndim):
    #     yarray = np.expand_dims(y_test[:, currentDim],1) + np.random.randn(y_test.shape[0], 1) * sigma
    #     #yarray = np.ones((y_test.shape[0],1)) * np.mean(y_test[:,currentDim])
    #     y_est[:,currentDim] = yarray.squeeze()
    #
    # for pingpong in range(pingpongs):
    #     y_preds = []
    #     for rfc,currentDim in zip(rfcs, range(len(rfcs))):
    #         X_testtemp = X_test
    #         if appendY:
    #             yset = list(set(range(y_test.ndim)) - {currentDim})
    #             yappendArray = y_est[:,yset]
    #             X_testtemp = np.hstack((X_test, yappendArray))
    #         y_predtemp = rfc.predict(X_testtemp)
    #         y_preds.append(y_predtemp)
    #     y_est2 = np.transpose(np.vstack(tuple(y_preds)))
    #     y_est = y_est2
    # y_pred = y_est

    y_preds = []
    numRfcs = len(rfcs) if isinstance(rfcs, tuple) else 1
    for rfc, currentDim in zip(rfcs, range(numRfcs)):
        y_predtemp = rfc.predict(X_test)
        y_preds.append(y_predtemp)
    # y_pred = np.transpose(np.vstack(tuple(y_preds)))
    y_pred = np.vstack(tuple(y_preds))

    if classifierParameters['classifierGoal'] == 'regression':
        RegressionUtils.getStatistics(predictedValues=y_pred,
                                      trueValues=y_test,
                                      setName=setNames[whichSet],
                                      statisticsStoreFolder=statisticsStoreFolder,
                                      datasetParameters=datasetParameters)
    else:
        raise ValueError("Classification isn't working yet for Sklearn types")
    if showFigures:
        plt.show()


def skleanensemble_parameterized(featureParameters, datasetParameters, classifierParameters, forceRebuildModel=False):
    """
    Train a Logistic Regression model using the features, datset, and classifier parameters given

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type forceRebuildModel: bool
    :param forceRebuildModel: forces to rebuild the model and train it again
    """

    rawDataFolder = datasetParameters['rawDataFolder']

    if os.path.exists(os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')):
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'] + '.hf')
    else:
        datasetFile = os.path.join(datasetParameters['processedDataFolder'], featureParameters['featureSetName'],
                                   datasetParameters['datasetName'] + '.pkl.gz')

    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureParameters['featureSetName'], datasetParameters['datasetName'],
                                     classifierParameters['classifierType'], classifierParameters['classifierSetName'])

    bestModelFilePath = os.path.join(experimentsFolder, 'best_model.pkl')
    if not os.path.exists(bestModelFilePath) or forceRebuildModel:
        datasets, inputs, outputs, max_batch_size = RegressionUtils.load_data(datasetFile, rogueClasses=(), makeSharedData=False)

        X_train = datasets[0][0]
        X_train[np.isnan(X_train)] = 0
        if classifierParameters['classifierType'] == 'RandomForest':
            max_features = classifierParameters['max_features'] if 'max_features' in classifierParameters else 'auto'
            max_depth = classifierParameters['max_depth'] if 'max_depth' in classifierParameters else None
            classifier = sklearn.ensemble.RandomForestRegressor(n_estimators=classifierParameters['treeNumber'],
                                                                random_state=classifierParameters['rngSeed'], verbose=True, n_jobs=4,
                                                                max_features=max_features, max_depth=max_depth)
            y_train = datasets[0][1]
            print("Fitting training data with classifier type {0}".format(classifierParameters['classifierType']))
            classifier.fit(X_train, y_train)

            # save the fitted model as a pickle
            modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'best_model.pkl')
            with open(modelStoreFilePathFullTemp, 'wb') as f:
                cPickle.dump(classifier, f)
        elif classifierParameters['classifierType'] == 'ADABoost':
            classifiers = []
            appendY = classifierParameters['appendY'] if 'appendY' in classifierParameters else False
            max_depth = classifierParameters['max_depth'] if 'max_depth' in classifierParameters else 20
            for currentDim in range(datasets[0][1].ndim):
                classifiertemp = sklearn.ensemble.AdaBoostRegressor(sklearn.tree.DecisionTreeRegressor(max_depth=max_depth),
                                                                    n_estimators=classifierParameters['estimators'],
                                                                    random_state=classifierParameters['rngSeed'])
                X_traintemp = X_train
                y_traintemp = datasets[0][1][:, currentDim]

                if appendY:
                    yset = list(set(range(datasets[0][1].ndim)) - {currentDim})
                    yarray = datasets[0][1][:, yset]
                    X_traintemp = np.hstack((X_train, yarray))

                print("Fitting training data with classifier type {0}".format(classifierParameters['classifierType']))
                classifiertemp.fit(X_traintemp, y_traintemp)
                classifiers.append(classifiertemp)

            # save the fitted model as a pickle
            modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'best_model.pkl')
            with open(modelStoreFilePathFullTemp, 'wb') as f:
                cPickle.dump(tuple(classifiers), f)
        elif classifierParameters['classifierType'] == 'GradientBoosting':
            classifiers = []
            for currentDim in datasets[0][1].ndim:
                classifiertemp = sklearn.ensemble.GradientBoostingRegressor(max_depth=2,
                                                                            n_estimators=classifierParameters['estimators'],
                                                                            random_state=classifierParameters['rngSeed'])
                y_traintemp = datasets[0][1][:, currentDim]

                print("Fitting training data with classifier type {0}".format(classifierParameters['classifierType']))
                classifiertemp.fit(X_train, y_traintemp)
                classifiers.append(classifiertemp)

            # save the fitted model as a pickle
            modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'best_model.pkl')
            with open(modelStoreFilePathFullTemp, 'wb') as f:
                cPickle.dump(tuple(classifiers), f)
        elif classifierParameters['classifierType'] == 'GaussianProcess':
            classifiers = []
            for currentDim in datasets[0][1].ndim:
                classifiertemp = gaussian_process.GaussianProcess(random_state=classifierParameters['rngSeed'])
                y_traintemp = datasets[0][1][:, currentDim]

                print("Fitting training data with classifier type {0}".format(classifierParameters['classifierType']))
                classifiertemp.fit(X_train, y_traintemp)
                classifiers.append(classifiertemp)

            # save the fitted model as a pickle
            modelStoreFilePathFullTemp = os.path.join(experimentsFolder, 'best_model.pkl')
            with open(modelStoreFilePathFullTemp, 'wb') as f:
                cPickle.dump(tuple(classifiers), f)
        else:
            raise ValueError()
