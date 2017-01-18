import os
import numpy as np
import datetime
import pytz
import logistic_sgd
import mlp
import convolutional_mlp
import DBN
import linearRegression
import SkleanEnsembleRegressors
import KerasClassifiers
import CreateFeature
import CreateDataset
import CreateClassifierConfig
import CreateUtils


def runExperiment(featureSetName,
                  datasetName,
                  classifierType,
                  classifierSetName,
                  whichSetName='valid',
                  whichSetNameStat='valid',
                  showFiguresArg=(False, False),
                  datasetNameStats=None,
                  modelStoreNameTypeArg="best",
                  trainValidTestSetNames=('train', 'valid', 'test'),
                  forceRefreshFeatures=False,
                  forceRefreshDataset=False,
                  forceRefreshModel=False,
                  forceRefreshStats=False,
                  useLabels=True,
                  randomName=None,
                  randomModelSeedNumber=None):
    # statistics name
    if datasetNameStats is None or datasetNameStats == '':
        datasetNameStats = datasetName

    print("Running Experiment on {featureSetName}, {datasetName}, {classifierType}, {classifierSetName}".format(
        featureSetName=featureSetName,
        datasetName=datasetName,
        classifierType=classifierType,
        classifierSetName=classifierSetName))
    print("Creating Stats on dataset {datasetNameStats}".format(datasetNameStats=datasetNameStats))

    # get location of this experiment folder
    if randomName is None:
        experimentsFolder = CreateUtils.getExperimentFolder(featureSetName, datasetName, classifierType, classifierSetName)
    else:
        if randomModelSeedNumber is None:
            np.random.seed()
            randomModelSeedNumber = np.random.randint(0, 4294967295)
        print("Random name: {randomName} with seed: {seed}".format(randomName=randomName, seed=randomModelSeedNumber))
        experimentsFolder = CreateUtils.getRandomExperimentFolder(str(randomModelSeedNumber),
                                                                  featureSetName,
                                                                  datasetName,
                                                                  classifierType,
                                                                  classifierSetName,
                                                                  randomName=randomName)
    # create the folders if required
    copyExperimentParameters = forceRefreshModel
    if not os.path.exists(experimentsFolder):
        copyExperimentParameters = True
        os.makedirs(experimentsFolder)
    if not os.path.exists(CreateUtils.getDatasetConfigFileName(baseFolder=experimentsFolder)):
        copyExperimentParameters = True
    statisticsStoreFolder = CreateUtils.getStatisticsFolder(experimentsFolder, datasetNameStats, whichSetNameStat)
    copyStatisticsParameters = forceRefreshStats
    if not os.path.exists(statisticsStoreFolder):
        os.makedirs(statisticsStoreFolder)
        copyStatisticsParameters = True

    # Copy over config files if not there or if we are refreshing
    if copyExperimentParameters:
        CreateUtils.copyConfigsToExperimentsFolder(experimentsFolder=experimentsFolder,
                                                   featureSetName=featureSetName,
                                                   datasetName=datasetName,
                                                   classifierType=classifierType,
                                                   classifierSetName=classifierSetName)
    if copyStatisticsParameters:
        CreateUtils.copyConfigsToExperimentsFolder(experimentsFolder=statisticsStoreFolder,
                                                   datasetName=datasetNameStats)

    # Load all the config files
    (featureParameters,
     datasetParameters,
     classifierParameters) = CreateUtils.getParameters(featureSetName=featureSetName,
                                                       datasetName=datasetName,
                                                       classifierType=classifierType,
                                                       classifierSetName=classifierSetName,
                                                       baseFolder=experimentsFolder)
    datasetStatsParameters = CreateUtils.getParameters(datasetName=datasetNameStats,
                                                       baseFolder=statisticsStoreFolder)

    # if doing random then
    if randomName is not None:
        randomConfigFileName = CreateUtils.getRandomConfigFileName(featureSetName,
                                                                   datasetName,
                                                                   classifierType,
                                                                   classifierSetName,
                                                                   randomName=randomName)
        randRangeDict = CreateUtils.loadConfigFile(randomConfigFileName)
        classifierParameters = CreateClassifierConfig.randomizeParameters(randomModelSeedNumber,
                                                                          classifierParameters,
                                                                          randRangeDict)
        configFileName = CreateUtils.getModelConfigFileName(baseFolder=experimentsFolder)
        CreateUtils.makeConfigFile(configFileName, classifierParameters)

    # if required or asked for build up the features and dataset
    if featureParameters['feature parameters']['featureMethod'] in CreateUtils.featureMethodNamesRebuildValid:
        removeFileNumbers = datasetParameters['removeFileNumbers'] if 'removeFileNumbers' in datasetParameters else {}
        onlyFileNumbers = datasetParameters['onlyFileNumbers'] if 'onlyFileNumbers' in datasetParameters else {}
        CreateFeature.buildFeatures(featureParameters,
                                    forceRefreshFeatures=forceRefreshFeatures,
                                    allBaseFileNames=datasetParameters['allBaseFileNames'],
                                    removeFileNumbers=removeFileNumbers,
                                    onlyFileNumbers=onlyFileNumbers)
        CreateDataset.buildDataSet(datasetParameters,
                                   featureParameters,
                                   forceRefreshDataset=forceRefreshDataset)

    # Start Experiment
    didFinishExperiment = True
    if classifierParameters['classifierType'] == 'LogisticRegression':
        logistic_sgd.sgd_optimization_parameterized(featureParameters,
                                                    datasetParameters,
                                                    classifierParameters,
                                                    forceRebuildModel=forceRefreshModel)
    elif classifierParameters['classifierType'] == 'MLP':
        mlp.mlp_parameterized(featureParameters,
                              datasetParameters,
                              classifierParameters,
                              forceRebuildModel=forceRefreshModel)
    elif classifierParameters['classifierType'] == 'ConvolutionalMLP':
        convolutional_mlp.convolutional_mlp_parameterized(featureParameters,
                                                          datasetParameters,
                                                          classifierParameters,
                                                          forceRebuildModel=forceRefreshModel)
    elif classifierParameters['classifierType'] == 'DBN':
        DBN.dbn_parameterized(featureParameters,
                              datasetParameters,
                              classifierParameters,
                              forceRebuildModel=forceRefreshModel)
    elif classifierParameters['classifierType'] == 'LinearRegression':
        linearRegression.sgd_optimization_parameterized(featureParameters,
                                                        datasetParameters,
                                                        classifierParameters,
                                                        forceRebuildModel=forceRefreshModel)
    elif classifierParameters['classifierType'] in CreateUtils.sklearnensembleTypes:
        if classifierParameters['classifierGoal'] == 'regression':
            SkleanEnsembleRegressors.skleanensemble_parameterized(featureParameters,
                                                                  datasetParameters,
                                                                  classifierParameters,
                                                                  forceRebuildModel=forceRefreshModel)
        else:
            raise NotImplementedError()
    elif classifierParameters['classifierType'] in CreateUtils.kerasTypes:
        didFinishExperiment = KerasClassifiers.kerasClassifier_parameterized(featureParameters,
                                                                             datasetParameters,
                                                                             classifierParameters,
                                                                             forceRebuildModel=forceRefreshModel,
                                                                             showModelAsFigure=showFiguresArg[1],
                                                                             trainValidTestSetNames=trainValidTestSetNames,
                                                                             experimentsFolder=experimentsFolder)

    # Make the statistics on the experiment
    if not os.path.exists(os.path.join(experimentsFolder, 'results.yaml')) or forceRefreshStats:
        if classifierParameters['classifierType'] in CreateUtils.sklearnensembleTypes:
            SkleanEnsembleRegressors.makeStatisticsForModel(experimentsFolder,
                                                            statisticsStoreFolder,
                                                            featureParameters,
                                                            datasetStatsParameters,
                                                            classifierParameters,
                                                            whichSetName=whichSetName,
                                                            showFigures=showFiguresArg[0],
                                                            useLabels=useLabels)
        elif classifierParameters['classifierType'] in CreateUtils.kerasTypes:
            KerasClassifiers.makeStatisticsForModel(experimentsFolder,
                                                    statisticsStoreFolder,
                                                    featureParameters,
                                                    datasetParameters,
                                                    classifierParameters,
                                                    whichSetName=whichSetName,
                                                    whichSetNameStat=whichSetNameStat,
                                                    showFigures=showFiguresArg[0],
                                                    useLabels=useLabels,
                                                    modelStoreNameType=modelStoreNameTypeArg)
        else:
            classifierGoal = classifierParameters['classifierGoal'] if 'classifierGoal' in classifierParameters else None
            if classifierGoal == 'classification' or classifierGoal is None:
                logistic_sgd.makeStatisticsForModel(experimentsFolder,
                                                    statisticsStoreFolder,
                                                    featureParameters,
                                                    datasetParameters,
                                                    classifierParameters,
                                                    whichSetName=whichSetName,
                                                    whichSetNameStat=whichSetNameStat,
                                                    showFigures=showFiguresArg[0],
                                                    useLabels=useLabels)
            elif classifierGoal == 'regression':
                linearRegression.makeStatisticsForModel(experimentsFolder,
                                                        statisticsStoreFolder,
                                                        featureParameters,
                                                        datasetStatsParameters,
                                                        classifierParameters,
                                                        whichSetName=whichSetNameStat,
                                                        showFigures=showFiguresArg[0],
                                                        useLabels=useLabels)
            else:
                raise ValueError("This classifier goal {0} not supported".format(classifierGoal))
    return didFinishExperiment


def runMain():
    # run experiment parameters
    featureMethod = "SignalPlaceholder"
    featureSetNameMain = 'FFTWindowLowFreq'
    datasetNameMain = ['bikeneighborhoodPackClassNormParticle']
    classifierTypeMain = ['LSTM']
    classifierSetNameMain = ['ClassificationAllClasses2LPlus2MLPStatefulAutoBatchDropReg2RlrRMSPropTDTF']
    datasetNameStats = 'bikeneighborhoodPackFileNormParticle'  # for right now you only get one
    trainValidTestSetNames = ('train', None, None)

    # per run variables
    forceRefreshFeatures = False
    forceRefreshDataset = False
    forceRefreshModel = True
    forceRefreshStats = True

    # statstics parameters
    useLabels = True
    showFigures = True
    whichSetNameModel = 'valid'
    whichSetNameStats = 'valid'

    # keras specific parameters
    showKerasFigure = False
    modelStoreNameType = "best"

    # this line sets the root folder for future calls
    CreateUtils.getRootDataFolder(featureMethod=featureMethod)

    # wildcard parameters
    removeFeatureSetNames = []
    removeDatasetNames = []
    removeClassifierTypeNames = []
    removeClassifierSetNames = []
    wildCards = (0, 0, 0, 0)
    onlyPreviousExperiments = True  # only works if the last wild card is 1

    # random Parameters
    randomName = None
    randomModelSeedNumber = None

    # start wildcard loops
    featureDataFolderMain = CreateUtils.getProcessedFeaturesFolder()
    if wildCards[0]:
        thisFeatureFolder = CreateUtils.getExperimentFolder() if onlyPreviousExperiments else featureDataFolderMain
        featureSetNamess = [fileIterator for fileIterator in os.listdir(thisFeatureFolder) if os.path.isdir(
            os.path.join(thisFeatureFolder, fileIterator)) and fileIterator not in removeFeatureSetNames]
    else:
        featureSetNamess = featureSetNameMain if isinstance(featureSetNameMain, list) else [featureSetNameMain]
    for featureSetName in featureSetNamess:
        processedDataFolderMain = CreateUtils.getProcessedDataDatasetsFolder()
        if wildCards[1]:
            thisDatasetFolder = CreateUtils.getExperimentFolder(featureSetName=featureSetName) if onlyPreviousExperiments else processedDataFolderMain
            datasets = [fileIterator for fileIterator in os.listdir(thisDatasetFolder) if os.path.isdir(
                os.path.join(thisDatasetFolder, fileIterator)) and fileIterator not in removeDatasetNames]
        else:
            datasets = datasetNameMain if isinstance(datasetNameMain, list) else [datasetNameMain]
        for datasetName in datasets:
            modelDataFolderMain = CreateUtils.getModelFolder()
            if wildCards[2]:
                thisClassifierTypeFolder = CreateUtils.getExperimentFolder(featureSetName=featureSetName, datasetName=datasetName) \
                    if onlyPreviousExperiments else modelDataFolderMain
                classifierTypes = [fileIterator for fileIterator in os.listdir(thisClassifierTypeFolder) if
                                   os.path.isdir(
                                       os.path.join(thisClassifierTypeFolder,
                                                    fileIterator)) and fileIterator not in removeClassifierTypeNames]
            else:
                classifierTypes = classifierTypeMain if isinstance(classifierTypeMain, list) else [classifierTypeMain]
            for classifierType in classifierTypes:
                if wildCards[3]:
                    onlyPreviousExperimentsFolder = CreateUtils.getExperimentFolder(featureSetName=featureSetName,
                                                                                    datasetName=datasetName,
                                                                                    classifierType=classifierType)
                    thisClassifierSetFolder = onlyPreviousExperimentsFolder if onlyPreviousExperiments else modelDataFolderMain
                    classifierSetNames = [fileIterator for fileIterator in os.listdir(thisClassifierSetFolder) if
                                          os.path.isdir(os.path.join(thisClassifierSetFolder, fileIterator)) and
                                          fileIterator not in removeClassifierSetNames]
                else:
                    classifierSetNames = classifierSetNameMain if isinstance(classifierSetNameMain, list) else [classifierSetNameMain]
                for classifierSetName in classifierSetNames:
                    runExperiment(featureSetName,
                                  datasetName,
                                  classifierType,
                                  classifierSetName,
                                  whichSetName=whichSetNameModel,
                                  whichSetNameStat=whichSetNameStats,
                                  showFiguresArg=(showFigures, showKerasFigure),
                                  datasetNameStats=datasetNameStats,
                                  modelStoreNameTypeArg=modelStoreNameType,
                                  trainValidTestSetNames=trainValidTestSetNames,
                                  forceRefreshFeatures=forceRefreshFeatures,
                                  forceRefreshDataset=forceRefreshDataset,
                                  forceRefreshModel=forceRefreshModel,
                                  forceRefreshStats=forceRefreshStats,
                                  useLabels=useLabels,
                                  randomName=randomName,
                                  randomModelSeedNumber=randomModelSeedNumber)


def runRandom():
    # run experiment parameters
    featureMethod = "SignalPlaceholder"
    featureSetName = 'FFTWindowLowFreq'
    datasetName = 'bikeneighborhoodPackClassNormParticle'
    classifierType = 'LSTM'
    classifierSetName = 'ClassificationAllClasses2LPlus2MLPStatefulAutoBatchDropReg2RMSPropTD'
    datasetNameStats = 'bikeneighborhoodPackFileNormParticle'  # for right now you only get one
    trainValidTestSetNames = ('train', 'valid', None)

    # per run variables
    forceRefreshFeatures = False
    forceRefreshDataset = False
    forceRefreshModel = False
    forceRefreshStats = False

    # statstics parameters
    useLabels = True
    showFigures = False
    whichSetNameModel = 'valid'
    whichSetNameStats = 'valid'

    # keras specific parameters
    showKerasFigure = False
    modelStoreNameType = "best"

    # random parameters
    randomName = 'default'
    randomModelSeedNumber = None
    dropoutMin = 0.1
    dropoutMax = 0.9
    l1Min = 1e-5
    l1Max = 1e-2
    l2Min = 1e-5
    l2Max = 1e-2
    randRangeDict = {
        # LSTM
        'lstm_layers_sizes': {'type': 'linDlinDDec', 'listStart': 1, 'listStop': 5, 'start': 50, 'stop': 1001},
        'dropout_W': {'type': 'lin', 'start': dropoutMin, 'stop': dropoutMax},
        'dropout_U': {'type': 'lin', 'start': dropoutMin, 'stop': dropoutMax},
        # 'dropout_LSTM': {'type': 'log', 'start': dropoutMin, 'stop': dropoutMax},
        'W_regularizer_l1_LSTM': {'type': 'log', 'start': l1Min, 'stop': l1Max},
        'U_regularizer_l1_LSTM': {'type': 'log', 'start': l1Min, 'stop': l1Max},
        'b_regularizer_l1_LSTM': {'type': 'log', 'start': l1Min, 'stop': l1Max},
        'W_regularizer_l2_LSTM': {'type': 'log', 'start': l2Min, 'stop': l2Max},
        'U_regularizer_l2_LSTM': {'type': 'log', 'start': l2Min, 'stop': l2Max},
        'b_regularizer_l2_LSTM': {'type': 'log', 'start': l2Min, 'stop': l2Max},
        # MLP
        'hidden_layers_sizes': {'type': 'linDlinDDec', 'listStart': 1, 'listStop': 5, 'start': 50, 'stop': 1001},
        'dropout_hidden': {'type': 'lin', 'start': dropoutMin, 'stop': dropoutMax},
        'W_regularizer_l1_hidden': {'type': 'log', 'start': l1Min, 'stop': l1Max},
        'b_regularizer_l1_hidden': {'type': 'log', 'start': l1Min, 'stop': l1Max},
        'W_regularizer_l2_hidden': {'type': 'log', 'start': l2Min, 'stop': l2Max},
        'b_regularizer_l2_hidden': {'type': 'log', 'start': l2Min, 'stop': l2Max},
        # Training
        'learning_rate': {'type': 'log', 'start': 1e-5, 'stop': 1e-1},
        'optimizerType': {'type': 'list', 'values': ['rmsprop', 'adam']},
        # rmsprop
        'rho': {'type': 'log', 'start': 0.8, 'stop': 0.999999},
        # adam
        'beta_1': {'type': 'log', 'start': 0.8, 'stop': 0.999999},
        'beta_2': {'type': 'log', 'start': 0.8, 'stop': 0.999999},
    }

    # this line sets the root folder for future calls
    CreateUtils.getRootDataFolder(featureMethod=featureMethod)
    # get the base random folder and create it
    randomExperimentsBaseFolder = CreateUtils.getRandomExperimentBaseFolder(featureSetName,
                                                                            datasetName,
                                                                            classifierType,
                                                                            classifierSetName,
                                                                            randomName=randomName)
    if not os.path.exists(randomExperimentsBaseFolder):
        os.makedirs(randomExperimentsBaseFolder)

    # write the random parameters
    randomConfigFileName = CreateUtils.getRandomConfigFileName(featureSetName,
                                                               datasetName,
                                                               classifierType,
                                                               classifierSetName,
                                                               randomName=randomName)
    CreateUtils.makeConfigFile(randomConfigFileName, randRangeDict)

    tzinfo = pytz.timezone('US/Eastern')
    timeoutDate = datetime.datetime(year=2017, month=1, day=2, hour=18, minute=0, second=0, microsecond=0, tzinfo=tzinfo)
    didFinish = True
    while timeoutDate > datetime.datetime.now(tz=tzinfo) and didFinish:
        didFinish = runExperiment(featureSetName,
                                  datasetName,
                                  classifierType,
                                  classifierSetName,
                                  datasetNameStats=datasetNameStats,
                                  whichSetName=whichSetNameModel,
                                  whichSetNameStat=whichSetNameStats,
                                  showFiguresArg=(showFigures, showKerasFigure),
                                  modelStoreNameTypeArg=modelStoreNameType,
                                  trainValidTestSetNames=trainValidTestSetNames,
                                  forceRefreshFeatures=forceRefreshFeatures,
                                  forceRefreshDataset=forceRefreshDataset,
                                  forceRefreshModel=forceRefreshModel,
                                  forceRefreshStats=forceRefreshStats,
                                  useLabels=useLabels,
                                  randomName=randomName,
                                  randomModelSeedNumber=randomModelSeedNumber)


if __name__ == '__main__':
    runMain()
    # runRandom()
