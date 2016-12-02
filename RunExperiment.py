import os
import yaml
import shutil

import logistic_sgd
import mlp
import convolutional_mlp
import DBN
import linearRegression
import SkleanEnsembleRegressors
import KerasClassifiers
import CreateFeature
import CreateDataset
import CreateUtils

rawDataFolder = CreateUtils.getRawDataFolder()

featureSetNameMain = 'FFTWindowLowFreq'
datasetNameMain = ['bikeneighborhoodPackParticleNormParticleM']
classifierTypeMain = ['LSTM']
classifierSetNameMain = ['ClassificationAllClasses2LPlus2MLPStatefulAutoBatchDropReg2RlrRMSPropTD']
datasetNameStatsMain = 'bikeneighborhoodPackFileNormParticleM'  # for right now you only get one

# per run variables
forceRefreshFeatures = False
forceRefreshDataset = False
forceRefreshModel = False
forceRefreshStats = True

removeFeatureSetNames = []
removeDatasetNames = []
removeClassifierTypeNames = []
removeClassifierSetNames = []

# statstics parameters
useLabels = True
showFigures = True
whichSetMaster = 1
wildCards = (0, 0, 0, 0)
onlyPreviousExperiments = True  # only works if the last wild card is 1

# keras
showKerasFigure = False
modelStoreNameType = "best"


def getParameters(rawDataFolderArg, featureSetName, datasetName, classifierType, classifierSetName):
    featureDataFolder = os.path.join(rawDataFolderArg, "Processed Data Features", featureSetName)
    featureConfigFileName = os.path.join(featureDataFolder, "feature parameters.yaml")
    with open(featureConfigFileName, 'r') as myConfigFile:
        featureParameters = yaml.load(myConfigFile)

    processedDataFolder = os.path.join(rawDataFolderArg, "Processed Data Datasets", datasetName)
    datasetConfigFileName = os.path.join(processedDataFolder, "dataset parameters.yaml")
    with open(datasetConfigFileName, 'r') as myConfigFile:
        datasetParameters = yaml.load(myConfigFile)

    modelDataFolder = os.path.join(rawDataFolderArg, "Processed Data Models", classifierType, classifierSetName)
    modelConfigFileName = os.path.join(modelDataFolder, "model set parameters.yaml")
    with open(modelConfigFileName, 'r') as myConfigFile:
        modelParameters = yaml.load(myConfigFile)
    return featureParameters, datasetParameters, modelParameters


def runExperiment(featureSetName, datasetName, classifierType, classifierSetName, whichSet=1,
                  showFiguresArg=(False, False),
                  datasetNameStats=None,
                  modelStoreNameTypeArg="best"):
    print("Running Experiment on {featureSetName}, {datasetName}, {classifierType}, {classifierSetName}".format(
        featureSetName=featureSetName, datasetName=datasetName, classifierType=classifierType,
        classifierSetName=classifierSetName))
    # Load all the config files
    featureDataFolder = os.path.join(rawDataFolder, "Processed Data Features", featureSetName)
    featureConfigFileName = os.path.join(featureDataFolder, "feature parameters.yaml")
    with open(featureConfigFileName, 'r') as myConfigFile:
        featureParameters = yaml.load(myConfigFile)

    processedDataFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)
    datasetConfigFileName = os.path.join(processedDataFolder, "dataset parameters.yaml")
    with open(datasetConfigFileName, 'r') as myConfigFile:
        datasetParameters = yaml.load(myConfigFile)

    modelDataFolder = os.path.join(rawDataFolder, "Processed Data Models", classifierType, classifierSetName)
    modelConfigFileName = os.path.join(modelDataFolder, "model set parameters.yaml")
    with open(modelConfigFileName, 'r') as myConfigFile:
        modelParameters = yaml.load(myConfigFile)

    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureSetName, datasetName, classifierType,
                                     classifierSetName)
    if not os.path.exists(experimentsFolder):
        os.makedirs(experimentsFolder)

    datasetStatsConfigFileName = ''
    if datasetNameStats is not None and datasetNameStats != '':
        processedDataTestFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetNameStats)
        datasetStatsConfigFileName = os.path.join(processedDataTestFolder, "dataset parameters.yaml")
        with open(datasetStatsConfigFileName, 'r') as myConfigFile:
            datasetStatsParameters = yaml.load(myConfigFile)
        statisticsStoreFolder = os.path.join(experimentsFolder, datasetNameStats)
    else:
        datasetStatsParameters = datasetParameters
        datasetStatsConfigFileName = datasetConfigFileName
        statisticsStoreFolder = os.path.join(experimentsFolder, datasetName)
    if not os.path.exists(statisticsStoreFolder):
        os.makedirs(statisticsStoreFolder)

    if featureParameters['feature parameters']['featureMethod'] in CreateUtils.featureMethodNamesRebuildValid:
        removeFileNumbers = datasetParameters['removeFileNumbers'] if 'removeFileNumbers' in datasetParameters else {}
        onlyFileNumbers = datasetParameters['onlyFileNumbers'] if 'onlyFileNumbers' in datasetParameters else {}
        CreateFeature.buildFeatures(featureParameters,
                                    forceRefreshFeatures=forceRefreshFeatures,
                                    allBaseFileNames=datasetParameters['allBaseFileNames'],
                                    removeFileNumbers=removeFileNumbers,
                                    onlyFileNumbers=onlyFileNumbers)
        CreateDataset.buildDataSet(datasetParameters, featureParameters, forceRefreshDataset=forceRefreshDataset)

    # Copy over config files
    shutil.copyfile(featureConfigFileName, os.path.join(experimentsFolder, os.path.basename(featureConfigFileName)))
    shutil.copyfile(datasetConfigFileName, os.path.join(experimentsFolder, os.path.basename(datasetConfigFileName)))
    shutil.copyfile(modelConfigFileName, os.path.join(experimentsFolder, os.path.basename(modelConfigFileName)))
    shutil.copyfile(datasetStatsConfigFileName, os.path.join(statisticsStoreFolder, os.path.basename(datasetStatsConfigFileName)))

    # Start Experiment
    if modelParameters['classifierType'] == 'LogisticRegression':
        logistic_sgd.sgd_optimization_parameterized(featureParameters, datasetParameters, modelParameters,
                                                    forceRebuildModel=forceRefreshModel)
    elif modelParameters['classifierType'] == 'MLP':
        mlp.mlp_parameterized(featureParameters, datasetParameters, modelParameters,
                              forceRebuildModel=forceRefreshModel)
    elif modelParameters['classifierType'] == 'ConvolutionalMLP':
        convolutional_mlp.convolutional_mlp_parameterized(featureParameters, datasetParameters, modelParameters,
                                                          forceRebuildModel=forceRefreshModel)
    elif modelParameters['classifierType'] == 'DBN':
        DBN.dbn_parameterized(featureParameters, datasetParameters, modelParameters,
                              forceRebuildModel=forceRefreshModel)
    elif modelParameters['classifierType'] == 'LinearRegression':
        linearRegression.sgd_optimization_parameterized(featureParameters, datasetParameters, modelParameters,
                                                        forceRebuildModel=forceRefreshModel)
    elif modelParameters['classifierType'] in CreateUtils.sklearnensembleTypes:
        if modelParameters['classifierGoal'] == 'regression':
            SkleanEnsembleRegressors.skleanensemble_parameterized(featureParameters, datasetParameters, modelParameters,
                                                                  forceRebuildModel=forceRefreshModel)
        else:
            raise NotImplementedError()
    elif modelParameters['classifierType'] in CreateUtils.kerasTypes:
        KerasClassifiers.kerasClassifier_parameterized(featureParameters, datasetParameters, modelParameters,
                                                       forceRebuildModel=forceRefreshModel,
                                                       showModelAsFigure=showFiguresArg[1])

    if not os.path.exists(os.path.join(experimentsFolder, 'results.yaml')) or forceRefreshStats:
        if modelParameters['classifierType'] in CreateUtils.sklearnensembleTypes:
            SkleanEnsembleRegressors.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                            datasetStatsParameters, modelParameters, whichSet=whichSet,
                                                            showFigures=showFiguresArg[0],
                                                            useLabels=useLabels)
        elif modelParameters['classifierType'] in CreateUtils.kerasTypes:
            KerasClassifiers.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                    datasetStatsParameters, modelParameters, whichSet=whichSet,
                                                    showFigures=showFiguresArg[0],
                                                    useLabels=useLabels, modelStoreNameType=modelStoreNameTypeArg)
        else:
            classifierGoal = modelParameters['classifierGoal'] if 'classifierGoal' in modelParameters else None
            if classifierGoal == 'classification' or classifierGoal is None:
                logistic_sgd.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                    datasetStatsParameters, modelParameters, whichSet=whichSet,
                                                    showFigures=showFiguresArg[0], useLabels=useLabels)
            elif classifierGoal == 'regression':
                linearRegression.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                        datasetStatsParameters, modelParameters, whichSet=whichSet,
                                                        showFigures=showFiguresArg[0], useLabels=useLabels)
            else:
                raise ValueError("This classifier goal {0} not supported".format(classifierGoal))


if __name__ == '__main__':
    featureDataFolderMain = os.path.join(rawDataFolder, "Processed Data Features")
    if wildCards[0]:
        thisFeatureFolder = os.path.join(rawDataFolder,
                                         "Data Experiments") if onlyPreviousExperiments else featureDataFolderMain
        featureSets = [fileIterator for fileIterator in os.listdir(thisFeatureFolder) if os.path.isdir(
            os.path.join(thisFeatureFolder, fileIterator)) and fileIterator not in removeFeatureSetNames]
    else:
        featureSets = featureSetNameMain if isinstance(featureSetNameMain, list) else [featureSetNameMain]
    for featureSet in featureSets:
        processedDataFolderMain = os.path.join(rawDataFolder, "Processed Data Datasets")
        if wildCards[1]:
            thisDatasetFolder = os.path.join(rawDataFolder, "Data Experiments",
                                             featureSet) if onlyPreviousExperiments else processedDataFolderMain
            datasets = [fileIterator for fileIterator in os.listdir(thisDatasetFolder) if os.path.isdir(
                os.path.join(thisDatasetFolder, fileIterator)) and fileIterator not in removeDatasetNames]
        else:
            datasets = datasetNameMain if isinstance(datasetNameMain, list) else [datasetNameMain]
        for dataset in datasets:
            modelDataFolderMain = os.path.join(rawDataFolder, "Processed Data Models")
            if wildCards[2]:
                thisClassifierTypeFolder = os.path.join(rawDataFolder, "Data Experiments", featureSet,
                                                        dataset) if onlyPreviousExperiments else modelDataFolderMain
                classifierTypes = [fileIterator for fileIterator in os.listdir(thisClassifierTypeFolder) if
                                   os.path.isdir(
                                       os.path.join(thisClassifierTypeFolder,
                                                    fileIterator)) and fileIterator not in removeClassifierTypeNames]
            else:
                classifierTypes = classifierTypeMain if isinstance(classifierTypeMain, list) else [classifierTypeMain]
            for classifierTypeIterator in classifierTypes:
                modelTypeDataFolderMain = os.path.join(rawDataFolder, "Processed Data Models", classifierTypeIterator)
                if wildCards[3]:
                    thisClassifierSetFolder = os.path.join(rawDataFolder, "Data Experiments", featureSet, dataset,
                                                           classifierTypeIterator) \
                        if onlyPreviousExperiments else modelDataFolderMain
                    classifierSets = [fileIterator for fileIterator in os.listdir(thisClassifierSetFolder) if
                                      os.path.isdir(
                                          os.path.join(thisClassifierSetFolder,
                                                       fileIterator)) and fileIterator not in removeClassifierSetNames]
                else:
                    classifierSets = classifierSetNameMain if isinstance(classifierSetNameMain, list) else [
                        classifierSetNameMain]
                for classifierSet in classifierSets:
                    runExperiment(featureSet, dataset, classifierTypeIterator, classifierSet, whichSet=whichSetMaster,
                                  showFiguresArg=(showFigures, showKerasFigure), datasetNameStats=datasetNameStatsMain,
                                  modelStoreNameTypeArg=modelStoreNameType)
