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

featureMethod = "FFTWindow"
featureSetNameMain = 'FFTWindowDefault'
datasetNameMain = ['bikeneighborhoodPackFileNormParticle']
classifierTypeMain = ['LSTM']
classifierSetNameMain = ['ClassificationAllClasses2LPlus2MLPStatefulAutoBatchDropReg2RlrRMSPropTD']
datasetNameStatsMain = ''  # for right now you only get one

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
whichSetNameMaster = 'valid'
whichSetNameStats = 'train'
wildCards = (0, 0, 0, 0)
onlyPreviousExperiments = True  # only works if the last wild card is 1

# keras
showKerasFigure = False
modelStoreNameType = "best"

# this line sets the root folder for future calls
rootDataFolder = CreateUtils.getRootDataFolder(featureMethod=featureMethod)


def runExperiment(featureSetName,
                  datasetName,
                  classifierType,
                  classifierSetName,
                  whichSetName='valid',
                  whichSetNameStat='valid',
                  showFiguresArg=(False, False),
                  datasetNameStats=None,
                  modelStoreNameTypeArg="best"):
    print("Running Experiment on {featureSetName}, {datasetName}, {classifierType}, {classifierSetName}".format(
        featureSetName=featureSetName, datasetName=datasetName, classifierType=classifierType,
        classifierSetName=classifierSetName))
    # Load all the config files
    (featureParameters,
     datasetParameters,
     modelParameters) = CreateUtils.getParameters(featureSetName=featureSetName, datasetName=datasetName,
                                                  classifierType=classifierType,
                                                  classifierSetName=classifierSetNameMain)
    experimentsFolder = CreateUtils.getExperimentFolder(featureSetName, datasetName, classifierType, classifierSetName)
    if not os.path.exists(experimentsFolder):
        os.makedirs(experimentsFolder)

    # statistics name
    if datasetNameStats is None or datasetNameStats == '':
        datasetNameStats = datasetName
    # create the folder
    statisticsStoreFolder = CreateUtils.getStatisticsFolder(experimentsFolder, datasetNameStats, whichSetNameStat)
    if not os.path.exists(statisticsStoreFolder):
        os.makedirs(statisticsStoreFolder)
    # get the parameters of the stats
    datasetStatsParameters = CreateUtils.getParameters(datasetName=datasetNameStats)

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
    CreateUtils.copyConfigsToExperimentsFolder(experimentsFolder=experimentsFolder,
                                               featureSetName=featureSetName,
                                               datasetName=datasetName,
                                               classifierType=classifierType,
                                               classifierSetName=classifierSetName)
    CreateUtils.copyConfigsToExperimentsFolder(experimentsFolder=statisticsStoreFolder, datasetName=datasetNameStats)

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
                                                            datasetStatsParameters, modelParameters, whichSetName=whichSetName,
                                                            showFigures=showFiguresArg[0],
                                                            useLabels=useLabels)
        elif modelParameters['classifierType'] in CreateUtils.kerasTypes:
            KerasClassifiers.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                    datasetParameters, modelParameters, whichSetName=whichSetName,
                                                    whichSetNameStat=whichSetNameStat,
                                                    showFigures=showFiguresArg[0],
                                                    useLabels=useLabels, modelStoreNameType=modelStoreNameTypeArg)
        else:
            classifierGoal = modelParameters['classifierGoal'] if 'classifierGoal' in modelParameters else None
            if classifierGoal == 'classification' or classifierGoal is None:
                logistic_sgd.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                    datasetParameters, modelParameters, whichSetName=whichSetName,
                                                    whichSetNameStat=whichSetNameStat,
                                                    showFigures=showFiguresArg[0], useLabels=useLabels)
            elif classifierGoal == 'regression':
                linearRegression.makeStatisticsForModel(experimentsFolder, statisticsStoreFolder, featureParameters,
                                                        datasetStatsParameters, modelParameters, whichSetName=whichSetNameStat,
                                                        showFigures=showFiguresArg[0], useLabels=useLabels)
            else:
                raise ValueError("This classifier goal {0} not supported".format(classifierGoal))


if __name__ == '__main__':
    featureDataFolderMain = CreateUtils.getProcessedFeaturesFolder()
    if wildCards[0]:
        thisFeatureFolder = CreateUtils.getExperimentFolder() if onlyPreviousExperiments else featureDataFolderMain
        featureSets = [fileIterator for fileIterator in os.listdir(thisFeatureFolder) if os.path.isdir(
            os.path.join(thisFeatureFolder, fileIterator)) and fileIterator not in removeFeatureSetNames]
    else:
        featureSets = featureSetNameMain if isinstance(featureSetNameMain, list) else [featureSetNameMain]
    for featureSet in featureSets:
        processedDataFolderMain = CreateUtils.getProcessedDataDatasetsFolder()
        if wildCards[1]:
            thisDatasetFolder = CreateUtils.getExperimentFolder(featureSetName=featureSet) if onlyPreviousExperiments else processedDataFolderMain
            datasets = [fileIterator for fileIterator in os.listdir(thisDatasetFolder) if os.path.isdir(
                os.path.join(thisDatasetFolder, fileIterator)) and fileIterator not in removeDatasetNames]
        else:
            datasets = datasetNameMain if isinstance(datasetNameMain, list) else [datasetNameMain]
        for dataset in datasets:
            modelDataFolderMain = CreateUtils.getModelFolder()
            if wildCards[2]:
                thisClassifierTypeFolder = CreateUtils.getExperimentFolder(featureSetName=featureSet, datasetName=dataset) \
                    if onlyPreviousExperiments else modelDataFolderMain
                classifierTypes = [fileIterator for fileIterator in os.listdir(thisClassifierTypeFolder) if
                                   os.path.isdir(
                                       os.path.join(thisClassifierTypeFolder,
                                                    fileIterator)) and fileIterator not in removeClassifierTypeNames]
            else:
                classifierTypes = classifierTypeMain if isinstance(classifierTypeMain, list) else [classifierTypeMain]
            for classifierTypeIterator in classifierTypes:
                modelTypeDataFolderMain = CreateUtils.getModelFolder(classifierType=classifierTypeIterator)
                if wildCards[3]:
                    thisClassifierSetFolder = CreateUtils.getExperimentFolder(featureSetName=featureSet,
                                                                              datasetName=dataset,
                                                                              classifierType=classifierTypeIterator) \
                        if onlyPreviousExperiments else modelDataFolderMain
                    classifierSets = [fileIterator for fileIterator in os.listdir(thisClassifierSetFolder) if
                                      os.path.isdir(
                                          os.path.join(thisClassifierSetFolder,
                                                       fileIterator)) and fileIterator not in removeClassifierSetNames]
                else:
                    classifierSets = classifierSetNameMain if isinstance(classifierSetNameMain, list) else [
                        classifierSetNameMain]
                for classifierSet in classifierSets:
                    runExperiment(featureSet,
                                  dataset,
                                  classifierTypeIterator,
                                  classifierSet,
                                  whichSetName=whichSetNameMaster,
                                  whichSetNameStat=whichSetNameStats,
                                  showFiguresArg=(showFigures, showKerasFigure),
                                  datasetNameStats=datasetNameStatsMain,
                                  modelStoreNameTypeArg=modelStoreNameType)
