import os
import re
import numpy as np
import yaml
import shutil

masterFeatureMethod = 'SignalPlaceholder'
# Feature Statics
signalSources = ['Loop Antenna with iPhone 4', '3-Axis Dipole With SRI Receiver']
featureMethodNames = ['Patch', 'Covariance', 'MFCC', 'FFT', 'FFTWindow', 'RawAmplitude' 'MNIST', 'Test', 'THoR']
featureMethodNamesRebuildValid = ['Patch', 'Covariance', 'MFCC', 'FFT', 'FFTWindow', 'RawAmplitude', 'SignalPlaceholder']
# Patch Specific
statOrderNames = ['mean', 'variance', 'standard deviation', 'skewness', 'kurtosis']
# FFT Window Specific
all_frequencyStats = ['power', 'in_phase_amplitude', 'out_of_phase_amplitude', 'angle', 'relative_angle']

# Dataset Statics
yValueTypes = ['file', 'gpsD', 'time', 'gpsC', 'gpsPolar', 'particle']
yValueGPSTypes = ['gpsD', 'gpsC', 'gpsPolar', 'particle']
yValueDiscreteTypes = ['file', 'gpsD', 'time', 'particle']
yValueContinuousTypes = ['gpsC', 'gpsPolar']
assert len(set(yValueContinuousTypes + yValueDiscreteTypes)) == len(yValueTypes), "Not all sets accounted for in yValueTypes"

rowPackagingTypes = [None, 'BaseFileNameWithNumber', 'gpsD', 'particle', 'class', 'classWithClassTransitions']
metadataMultipliersFromRowPackagingStyle = {
    None: 0,
    'BaseFileNameWithNumber': 1,
    'gpsD': 1,
    'particle': 1,
    'class': 1,
    'classWithClassTransitions': 2
}

# Classifier Statics
classifierTypes = ['LogisticRegression', 'MLP', 'ConvolutionalMLP', 'DBN', 'RandomForest', 'ADABoost']
sklearnensembleTypes = ['RandomForest', 'ADABoost', 'GradientBoosting', 'GaussianProcess']
kerasTypes = ['LSTM']

# region Metadata types
cadenceNames = ['CadenceBike', 'CrankRevolutions']
speedNames = ['SpeedInstant', 'WheelRevolutions', 'WheelCircumference']
locationWahooNames = ['LongitudeWahoo', 'DistanceOffset', 'Accuracy', 'AltitudeWahoo', 'LatitudeWahoo',
                      'TotalDistanceWahoo', 'GradeDeg', 'SpeedWahoo']
heartNames = ['TotalHeartbeats', 'Heartrate']
footpodNames = ['TotalStrides', 'TotalDistanceFootpod', 'CadenceFootpod', 'SpeedFootpod']
maNames = ['GroundContactTime', 'MotionCount', 'MotionPowerZ', 'CadenceMA', 'MotionPowerX', 'Smoothness',
           'MotionPowerY', 'VerticalOscillation']
accelerometerNames = ['x', 'y', 'z']
gyroscopeNames = ['xrate', 'yrate', 'zrate']
locationiPhoneNames = ['AltitudeiPhone', 'course', 'horizontalAccuracy', 'LatitudeiPhone', 'LongitudeiPhone',
                       'SpeediPhone', 'verticalAccuracy']
headingNames = ['trueHeading', 'headingAccuracy', 'magneticHeading']
timeNames = ['ElapsedSeconds', 'Hour', 'DayOfYear', 'DayPercent', 'SeasonSineWave']
otherNames = ['BaseFileName', 'BaseFileNameWithNumber']

allMetadataNamesList = cadenceNames + speedNames + locationWahooNames + heartNames + footpodNames + maNames + \
                       accelerometerNames + gyroscopeNames + locationiPhoneNames + headingNames + timeNames + otherNames
allMetadataNamesSet = set(allMetadataNamesList)
assert len(allMetadataNamesList) == len(allMetadataNamesSet), "There are name collisions in metadata names"

cadenceDtypes = [('CadenceBike', float), ('CrankRevolutions', int)]
speedDtypes = [('SpeedInstant', float), ('WheelRevolutions', int), ('WheelCircumference', float)]
locationWahooDtypes = [('LongitudeWahoo', float), ('DistanceOffset', float), ('Accuracy', float),
                       ('AltitudeWahoo', float), ('LatitudeWahoo', float), ('TotalDistanceWahoo', float),
                       ('GradeDeg', float), ('SpeedWahoo', float)]
heartDtypes = [('TotalHeartbeats', int), ('Heartrate', int)]
footpodDtypes = [('TotalStrides', float), ('TotalDistanceFootpod', float), ('CadenceFootpod', int),
                 ('SpeedFootpod', float)]
maDtypes = ndtype = [('GroundContactTime', float), ('MotionCount', int), ('MotionPowerZ', float), ('CadenceMA', int),
                     ('MotionPowerX', float), ('Smoothness', float), ('MotionPowerY', float),
                     ('VerticalOscillation', float)]
accelerometerDtypes = [('x', float), ('y', float), ('z', float)]
gyroscopeDtypes = [('xrate', float), ('yrate', float), ('zrate', float)]
locationiPhoneDtypes = [('AltitudeiPhone', float), ('course', float), ('horizontalAccuracy', int),
                        ('LatitudeiPhone', float), ('LongitudeiPhone', float), ('SpeediPhone', float),
                        ('verticalAccuracy', int)]
headingDtypes = [('trueHeading', float), ('headingAccuracy', int), ('magneticHeading', float)]
timeNamesDtype = [('ElapsedSeconds', float), ('Hour', int), ('DayOfYear', int), ('DayPercent', float), ('SeasonSineWave', float)]
otherNamesDtype = [('BaseFileName', "|S30"), ('BaseFileNameWithNumber', "|S30")]


def getNamesFromDtypes(dtyper):
    return [tup[0] for tup in dtyper]


allMetadataDtyesList = cadenceDtypes + speedDtypes + locationWahooDtypes + heartDtypes + footpodDtypes + maDtypes + \
                       accelerometerDtypes + gyroscopeDtypes + locationiPhoneDtypes + headingDtypes + \
                       timeNamesDtype + otherNamesDtype
allMetadataDtyesNamesList = getNamesFromDtypes(allMetadataDtyesList)
allMetadataDtypesNamesSet = set(allMetadataDtyesNamesList)
assert len(allMetadataDtyesNamesList) == len(allMetadataDtypesNamesSet), "The Dtypes had a duplicate name {0}"

assert len(set.union(allMetadataNamesSet, allMetadataDtypesNamesSet)) == len(allMetadataNamesSet) == len(
    allMetadataDtypesNamesSet), "The names and Dtypes don't match exactly"


# endregion


class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """

    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)

    def added(self):
        return self.set_current - self.intersect

    def removed(self):
        return self.set_past - self.intersect

    def changed(self):
        outSet = set()
        for o in self.intersect:
            if type(self.past_dict[o]).__module__ == np.__name__ or type(
                    self.current_dict[o]).__module__ == np.__name__:
                if np.array((self.past_dict[o] != self.current_dict[o])).any():
                    outSet.add(o)
            elif self.past_dict[o] != self.current_dict[o]:
                outSet.add(o)
        # return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])
        return outSet

    def unchanged(self):
        outSet = set()
        for o in self.intersect:
            if type(self.past_dict[o]).__module__ == np.__name__ or type(
                    self.current_dict[o]).__module__ == np.__name__:
                if np.array((self.past_dict[o] == self.current_dict[o])).all():
                    outSet.add(o)
            elif self.past_dict[o] == self.current_dict[o]:
                outSet.add(o)
        # return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])
        return outSet

    def printAll(self):
        return """Added:     {added}
Removed:   {removed}
Changed:   {changed}
Unchanged: {unchanged}""".format(added=self.added(), removed=self.removed(), changed=self.changed(),
                                 unchanged=self.unchanged())

    def printAllDiff(self):
        return """Added:     {added}
Removed:   {removed}
Changed:   {changed}""".format(added=self.added(), removed=self.removed(), changed=self.changed(),
                               unchanged=self.unchanged())


def getPathRelativeToRoot(abspath):
    rootDataFolder = getRootDataFolder()
    relpath = os.path.relpath(abspath, rootDataFolder)
    return relpath


def getAbsolutePath(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(getRootDataFolder(), path)


def convertPathToThisOS(path):
    isNT = re.match(r"\S:[\\]+", path)
    isPOSIX = re.match(r"/[^/]+", path)
    thisPathOS = ''
    if isNT:
        thisPathOS = 'nt'
    elif isPOSIX:
        thisPathOS = 'posix'

    retPath = path
    if os.name == 'nt' and thisPathOS == 'posix':
        r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"
        posixMatch = re.match(
            r"(?P<basePath>/media/sena/Greed Island/Users/Joey/Documents/Virtual Box Shared Folder/?)(?P<endPath>.*)",
            path)
        if posixMatch:
            ntBasePathArray = ["E:\\", "Users", "Joey", "Documents", "Virtual Box Shared Folder"]
            endPath = posixMatch.group("endPath")
            endPathArray = endPath.split("/")
            retPath = os.path.join(*ntBasePathArray + endPathArray)
    elif os.name == 'posix' and thisPathOS == 'nt':
        ntMatch = re.match(
            r"(?P<basePath>E:[\\]+Users[\\]+Joey[\\]+Documents[\\]+Virtual Box Shared Folder[\\]*)(?P<endPath>.*)",
            path)
        if ntMatch:
            linuxBasePathArray = ["/media", "sena", "Greed Island", "Users", "Joey", "Documents",
                                  "Virtual Box Shared Folder"]  # VLF signals raw data folder
            endPath = ntMatch.group("endPath")
            endPathArray = endPath.split("\\")
            retPath = os.path.join(*linuxBasePathArray + endPathArray)

    return retPath


def getSetNameForFile(filename, defaultSetName, fileNamesNumbersToSets):
    """
    :param filename: name of the base file we want to assign a set to
    :param defaultSetName: the default name for the set if this filename isn't in the fileNamesNumberstoSets array
    :param fileNamesNumbersToSets: array of tuples that give set names
        to the filename [(setName, fileBaseName, fileNumber), ... ]
    :return: the set name for this file
    """
    setNames = []
    matcher = re.match(r'(?P<baseName>[\d\w]+)(?P<fileNumber>[\d]{5,10})\.(wav|hf)', filename)
    if matcher:
        filename = matcher.group("baseName")
        for setNameAndNumbers in fileNamesNumbersToSets:
            if filename == setNameAndNumbers[1]:
                fileNumber = int(matcher.group("fileNumber"))
                if fileNumber in setNameAndNumbers[2]:
                    setNames.append(setNameAndNumbers[0])
    else:
        print ("file didn't match problem {0}".format(filename))
    if len(setNames) == 0:
        setNames.append(defaultSetName)
    return setNames


def filterFilesByFileNumber(files, baseFileName, removeFileNumbers=(), onlyFileNumbers=()):
    if baseFileName in removeFileNumbers or baseFileName in onlyFileNumbers:
        largestFileNumberRemove = np.max(np.array(removeFileNumbers[baseFileName])) if baseFileName in removeFileNumbers and len(
            removeFileNumbers[baseFileName]) > 0 else 0
        largestFileNumberOnly = np.max(np.array(onlyFileNumbers[baseFileName])) if baseFileName in onlyFileNumbers and len(
            onlyFileNumbers[baseFileName]) > 0 else 0
        largestFileNumber = max(largestFileNumberRemove, largestFileNumberOnly)
        errorString = "you have a file number {0} that is over the possible files {1}".format(largestFileNumber, files.size)
        assert largestFileNumber < files.size, errorString
        removeMask = np.ones(files.shape, dtype=bool)
        if baseFileName in removeFileNumbers and len(removeFileNumbers[baseFileName]) > 0:
            removeMask[np.array(removeFileNumbers[baseFileName])] = False
        onlyMask = np.ones(files.shape, dtype=bool)
        if baseFileName in onlyFileNumbers and len(onlyFileNumbers[baseFileName]) > 0:
            onlyMask[np.array(onlyFileNumbers[baseFileName])] = False
            onlyMask = np.logical_not(onlyMask)
        finalMask = np.logical_and(removeMask, onlyMask)
        files = files[finalMask]
    return files


def getAllBaseFileNames(rawDataFolderArg, nomatch=None):
    rawFiles = os.listdir(rawDataFolderArg)
    fileSet = set()
    for rawFile in rawFiles:
        matcher = re.match(r'(?P<baseName>[\d\w]+)[\d]{5,10}\.(wav|hf)', rawFile)
        if matcher:
            baseName = matcher.group('baseName')
            if nomatch is None or baseName not in nomatch:
                fileSet.add(baseName)
    if 'myMusicFile' in fileSet:
        fileSet.remove('myMusicFile')

    return list(fileSet)


def getRootDataFolder(featureMethod=None, signalSource='Loop Antenna With iPhone 5c', includeSamplingRate=False):
    samplingRate = None
    global masterFeatureMethod
    if featureMethod is None:
        featureMethod = masterFeatureMethod
    else:
        masterFeatureMethod = featureMethod

    # get source root based on OS
    if os.name == 'nt':
        sourceRoot = "M:\\"
    elif os.name == 'posix':
        sourceRoot = os.path.join("/media", "sena", "Mystery Shack")
    else:
        raise ValueError("This OS is not allowed")

    # get the root data folder based on the feature method
    if featureMethod in featureMethodNamesRebuildValid:
        if signalSource == 'Loop Antenna With iPhone 4' or signalSource == 'Loop Antenna With iPhone 5c':
            rootDataFolder = os.path.join(sourceRoot, "iPhoneVLFSignals")
            samplingRate = 44100
        elif signalSource == '3-Axis Dipole With SRI Receiver':
            rootDataFolder = os.path.join(sourceRoot, "3AxisVLFSignals")
            samplingRate = 200000
        else:
            raise ValueError("Signal Source of {0} is not supported".format(signalSource))
    elif featureMethod == 'MNIST':
        rootDataFolder = os.path.join(sourceRoot, "MNIST Data Root")
    elif featureMethod == 'THoR':
        rootDataFolder = os.path.join(sourceRoot, "THoR Data Root")
    else:  # featureMethod == "Test"
        rootDataFolder = os.path.join(sourceRoot, "Test Data Root")

    if includeSamplingRate:
        ret = (rootDataFolder, samplingRate)
    else:
        ret = rootDataFolder
    return ret


def getRawDataFolder():
    rootDataFolder = getRootDataFolder()
    rawDataFolder = os.path.join(rootDataFolder, "Raw Data")
    return rawDataFolder


def getProcessedFeaturesFolder(featureName=None):
    if featureName is None:
        ret = os.path.join(getRootDataFolder(), "Processed Data Features")
    else:
        ret = os.path.join(getRootDataFolder(), "Processed Data Features", featureName)
    return ret


def getProcessedDataDatasetsFolder(datasetName=None):
    if datasetName is None:
        processedDataFolder = os.path.join(getRootDataFolder(), "Processed Data Datasets")
    else:
        processedDataFolder = os.path.join(getRootDataFolder(), "Processed Data Datasets", datasetName)
    return processedDataFolder


def getModelFolder(classifierType=None, classifierSetName=None, baseFolder=None):
    if baseFolder is None:
        rootFolder = getRootDataFolder()
    else:
        rootFolder = baseFolder
    if classifierType is None or classifierSetName is None:
        modelFolder = os.path.join(rootFolder, "Processed Data Models")
    elif classifierType is not None and classifierSetName is None:
        modelFolder = os.path.join(rootFolder, "Processed Data Models", classifierType)
    else:
        modelFolder = os.path.join(rootFolder, "Processed Data Models", classifierType, classifierSetName)
    return modelFolder


def getExperimentFolder(featureSetName=None, datasetName=None, classifierType=None, classifierSetName=None):
    if featureSetName is None and datasetName is None and classifierType is None and classifierSetName is None:
        experimentFolder = os.path.join(getRootDataFolder(), "Data Experiments")
    elif featureSetName is not None and datasetName is not None and classifierType is None and classifierSetName is None:
        experimentFolder = os.path.join(getRootDataFolder(), "Data Experiments", featureSetName)
    elif featureSetName is not None and datasetName is None and classifierType is None and classifierSetName is None:
        experimentFolder = os.path.join(getRootDataFolder(), "Data Experiments", featureSetName, datasetName)
    elif featureSetName is not None and datasetName is None and classifierType is not None and classifierSetName is None:
        experimentFolder = os.path.join(getRootDataFolder(), "Data Experiments", featureSetName, datasetName, classifierType)
    else:
        experimentFolder = os.path.join(getRootDataFolder(), "Data Experiments", featureSetName, datasetName, classifierType, classifierSetName)
    return experimentFolder


def getRandomExperimentBaseFolder(featureSetName=None, datasetName=None, classifierType=None, classifierSetName=None, randomName=None):
    experimentFolder = getExperimentFolder(featureSetName=featureSetName,
                                           datasetName=datasetName,
                                           classifierType=classifierType,
                                           classifierSetName=classifierSetName)
    baseRandomName = "Random" if randomName is None else "Random {0}".format(randomName)
    randomExperimentFolder = os.path.join(experimentFolder, baseRandomName)
    return randomExperimentFolder


def getRandomExperimentFolder(randomSeedString, featureSetName=None, datasetName=None, classifierType=None, classifierSetName=None, randomName=None):
    experimentFolder = getRandomExperimentBaseFolder(featureSetName=featureSetName,
                                                     datasetName=datasetName,
                                                     classifierType=classifierType,
                                                     classifierSetName=classifierSetName,
                                                     randomName=randomName)
    randomExperimentFolder = os.path.join(experimentFolder, randomSeedString)
    return randomExperimentFolder


def getRandomConfigFileName(featureSetName=None, datasetName=None, classifierType=None, classifierSetName=None, randomName=None, baseFolder=None):
    if baseFolder is None:
        randomExperimentBaseFolder = getRandomExperimentBaseFolder(featureSetName, datasetName, classifierType, classifierSetName, randomName)
    else:
        randomExperimentBaseFolder = baseFolder
    return os.path.join(randomExperimentBaseFolder, "random parameters.yaml")


def getStatisticsFolder(experimentsFolder, datasetNameStats, whichSetNameStat):
    statisticsStoreFolder = os.path.join(experimentsFolder, datasetNameStats, whichSetNameStat)
    return statisticsStoreFolder


def getImageryFolder():
    return os.path.join(getRootDataFolder(), "Imagery")


def getDatasetFile(featureSetName, datasetName, baseFolder=None, checkExistence=True):
    if baseFolder is None:
        processedDataFolder = getProcessedDataDatasetsFolder(datasetName)
    else:
        processedDataFolder = baseFolder
    if checkExistence is False:
        datasetFile = os.path.join(processedDataFolder, featureSetName + '.hf')
    else:
        if os.path.exists(os.path.join(processedDataFolder, featureSetName + '.hf')):
            datasetFile = os.path.join(processedDataFolder, featureSetName + '.hf')
        elif os.path.exists(os.path.join(processedDataFolder, featureSetName, datasetName + '.pkl.gz')):
            datasetFile = os.path.join(processedDataFolder, featureSetName, datasetName + '.pkl.gz')
        else:
            raise ValueError("The given dataset file combination does not exist")
    return datasetFile


def getParameters(featureSetName=None, datasetName=None, classifierType=None, classifierSetName=None, baseFolder=None):
    returnParameters = ()
    if featureSetName is not None:
        featureConfigFileName = getFeatureConfigFileName(featureSetName, baseFolder=baseFolder)
        with open(featureConfigFileName, 'r') as myConfigFile:
            featureParameters = yaml.load(myConfigFile)
        returnParameters += (featureParameters,)

    if datasetName is not None:
        datasetConfigFileName = getDatasetConfigFileName(datasetName, baseFolder=baseFolder)
        with open(datasetConfigFileName, 'r') as myConfigFile:
            datasetParameters = yaml.load(myConfigFile)
        returnParameters += (datasetParameters,)

    if classifierType is not None and classifierSetName is not None:
        modelConfigFileName = getModelConfigFileName(classifierType, classifierSetName, baseFolder=baseFolder)
        with open(modelConfigFileName, 'r') as myConfigFile:
            modelParameters = yaml.load(myConfigFile)
        returnParameters += (modelParameters,)
    return returnParameters


def get3AxisCollectFolder():
    if os.name == 'nt':
        dataCollectFolderMain = os.path.join("L:", "Thesis Files", "afitdata")
    elif os.name == 'posix':
        dataCollectFolderMain = os.path.join("media", "sena", "blue", "Thesis Files", "afitdata")
    else:
        raise ValueError("This OS is not allowed")
    return dataCollectFolderMain


def getFeatureConfigFileName(featureSetName, baseFolder=None):
    if baseFolder is None:
        featureConfigFileName = os.path.join(getProcessedFeaturesFolder(featureSetName), "feature parameters.yaml")
    else:
        featureConfigFileName = os.path.join(baseFolder, "feature parameters.yaml")
    return featureConfigFileName


def getDatasetConfigFileName(datasetName=None, baseFolder=None):
    if datasetName is not None:
        datasetConfigFileName = os.path.join(getProcessedDataDatasetsFolder(datasetName), "dataset parameters.yaml")
    elif baseFolder is not None:
        datasetConfigFileName = os.path.join(baseFolder, "dataset parameters.yaml")
    else:
        raise ValueError("One parameter must be supplied")
    return datasetConfigFileName


def getModelConfigFileName(classifierType=None, classifierSetName=None, baseFolder=None):
    if classifierType is not None and classifierSetName is not None:
        modelConfigFileName = os.path.join(getModelFolder(classifierType=classifierType, classifierSetName=classifierSetName),
                                           "model set parameters.yaml")
    elif baseFolder is not None:
        modelConfigFileName = os.path.join(baseFolder, "model set parameters.yaml")
    else:
        raise ValueError("Some arguments must be given")
    return modelConfigFileName


def getDatasetStatConfigFileName(statisticsFolder):
    return os.path.join(statisticsFolder, 'dataset parameters.yaml')


def copyConfigsToExperimentsFolder(experimentsFolder, featureSetName=None, datasetName=None, classifierType=None, classifierSetName=None):
    if featureSetName is not None:
        featureConfigFileName = getFeatureConfigFileName(featureSetName)
        shutil.copyfile(featureConfigFileName, os.path.join(experimentsFolder, os.path.basename(featureConfigFileName)))
    if datasetName is not None:
        datasetConfigFileName = getDatasetConfigFileName(datasetName)
        shutil.copyfile(datasetConfigFileName, os.path.join(experimentsFolder, os.path.basename(datasetConfigFileName)))
    if classifierType is not None and classifierSetName is not None:
        modelConfigFileName = getModelConfigFileName(classifierType, classifierSetName)
        shutil.copyfile(modelConfigFileName, os.path.join(experimentsFolder, os.path.basename(modelConfigFileName)))


def makeConfigFile(configFileName, configDict):
    with open(configFileName, 'w') as myConfigFile:
        yaml.dump(configDict, myConfigFile, default_flow_style=False, width=1000)
    return


def loadConfigFile(configFileName):
    with open(configFileName, 'r') as myConfigFile:
        parameters = yaml.load(myConfigFile)
    return parameters


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
