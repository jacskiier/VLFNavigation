import os
import re
import numpy as np

# Feature Statics
signalSources = ['Loop Antenna with iPhone 4', '3-Axis Dipole With SRI Receiver']
featureMethodNames = ['Patch', 'Covariance', 'MFCC', 'FFT', 'FFTWindow', 'RawAmplitude' 'MNIST', 'Test', 'THoR']
featureMethodNamesRebuildValid = ['Patch', 'Covariance', 'MFCC', 'FFT', 'FFTWindow', 'RawAmplitude']
# Patch Specific
statOrderNames = ['mean', 'variance', 'standard deviation', 'skewness', 'kurtosis']

# Dataset Statics
yValueTypes = ['file', 'gpsD', 'time', 'gpsC', 'gpsPolar', 'particle']
yValueGPSTypes = ['gpsD', 'gpsC', 'gpsPolar', 'particle']
yValueDiscreteTypes = ['file', 'gpsD', 'time', 'particle']
yValueContinuousTypes = ['gpsC', 'gpsPolar']
assert len(set(yValueContinuousTypes + yValueDiscreteTypes)) == len(yValueTypes), "Not all sets accounted for in yValueTypes"

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
footpadNames = ['TotalStrides', 'TotalDistanceFootpad', 'CadenceFootpad', 'SpeedFootpad']
maNames = ['GroundContactTime', 'MotionCount', 'MotionPowerZ', 'CadenceMA', 'MotionPowerX', 'Smoothness',
           'MotionPowerY', 'VerticalOscillation']
accelerometerNames = ['x', 'y', 'z']
gyroscopeNames = ['xrate', 'yrate', 'zrate']
locationiPhoneNames = ['AltitudeiPhone', 'course', 'horizontalAccuracy', 'LatitudeiPhone', 'LongitudeiPhone',
                       'SpeediPhone', 'verticalAccuracy']
headingNames = ['trueHeading', 'headingAccuracy', 'magneticHeading']
timeNames = ['ElapsedSeconds', 'Hour', 'DayOfYear', 'DayPercent', 'SeasonSineWave']
otherNames = ['BaseFileName', 'BaseFileNameWithNumber']

allMetadataNamesList = cadenceNames + speedNames + locationWahooNames + heartNames + footpadNames + maNames + \
                       accelerometerNames + gyroscopeNames + locationiPhoneNames + headingNames + timeNames + otherNames
allMetadataNamesSet = set(allMetadataNamesList)
assert len(allMetadataNamesList) == len(allMetadataNamesSet), "There are name collisions in metadata names"

cadenceDtypes = [('CadenceBike', float), ('CrankRevolutions', int)]
speedDtypes = [('SpeedInstant', float), ('WheelRevolutions', int), ('WheelCircumference', float)]
locationWahooDtypes = [('LongitudeWahoo', float), ('DistanceOffset', float), ('Accuracy', float),
                       ('AltitudeWahoo', float), ('LatitudeWahoo', float), ('TotalDistanceWahoo', float),
                       ('GradeDeg', float), ('SpeedWahoo', float)]
heartDtypes = [('TotalHeartbeats', int), ('Heartrate', int)]
footpadDtypes = [('TotalStrides', float), ('TotalDistanceFootpad', float), ('CadenceFootpad', int),
                 ('SpeedFootpad', float)]
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


allMetadataDtyesList = cadenceDtypes + speedDtypes + locationWahooDtypes + heartDtypes + footpadDtypes + maDtypes + \
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
        largestFileNumberRemove = np.max(np.array(removeFileNumbers[baseFileName])) if len(removeFileNumbers[baseFileName]) > 0 else 0
        largestFileNumberOnly = np.max(np.array(onlyFileNumbers[baseFileName])) if len(onlyFileNumbers[baseFileName]) > 0 else 0
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


def getRawDataFolder():
    # os.path.abspath(os.path.expanduser("~/Virtual Box Shared Folder/"))
    if os.name == 'nt':
        rawDataFolder = os.path.join("E:\\", "Users", "Joey", "Documents",
                                     "Virtual Box Shared Folder")  # VLF signals raw data folder
        # rawDataFolder = r"E:\\Users\\Joey\\Documents\\DataFolder\\" # 3 Axis VLF Antenna signals raw data folder
        # rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\Spyder\\deeplearningfiles\\mnist raw data folder\\" # MNIST raw data folder
        # rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\Spyder\\deeplearningfiles\\test raw data folder\\" # Test raw data folder
        # rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\parse NHL\\"
    elif os.name == 'posix':
        rawDataFolder = os.path.join("/media", "sena", "Greed Island", "Users", "Joey", "Documents",
                                     "Virtual Box Shared Folder")  # VLF signals raw data folder

    else:
        raise ValueError("This OS is not allowed")
    return rawDataFolder
