import yaml
import readwav
import os

yValueTypes = ['file', 'gpsD', 'time', 'gpsC', 'gpsPolar']
yValueGPSTypes = ['gpsD', 'gpsC', 'gpsPolar']
yValueDiscreteTypes = ['file', 'gpsD', 'time']
yValueContinuousTypes = ['gpsC', 'gpsPolar']

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

if __name__ == '__main__':
    # Run Parameters   ############
    runNow = True
    forceRefreshFeatures = False
    overwriteConfigFile = True
    forceRefreshDataset = True
    rebuildFromConfig = True
    rebuildAllFromConfig = False
    removeDatasetNames = []
    removeFileNumbers = {}
    onlyFileNumbers = {}
    removeFeatureSetNames = []
    onlyThisFeatureSetNames = ['PatchShortTallAllFreq']
    showFigures = True

    # Parameters Begin ############
    shuffleFinalSamples = False
    shuffleSamplesPerFile = False
    rngSeed = 0
    maxSamplesPerFile = 0

    setFractions = [("train", "valid", 1.0 / 7), ("train", "test", 1.0 / 7)]

    # X value changes ######################################################
    # Sequences
    makeSequences = False
    timestepsPerSequence = 1000  # if None use longest file timesteps
    offsetBetweenSequences = 1000  # negative or 0 will mean use random start locations
    fractionOfRandomToUse = 1.0
    padSequenceWithZeros = True
    # if False will use only full sequences, if True will extend last sequence and pad with zeros
    repeatSequenceBeginningAtEnd = True
    # if padSequenceWithZeros is True this will replace the zeros with the beginning of the sequence
    repeatSequenceEndingAtEnd = False
    # if padSequenceWithZeros is True this will replace the zeros with the end of the sequence backwards
    assert not (repeatSequenceBeginningAtEnd and repeatSequenceEndingAtEnd), \
        "you can't have both repeatSequenceBeginningAtEnd and repeatSequenceEndingAtEnd"

    # Packaging
    rowPackagingStyle = 'gpsD'  # None, 'BaseFileNameWithNumber', 'gpsD'
    gridSizePackage = (100, 100, 1000)
    padRowPackageWithZeros = True
    repeatRowPackageBeginningAtEnd = False
    repeatRowPackageEndingAtEnd = True
    assert not (repeatRowPackageBeginningAtEnd and repeatRowPackageEndingAtEnd), \
        "you can't have both repeatRowPackageBeginningAtEnd and repeatRowPackageEndingAtEnd"
    allSetsSameRows = True
    # keras packaging
    alternateRowsForKeras = True
    timestepsPerKerasBatchRow = 1000
    assert not (not allSetsSameRows and alternateRowsForKeras), \
        "You must keep all sets same rows for keras packging to work in keras"

    # filter features
    filterXToUpperDiagonal = False
    filterPCA = True
    filterFitSets = ["train"]  # names of the sets you want to use to filter
    filterPCAn_components = None
    filterPCAwhiten = True

    # x scaling
    xScaleFactor = 1.0
    xBias = 0.0
    xNormalized = False

    # y value changes #######################################################
    timeDistributedY = True
    # y scale factor for continuous sets
    yScaleFactor = 1.0
    yBias = 0.0
    yNormalized = True

    # yvalue by gps variables
    includeAltitude = False
    # precision for lat, lon, and altitude to round to a grid
    decimalPrecision = (4, 4, -1)
    # size of the grid used for gps discrete blocks
    gridSize = (20, 20, 1000)
    # localLevelOriginInECEF = [506052.051626,-4882162.055080,4059778.630410] # AFIT
    localLevelOriginInECEF = [507278.89822834, -4884824.02376298, 4056425.76820216]  # Neighborhood Center

    # metadata features
    useMetadata = True
    metadataList = ['CadenceBike', 'CrankRevolutions', 'SpeedInstant', 'WheelRevolutions', 'DayPercent']
    metadataShape = (len(metadataList),)

    assert len(set(metadataList).union(allMetadataNamesSet)) == len(
        allMetadataNamesList), "You are using a metadata name not in the master list"
    # Other Datasets ###
    ######################

    # datasetName = 'MNIST'
    # allBaseFileNames = ['MNIST']
    # yValueType = 'Written Number'

    # datasetName = 'Test'
    # allBaseFileNames = ['Test']
    # yValueType = 'Multiple outputs'

    # datasetName = 'THoR'
    # allBaseFileNames = ["xAll"]
    # yValueType = 'Goal Value'
    # Static Datasets  ###
    ########################

    # datasetName = 'staticLocationsSequence'
    # allBaseFileNames = ['642lobby','646lobby','library','640244class','enyconf','einstein','jimsoffice',
    # 'dolittles','kennylobby','642lobbytop','antoffice','cube','640sideroom',
    # 'algebra','engconf','640338class','640220lab']
    # ##allBaseFileNames =  readwav.getAllBaseFileNames(rawDataFolder, nomatch = ["afitwalk","afitwalkFail",
    # "afitinsidewalk",'mixer','microwave','transformer', 'kitchentable'])
    # yValueType = 'file'

    # datasetName = 'staticLocations'
    # allBaseFileNames = ['afittest', 'hometest']
    # yValueType = 'file'

    # datasetName = 'threeClassProblemSequence'
    # allBaseFileNames = ['algebra', 'cube', 'jimsoffice']
    # yValueType = 'file'

    # datasetName = 'threeClassProblemgpsC'
    # allBaseFileNames = ['algebra', 'cube', 'jimsoffice']
    # yValueType = 'gpsC'

    # datasetName = 'fourClassProblem'
    # allBaseFileNames = ['kitchentable','algebra', 'cube', 'jimsoffice']
    # yValueType = 'file'
    # Walking Datasets ###
    ########################

    # datasetName = 'vlfsignalsAfitInsideWalk'
    # allBaseFileNames = ["afitinsidewalk"]
    # yValueType = 'time'

    # datasetName = 'vlfsignalsAfitWalkAll'
    # allBaseFileNames = ["afitwalk", "afitwalkFail"]
    # yValueType = 'time'

    # datasetName = 'vlfsignalsAfitWalkSequenceC'
    # allBaseFileNames = ["afitwalk"]
    # yValueType = 'gpsC'

    # datasetName = 'vlfsignalsAfitWalkC'
    # allBaseFileNames = ["afitwalk"]
    # yValueType = 'gpsC'

    # datasetName = 'vlfsignalsAfitWalkSorted'
    # allBaseFileNames = ["afitwalk"]
    # yValueType = 'gpsC'
    # Car Datasets ###
    ####################

    # datasetName = 'toAFITandBackSequenceCTest'
    # allBaseFileNames = ["carafittohome","cartoafit"]
    # yValueType = 'gpsC'

    # datasetName = 'toYogaandBackSequenceC'
    # allBaseFileNames = ["carhometoyoga","caryogatohome"]
    # yValueType = 'gpsC'

    # datasetName = 'allcar'
    # allBaseFileNames = ["carhometoyoga","caryogatohome","carafittohome","cartoafit",
    # "carrandom", "carrandomback", "carneighborhood"]
    # yValueType = 'gpsC'

    # datasetName = 'carneighborhoodDiscrete'
    # allBaseFileNames = ["carneighborhood"]
    # yValueType = 'gpsD'
    # removeFileNumbers = {"carneighborhood":[0]}

    # datasetName = 'carneighborhoodNewSequenceC'
    # allBaseFileNames = ["carneighborhood"]
    # yValueType = 'gpsC'
    # removeFileNumbers = {"carneighborhood":[0,1]}
    # # onlyFileNumbers = {"carneighborhood":[4,5]}

    # datasetName = 'carneighborhoodTestSequenceC'
    # allBaseFileNames = ["carneighborhood"]
    # yValueType = 'gpsC'
    # # removeFileNumbers = {"carneighborhood":[0,1]}
    # onlyFileNumbers = {"carneighborhood": [11]}

    # datasetName = 'carAFITParkingLot'
    # allBaseFileNames = ["carafitparkinglot"]
    # yValueType = 'gpsC'
    # Bike Datasets ###
    #####################

    # datasetName = 'bikeneighborhoodC'
    # allBaseFileNames = ["bikeneighborhood"]
    # yValueType = 'gpsC'

    # datasetName = 'bikeneighborhoodSequenceC'
    # allBaseFileNames = ["bikeneighborhood"]
    # yValueType = 'gpsC'

    datasetName = 'bikeneighborhoodPackgpsDNormDTDMG20PG20'
    allBaseFileNames = ["bikeneighborhood"]
    yValueType = 'gpsD'
    onlyFileNumbers = {"bikeneighborhood": []}
    removeFileNumbers = {"bikeneighborhood": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]}
    defaultSetName = "train"
    fileNamesNumbersToSets = [("valid", "bikeneighborhood", [12, 14, 16, 18, 20, 22, 24]), ('test', "bikeneighborhood", [8])]

    # datasetName = 'bikeneighborhoodFilePackageCTDM'
    # allBaseFileNames = ["bikeneighborhood"]
    # yValueType = 'gpsC'
    # onlyFileNumbers = {"bikeneighborhood": []}
    # removeFileNumbers = {"bikeneighborhood": range(21) + [25]}
    # defaultSetName = "train"
    # fileNamesNumbersToSets = [("valid", "bikeneighborhood", [22]),
    #                           ('test', "bikeneighborhood", [23])]

    ################################
    # Parameters End   ############
    ################################

    processedDataFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)
    datasetConfigFileName = os.path.join(processedDataFolder, "dataset parameters.yaml")

    processedDataFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)

    datasetParametersToDump = {
        'rawDataFolder': rawDataFolder,
        'processedDataFolder': processedDataFolder,
        'datasetName': datasetName,
        'yValueType': yValueType,
        'removeFileNumbers': removeFileNumbers,
        'onlyFileNumbers': onlyFileNumbers,
        'y value parameters': {
            'y value by gps Parameters': {
                'includeAltitude': includeAltitude,
                'decimalPrecision': decimalPrecision,
                'gridSize': gridSize,
                'localLevelOriginInECEF': localLevelOriginInECEF
            },
            'yScaleFactor': yScaleFactor,
            'yBias': yBias,
            'yNormalized': yNormalized,
        },
        'allBaseFileNames': allBaseFileNames,
        'shuffleFinalSamples': shuffleFinalSamples,
        'shuffleSamplesPerFile': shuffleSamplesPerFile,
        'rngSeed': rngSeed,
        'maxSamplesPerFile': maxSamplesPerFile,
        'filterXToUpperDiagonal': filterXToUpperDiagonal,

        'xScaleFactor': xScaleFactor,
        'xBias': xBias,
        'xNormalized': xNormalized,

        'defaultSetName': defaultSetName,
        'fileNamesNumbersToSets': fileNamesNumbersToSets,
    }
    if makeSequences:
        sequencesDict = {
            'makeSequences': makeSequences,
            'timestepsPerSequence': timestepsPerSequence,
            'offsetBetweenSequences': offsetBetweenSequences,
            'fractionOfRandomToUse': fractionOfRandomToUse,
            'padSequenceWithZeros': padSequenceWithZeros,

            'timeDistributedY': timeDistributedY,
        }
        if padSequenceWithZeros:
            if repeatSequenceBeginningAtEnd:
                sequencesDict.update({'repeatSequenceBeginningAtEnd': repeatSequenceBeginningAtEnd, })
            elif repeatSequenceEndingAtEnd:
                sequencesDict.update({'repeatSequenceEndingAtEnd': repeatSequenceEndingAtEnd, })
        datasetParametersToDump.update(sequencesDict)
    if rowPackagingStyle is not None:
        packageDict = {
            'rowPackagingStyle': rowPackagingStyle,
            'padRowPackageWithZeros': padRowPackageWithZeros,
            'repeatRowPackageBeginningAtEnd': repeatRowPackageBeginningAtEnd,
            'repeatRowPackageEndingAtEnd': repeatRowPackageEndingAtEnd,
        }
        if rowPackagingStyle == 'gpsD':
            packageDict.update({'gridSizePackage':gridSizePackage})
        if alternateRowsForKeras:
            kerasDict = {
                'alternateRowsForKeras': alternateRowsForKeras,
                'timestepsPerKerasBatchRow': timestepsPerKerasBatchRow,
                'allSetsSameRows': allSetsSameRows,
            }
            packageDict.update(kerasDict)
        datasetParametersToDump.update(packageDict)
    if filterPCA:
        filterDict = {
            'filterPCA': filterPCA,
            'filterFitSets': filterFitSets,
            'filterPCAn_components': filterPCAn_components,
            'filterPCAwhiten': filterPCAwhiten,
        }
        datasetParametersToDump.update(filterDict)
    if useMetadata:
        metadataDict = {
            'useMetadata': useMetadata,
            'metadataList': metadataList,
            'metadataShape': metadataShape
        }
        datasetParametersToDump.update(metadataDict)

    if (not os.path.exists(datasetConfigFileName)) or overwriteConfigFile:
        if not os.path.exists(processedDataFolder):
            os.makedirs(processedDataFolder)
        configFileName = os.path.join(processedDataFolder, "dataset parameters.yaml")
        if not overwriteConfigFile:
            assert not os.path.exists(configFileName), 'do you want to overwirte the config file?'
        with open(configFileName, 'w') as myConfigFile:
            yaml.dump(datasetParametersToDump, myConfigFile, default_flow_style=False, width=1000)
        datasets = [datasetName]
    else:
        if rebuildAllFromConfig:
            processedDataFolderMain = os.path.join(rawDataFolder, "Processed Data Datasets")
            datasets = [fileIterator for fileIterator in os.listdir(processedDataFolderMain) if
                        os.path.isdir(os.path.join(processedDataFolderMain,
                                                   fileIterator)) and fileIterator not in removeDatasetNames]
        elif rebuildFromConfig:
            datasets = [datasetName]
        else:
            datasets = []

    if runNow:
        featureDataFolderRoot = os.path.join(rawDataFolder, "Processed Data Features")
        if len(onlyThisFeatureSetNames) == 0:
            featureSetNamesAll = os.listdir(featureDataFolderRoot)
            featureSetNames = list(featureSetNamesAll)
            for featureSetName in featureSetNamesAll:
                if featureSetName in removeFeatureSetNames:
                    featureSetNames.remove(featureSetName)
        else:
            featureSetNames = list(onlyThisFeatureSetNames)
        for featureSetName in featureSetNames:
            featureDataFolderMain = os.path.join(rawDataFolder, "Processed Data Features", featureSetName)
            featureConfigFileName = os.path.join(featureDataFolderMain, "feature parameters.yaml")
            with open(featureConfigFileName, 'r') as myConfigFile:
                featureParametersDefault = yaml.load(myConfigFile)
            for datasetName in datasets:
                processedDataFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)
                datasetConfigFileName = os.path.join(processedDataFolder, "dataset parameters.yaml")
                with open(datasetConfigFileName, 'r') as myConfigFile:
                    datasetParameters = yaml.load(myConfigFile)
                if overwriteConfigFile:
                    dictDiffer = readwav.DictDiffer(datasetParameters, datasetParametersToDump)
                    print(dictDiffer.printAllDiff())
                removeFileNumbers = datasetParameters['removeFileNumbers'] \
                    if 'removeFileNumbers' in datasetParameters else {}
                onlyFileNumbers = datasetParameters['onlyFileNumbers'] \
                    if 'onlyFileNumbers' in datasetParameters else {}
                readwav.buildFeatures(featureParametersDefault,
                                      allBaseFileNames=allBaseFileNames,
                                      removeFileNumbers=removeFileNumbers,
                                      onlyFileNumbers=onlyFileNumbers,
                                      forceRefreshFeatures=forceRefreshFeatures)
                readwav.buildDataSet(datasetParameters,
                                     featureParametersDefault,
                                     forceRefreshDataset=forceRefreshDataset)
                readwav.getStatisticsOnSet(datasetParameters,
                                           featureParameters=featureParametersDefault,
                                           allBaseFileNames=allBaseFileNames,
                                           removeFileNumbers=removeFileNumbers,
                                           onlyFileNumbers=onlyFileNumbers,
                                           showFiguresArg=showFigures)
