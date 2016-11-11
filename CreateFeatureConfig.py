import yaml
import readwav
import os
import numpy as np

#######################
## static variables ###
#######################

signalSources = ['Loop Antenna with iPhone 4', '3-Axis Dipole With SRI Receiver']
featureMethodNames = ['Patch', 'Covariance', 'MFCC', 'FFT', 'RawAmplitude' 'MNIST', 'Test', 'THoR']
featureMethodNamesRebuildValid = ['Patch', 'Covariance', 'MFCC', 'FFT', 'RawAmplitude']

# Patch Specific
statOrderNames = ['mean', 'variance', 'standard deviation', 'skewness', 'kurtosis']

if __name__ == '__main__':
    ################################
    ## Run Parameters Begin ########
    ################################
    runNow = False
    overwriteConfigFile = True
    forceRefreshFeatures = True  # rewrite existing files
    rebuildFromConfig = True  # rebuild current feature name
    rebuildAllFromConfig = False  # rebuild all current features
    removeFileNumbers = {}
    onlyFileNumbers = {}
    removeFeatureSetNames = []
    maxFiles = 100

    featureSetName = 'RawAmplitudeShortTime'
    featureMethod = 'RawAmplitude'
    signalSource = 'Loop Antenna With iPhone 4'
    # signalSource = '3-Axis Dipole With SRI Receiver'
    samplingRate = None

    if os.name == 'nt':
        if featureMethod in featureMethodNamesRebuildValid:
            if signalSource == 'Loop Antenna With iPhone 4':
                rawDataFolder = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"
                samplingRate = 44100
            elif signalSource == '3-Axis Dipole With SRI Receiver':
                rawDataFolder = r"E:\\Users\\Joey\\Documents\\DataFolder\\"
                samplingRate = 200000
            else:
                raise ValueError("Signal Source of {0} is not supported".format(signalSource))
        elif featureMethod == 'MNIST':
            rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\Spyder\\deeplearningfiles\\mnist raw data folder\\"
        elif featureMethod == 'THoR':
            rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\parse NHL\\"
        else:  # featureMethod == "Test"
            rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\Spyder\\deeplearningfiles\\test raw data folder\\"
    elif os.name == 'posix':
        if featureMethod in featureMethodNamesRebuildValid:
            if signalSource == 'Loop Antenna With iPhone 4':
                rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Virtual Box Shared Folder/"
                samplingRate = 44100
            elif signalSource == '3-Axis Dipole With SRI Receiver':
                rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/DataFolder/"
                samplingRate = 200000
            else:
                raise ValueError("Signal Source of {0} is not supported".format(signalSource))
        elif featureMethod == 'MNIST':
            rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/Spyder/deeplearningfiles/mnist raw data folder/"
        elif featureMethod == 'THoR':
            rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/parse NHL/"
        else:  # featureMethod == "Test"
            rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/Spyder/deeplearningfiles/test raw data folder/"
    else:
        raise ValueError("This OS is not allowed")

    ##################################
    ## Feature specific parameters ###
    ##################################

    featureParametersDefault = {}
    featureDataFolder = os.path.join(rawDataFolder, "Processed Data Features", featureSetName)
    configFileName = os.path.join(featureDataFolder, "feature parameters.yaml")
    if (not os.path.exists(configFileName)) or overwriteConfigFile:
        if featureMethod == 'Patch':
            # x matrix parameters
            fftPower = 10
            windowTimeLength = 0.1  # in seconds
            burstMethod = False
            windowFreqBounds = [0, 22500]  # the start and stop frequencies in Hz
            # patch specific
            NT = 1  # blocks in time of each patch
            NF = 50  # blocks in frequency of each patch
            useOverallWindowStat = False  # should the last set of stat features be the whole window
            # mean variance, standard deviation, skewness kurtosis
            statOrder = [1, 3, 4, 0]
            normalizeStats = True  # done on a patch by patch basis

            # calculated stats
            totalStats = len(statOrder)  # how many stats to use from each patch
            NTF = NT * NF  # total blocks in a patch
            totalFeatures = (NT * NF + useOverallWindowStat) * totalStats  # length of an fP row

            # shape of input features as C style reshaping
            imageShape = (NF, NT, totalStats)
            # image shape order says the order you have to shuffle to get the image shape as (channels,width,height)
            imageShapeOrder = (2, 0, 1)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': windowTimeLength,
                    'burstMethod': burstMethod,
                    'windowFreqBounds': windowFreqBounds,
                    'NT': NT,
                    'NF': NF,
                    'totalStats': totalStats,
                    'NTF': NTF,
                    'totalFeatures': totalFeatures,
                    'useOverallWindowStat': useOverallWindowStat,
                    'statOrder': statOrder,
                    'normalizeStats': normalizeStats,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'Covariance':

            # x matrix parameters
            fftPower = 5
            windowTimeLength = 1.0  # in seconds
            burstMethod = False
            windowFreqBounds = [0, 22500]  # the start and stop frequencies in Hz

            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 2 ** (fftPower - 1) + 1, 2 ** (fftPower - 1) + 1)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': windowTimeLength,
                    'burstMethod': burstMethod,
                    'windowFreqBounds': windowFreqBounds,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'MFCC':

            fftPower = 5

            winlen = 0.1
            winstep = 0.01
            numcep = 100
            nfilt = numcep * 2
            lowfreq = 0
            highfreq = 6000  # use none to get samplingRate/2
            windowFreqBounds = (lowfreq, highfreq)

            convertToInt = True
            convertToIntMax = 10000
            convertToIntMultiplier = None

            imageShapeOrder = (0, 1, 2)
            imageShape = (1, numcep, 1)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': winlen,
                    'winstep': winstep,
                    'numcep': numcep,
                    'nfilt': nfilt,
                    'windowFreqBounds': windowFreqBounds,
                    'convertToInt': convertToInt,
                    'convertToIntMax': convertToIntMax,
                    'convertToIntMultiplier': convertToIntMultiplier,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'FFT':
            # x matrix parameters
            fftPower = 10
            burstMethod = False
            windowFreqBounds = [0, 22500]  # the start and stop frequencies in Hz

            # Determine the frequencies for this run so I can make the output shape for this frequency
            freqs = np.fft.fftfreq(2 ** fftPower, 1.0 / samplingRate)


            def find_nearestIndex(array, value):
                idx = (np.abs(array - value)).argmin()
                return idx


            windowFreqIndex = [find_nearestIndex(freqs, windowFreqBounds[0]),
                               find_nearestIndex(freqs, windowFreqBounds[1])]

            windowTimeLength = (2 ** fftPower * (
            1.0 / samplingRate)) / 2.0  # in seconds ## OK technically the window length isn't /2 but I'm really looking for window split which is this
            imageShape = (1 + int(windowFreqIndex[1] - windowFreqIndex[0]),)
            imageShapeOrder = (0,)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'fftPower': fftPower,
                    'windowTimeLength': windowTimeLength,
                    'burstMethod': burstMethod,
                    'windowFreqBounds': windowFreqBounds,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'RawAmplitude':
            poolingSize = 1
            poolingType = None

            if poolingType == 'none' or poolingType is None:
                poolingSize = 1
            windowTimeLength = 1.0 / samplingRate * poolingSize

            maxTime = 10 # in seconds

            imageShape = (1,)
            imageShapeOrder = (0,)

            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'feature parameters': {
                    'featureMethod': featureMethod,
                    'windowTimeLength': windowTimeLength,
                    'poolingSize': poolingSize,
                    'poolingType': poolingType,
                    'maxTime': maxTime,
                },
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == 'MNIST':
            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 28, 28)
            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == "THoR":
            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 10000, 1)
            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }
        elif featureMethod == "Test":
            imageShapeOrder = (0, 1, 2)
            imageShape = (1, 1, 2)
            featureParametersDefault = {
                'featureSetName': featureSetName,
                'signalSource': signalSource,
                'rawDataFolder': rawDataFolder,
                'featureDataFolder': featureDataFolder,
                'imageShape': imageShape,
                'imageShapeOrder': imageShapeOrder,
            }

        ################################
        ## Parameters End   ############
        ################################

        if not os.path.exists(featureDataFolder):
            os.makedirs(featureDataFolder)
        doesFileAlreadyExist = os.path.exists(configFileName)
        if not overwriteConfigFile:
            assert not doesFileAlreadyExist, 'do you want to overwirte the config file?'
        with open(configFileName, 'w') as myConfigFile:
            yaml.dump(featureParametersDefault, myConfigFile, default_flow_style=False, width=1000)
            if doesFileAlreadyExist:
                print("Overwrote Feature config file {0}".format(configFileName))
            else:
                print("Wrote Feature config file {0}".format(configFileName))
        featureSets = [featureSetName]
    else:
        if rebuildAllFromConfig:
            featureDataFolderRoot = os.path.join(rawDataFolder, "Processed Data Features")
            featureSets = [fileIterator for fileIterator in os.listdir(featureDataFolderRoot) if
                           os.path.isdir(os.path.join(featureDataFolderRoot,
                                                      fileIterator)) and fileIterator not in removeFeatureSetNames]
        elif rebuildFromConfig:
            featureSets = [featureSetName]
        else:
            featureSets = []

    if runNow:
        for featureSetFolder in featureSets:
            featureConfigFileName = os.path.join(rawDataFolder, "Processed Data Features", featureSetFolder,
                                                 "feature parameters.yaml")
            with open(featureConfigFileName, 'r') as myConfigFile:
                featureParametersDefault = yaml.load(myConfigFile)
            # didProcessAllFiles = False
            # while not didProcessAllFiles:
            didProcessAllFiles = readwav.buildFeatures(featureParametersDefault,
                                                       forceRefreshFeatures=forceRefreshFeatures, maxFiles=maxFiles,
                                                       removeFileNumbers=removeFileNumbers,
                                                       onlyFileNumbers=onlyFileNumbers)
