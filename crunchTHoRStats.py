import numpy
import os
import pandas as pd
import scipy

if os.name == 'nt':
    rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\parse NHL\\"
elif os.name == 'posix':
    rawDataFolder = r"/media/sena/Greed Island/Users/Joey/Documents/Python Scripts/parse NHL/"
else:
    raise ValueError("This OS is not allowed")

datasetName = "THoR"
featureSetName = 'DefaultTHoRFeatures'
whichSet = 0

allStats = numpy.genfromtxt(os.path.join(rawDataFolder,'allStats.csv'), delimiter=',', usecols = [13], dtype='a')
if len(allStats.shape) <= 1:
    allStats = numpy.expand_dims(allStats,1)

processedDataFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)
datasetFile = os.path.join(processedDataFolder, featureSetName + '.hf')
with pd.HDFStore(datasetFile, 'r') as featureStore:
    p_train = featureStore['p_train'.format(whichSet)].as_matrix()
    p_valid = featureStore['p_valid'.format(whichSet)].as_matrix()
    p_test = featureStore['p_test'.format(whichSet)].as_matrix()
    columnNames = numpy.array(featureStore['columnNames'.format(whichSet)].as_matrix(), dtype='a').squeeze()

experimentStore = os.path.join(rawDataFolder, r"Data Experiments\\DefaultTHoRFeatures\\THoR\\MLP\\RegressionAllClassesDefault\\")
gradFile = os.path.join(experimentStore, 'gradArray.hf')
with pd.HDFStore(gradFile, 'r') as featureStore:
    gradArray = featureStore['gradArray{0}'.format(whichSet)].as_matrix()

if whichSet == 0:
    p = p_train
elif whichSet == 1:
    p = p_valid
else:
    p = p_test

allStatsThisSet = allStats[p,:]

playTypes = ["FAC", "BLOCK", "MISS", "SHOT", "HIT", "GIVE", "TAKE", "PENL", "GOAL"]
# longestNameLength = len(max(columnNames, key=len))
# finalCsvDtype = [(playType, float) for playType in playTypes]
# finalCsvDtype = [("players", 'S51')] + finalCsvDtype
# fmtArg = ['%.18e' for playType in playTypes]
# fmtArg = ['%s'] + fmtArg
finalCsvArray = numpy.zeros((gradArray.shape[1], len(playTypes)+1), dtype='S51')
columnNames = numpy.core.defchararray.replace(columnNames,',','_')
finalCsvArray[:,0] = columnNames
counter = 1
for playType in playTypes:
    playTypeMask = numpy.squeeze(allStatsThisSet[:, 0] == playType)
    playTypeValsPerPlayer = numpy.sum(gradArray[playTypeMask, :], axis=0)
    finalCsvArray[:,counter] = playTypeValsPerPlayer
    counter+=1

finalCsvFile = os.path.join(rawDataFolder,'gradStats.csv')

headerString = ','.join(['Player Name'] +playTypes)

numpy.savetxt(fname=finalCsvFile, X=finalCsvArray, delimiter=',', header=headerString, footer= '', comments='', fmt='%s')