import numpy
import os
import pandas as pd
from readwav import shuffle_in_unison_inplaceWithP

rawDataFolder = r"E:\\Users\\Joey\\Documents\\Python Scripts\\parse NHL\\"

xAll = numpy.genfromtxt(os.path.join(rawDataFolder ,'xAll.csv'), dtype=int, delimiter=',')
y0 = numpy.genfromtxt(os.path.join(rawDataFolder ,'Y0.csv'), dtype=float, delimiter=',')
y1 = numpy.genfromtxt(os.path.join(rawDataFolder,'Y1.csv'), dtype=float, delimiter=',')
columnNames = numpy.genfromtxt(os.path.join(rawDataFolder,'playersAll.csv'), dtype='a', delimiter='\t')

fX = xAll
usey0 = True
if usey0:
    fY = y0
else:
    fY = y1

numpy.random.seed(1)
(fX, fY, p) = shuffle_in_unison_inplaceWithP(fX, fY)

fY = numpy.expand_dims(fY ,1)
numberOfSamples = y0.shape[0]
trainSetEnd = int(numberOfSamples * (5.0 / 7.0))
validSetEnd = trainSetEnd + int(numberOfSamples * (1.0 / 7.0))
train_set = (fX[:trainSetEnd, :], fY[:trainSetEnd ,:])
valid_set = (fX[trainSetEnd:validSetEnd, :], fY[trainSetEnd:validSetEnd ,:])
test_set = (fX[validSetEnd:-1, :], fY[validSetEnd:-1 ,:])

p = numpy.array(p)
p_train = p[:trainSetEnd]
p_valid = p[trainSetEnd:validSetEnd]
p_test = p[validSetEnd:-1]

datasetName = "THoR"
featureSetName = 'DefaultTHoRFeatures'

processedDataFolder = os.path.join(rawDataFolder, "Processed Data Datasets", datasetName)
datasetFile = os.path.join(processedDataFolder, featureSetName + '.hf')
with pd.HDFStore(datasetFile, 'a') as datasetStore:
    datasetStore['train_set_x'] = pd.DataFrame(train_set[0])
    datasetStore['valid_set_x'] = pd.DataFrame(valid_set[0])
    datasetStore['test_set_x']  = pd.DataFrame(test_set[0])
    datasetStore['train_set_y'] = pd.DataFrame(train_set[1])
    datasetStore['valid_set_y'] = pd.DataFrame(valid_set[1])
    datasetStore['test_set_y'] = pd.DataFrame(test_set[1])
    datasetStore['labels'] = pd.DataFrame(numpy.array([]))
    datasetStore['columnNames'] = pd.DataFrame(columnNames)
    datasetStore['p'] = pd.DataFrame(p)
    datasetStore['p_train'] = pd.DataFrame(p_train)
    datasetStore['p_valid'] = pd.DataFrame(p_valid)
    datasetStore['p_test'] = pd.DataFrame(p_test)