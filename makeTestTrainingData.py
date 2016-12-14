# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:14:05 2015

@author: jacsk
"""
import numpy as np
import pandas as pd
import pandas.tools.plotting
import matplotlib.pylab as plt
import os
import CreateDataset
import CreateUtils

numOfFeatures = 3
numberOfSamples = 7000
outputs = 1
samplesPerOutput = numberOfSamples / outputs
runType = 'regression'
featureMethod = "Test"

datasetName = "Test"

rootDataFolder = CreateUtils.getRootDataFolder(featureMethod=featureMethod)
rawDataFolder = CreateUtils.getRawDataFolder()

fX = np.array([])
fY = np.array([])

fx = np.ones((samplesPerOutput, numOfFeatures))

np.random.seed(10)
F = np.random.randint(1, 10, (numOfFeatures, outputs))
print (F)
for x in range(outputs):
    if runType == 'classifier':
        fx = np.zeros((samplesPerOutput, numOfFeatures))
        fx[:, (x / float(outputs)) * numOfFeatures:((x + 1) / float(outputs)) * numOfFeatures] = np.expand_dims(
            np.arange(numOfFeatures / outputs) * (x + 1), 0)
        fx[:, (x / float(outputs)) * numOfFeatures:((x + 1) / float(outputs)) * numOfFeatures] = np.expand_dims(np.arange(numOfFeatures / outputs), 0)

        fy = np.ones((samplesPerOutput,)) * x
    elif runType == 'regression':
        fx = np.random.rand(samplesPerOutput, numOfFeatures) * 10.0

        fy = np.dot(fx, F) + np.random.randn(samplesPerOutput, outputs) * 2.0

    if fX.shape == (0L,):
        fX = fx
    else:
        fX = np.vstack((fX, fx))

    if fY.shape == (0L,):
        fY = fy
    else:
        fY = np.vstack((fY, fy))

(fX, fY) = CreateDataset.shuffle_in_unison_inplace(fX, fY)

numberOfSamples = fX.shape[0]
train_set = np.array([])
valid_set = np.array([])
test_set = np.array([])
if numberOfSamples > 0:
    trainSetEnd = int(numberOfSamples * (5.0 / 7.0))
    validSetEnd = trainSetEnd + int(numberOfSamples * (1.0 / 7.0))
    train_set = (fX[:trainSetEnd, :], fY[:trainSetEnd])
    valid_set = (fX[trainSetEnd:validSetEnd, :], fY[trainSetEnd:validSetEnd])
    test_set = (fX[validSetEnd:-1, :], fY[validSetEnd:-1])

outputLabelsFinal = [str(a) for a in range(outputs)]
datasetFile = CreateUtils.getDatasetFile(featureSetName="Test", datasetName=datasetName)
with pd.HDFStore(datasetFile, 'a') as datasetStore:
    datasetStore['train_set_x'] = pd.DataFrame(train_set[0])
    datasetStore['valid_set_x'] = pd.DataFrame(valid_set[0])
    datasetStore['test_set_x'] = pd.DataFrame(test_set[0])
    datasetStore['train_set_y'] = pd.DataFrame(train_set[1])
    datasetStore['valid_set_y'] = pd.DataFrame(valid_set[1])
    datasetStore['test_set_y'] = pd.DataFrame(test_set[1])
    datasetStore['labels'] = pd.DataFrame(outputLabelsFinal)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(fX[:,0],fX[:,1],fY[:,0], label = 'some data')
# plt.show()

df = pd.DataFrame(train_set[0])
df2 = pd.DataFrame(train_set[1])
df3 = pd.concat([df, df2], axis=1)
df3.columns = ["x" + str(x) for x in range(numOfFeatures)] + ["y" + str(y) for y in range(outputs)]
print(df3)
axes = pd.tools.plotting.scatter_matrix(df3, alpha=0.2)
plt.tight_layout()
plt.show()
