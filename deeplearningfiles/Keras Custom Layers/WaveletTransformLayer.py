# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:31:06 2016

@author: rhefron and jcurro
"""

import matplotlib.pylab as plt
import numpy as np

import keras.backend as K
from keras.engine.topology import Layer

import theano
import theano.tensor as T


class WaveletTransformLayer(Layer):
    def __init__(self, output_dim, maxWindowSize, kValues=None, sigmaValues=None, minSigmasPerWindow=None,
                 minCyclesPerOneSigma=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = None
        self.timesteps = None
        self.input_channels = None
        self.batch_size = None

        self.initDict = {
            'kValues': kValues,
            'sigmaValues': sigmaValues,
            'minSigmasPerWindow': minSigmasPerWindow,
            'minCyclesPerOneSigma': minCyclesPerOneSigma,
        }
        self.weightsDict = {}
        self.kValues = None
        self.sigmaValues = None
        self.maxWindowSize = maxWindowSize
        self.minSigmasPerWindow = None

        super(WaveletTransformLayer, self).__init__(**kwargs)

    def doSizeAsserts(self, weightName, initValue):
        if weightName == 'kValues':
            assert initValue.size == self.input_channels * self.output_dim, \
                "You don't have enough k values for the banks and input channels"
            if len(initValue.shape) > 1:
                assert initValue.shape[0] == 1 or initValue.shape[1] == 1, "kValues must be 1D"
                assert initValue.shape[0] == self.output_dim * self.input_channels or \
                       initValue.shape[1] == self.output_dim * self.input_channels, "kValues bad size"
            else:
                assert initValue.shape[0] == self.output_dim * self.input_channels, "kValues Bad size"
        if weightName == 'sigmaValues':
            assert initValue.size == self.input_channels * self.output_dim, \
                "You don't have enough sigma values for the banks and input channels"
            if len(initValue.shape) > 1:
                assert initValue.shape[0] == 1 or initValue.shape[1] == 1, "sigma Values must be 1D"
                assert initValue.shape[0] == self.output_dim * self.input_channels or \
                       initValue.shape[1] == self.output_dim * self.input_channels, "sigmaValues bad size"
            else:
                assert initValue.shape[0] == self.output_dim * self.input_channels, "sigmaValues Bad size"

    def build(self, input_shape):
        # input_shape = (bs x timesteps x input_channels x input_dim)
        self.input_dim = input_shape[3]
        self.input_channels = input_shape[2]
        self.timesteps = input_shape[1]
        self.batch_size = input_shape[0]

        sigma_w = 8 if self.initDict['minSigmasPerWindow'] is None else self.initDict['minSigmasPerWindow']
        c_sigma = 3 / 8.0 if self.initDict['minSigmasPerWindow'] is None else 3 / float(
            self.initDict['minSigmasPerWindow'])
        defaultKs = np.random.randint(sigma_w * c_sigma, self.maxWindowSize - sigma_w * c_sigma,
                                      (self.output_dim,)).astype(theano.config.floatX)
        defaultSigmas = np.zeros((self.output_dim,))
        for index in range(defaultSigmas.shape[0]):
            defaultSigmas[index] = np.random.randint(self.maxWindowSize * c_sigma / defaultKs[index],
                                                     int(self.maxWindowSize / float(sigma_w)), (1,)).astype(
                theano.config.floatX)
        defaultInitDict = {
            'kValues': defaultKs,
            'sigmaValues': defaultSigmas,
            'minSigmasPerWindow': 8,
            'minCyclesPerOneSigma': 3 / 8.0 if self.initDict['minSigmasPerWindow'] is None else 3 / float(
                self.initDict['minSigmasPerWindow']),
        }

        for (weightName, initValue) in self.initDict.iteritems():
            if initValue is None:
                finalInit = defaultInitDict[weightName]
            else:
                finalInit = initValue
            self.doSizeAsserts(weightName, finalInit)
            self.weightsDict[weightName] = K.variable(finalInit, name=weightName)

        self.kValues = self.weightsDict['kValues']
        self.sigmaValues = self.weightsDict['sigmaValues']

        self.minSigmasPerWindow = self.weightsDict['minSigmasPerWindow']
        self.trainable_weights.append(self.kValues)
        self.trainable_weights.append(self.sigmaValues)
        self.non_trainable_weights.append(self.maxWindowSize)
        self.non_trainable_weights.append(self.minSigmasPerWindow)

    def call(self, inputSignal, mask=None):
        # inputSignal = (bs x timesteps x input_channels x input_dim)

        inputSignalReshaped = T.swapaxes(inputSignal, 1, 2)
        # (bs x  input_channels x timesteps x input_dim)

        isrShape = inputSignalReshaped.shape
        inputSignalReshaped = T.reshape(inputSignalReshaped, (isrShape[0], isrShape[1], isrShape[2] * isrShape[3], 1))
        # (bs x  input_channels x timesteps*input_dim, 1)

        inputSignalReshaped = T.swapaxes(inputSignalReshaped, 2, 3)
        # (bs x  input_channels x 1 x input_dim*timesteps)

        inputSignalReshaped.name = 'inputSignalReshaped'
        kValuesT = self.kValues  # (output_dim, )
        sigmaValuesT = self.sigmaValues  # (output_dim, )

        batch_size = self.batch_size
        input_channels = self.input_channels
        input_rows = 1 # I force reshape this to 1 by dumping input dim into the timestep dim
        input_columns = self.timesteps
        input_shape = (batch_size, input_channels, input_rows, input_columns)
        output_channels = self.input_channels
        filter_rows = self.output_dim  # we will have a row for each filter
        filter_columns = self.maxWindowSize
        filter_shape = (output_channels, input_channels, filter_rows, filter_columns)

        filterBankRealFinal = T.zeros(filter_shape, dtype=theano.config.floatX)
        filterBankImagFinal = T.zeros(filter_shape, dtype=theano.config.floatX)

        filterColumnsIndexShift = T.arange(0, self.maxWindowSize)
        # fft coefficients
        coefficientsReal = T.cos(
            -2 * np.pi * T.outer(kValuesT, filterColumnsIndexShift) / filterColumnsIndexShift.shape[0])
        coefficientsImag = T.sin(
            -2 * np.pi * T.outer(kValuesT, filterColumnsIndexShift) / filterColumnsIndexShift.shape[0])
        # gaussian window
        mu = T.cast(filterColumnsIndexShift[filterColumnsIndexShift.shape[0] / 2], dtype=theano.config.floatX)
        gaussianWindows = 1 / T.sqrt(2 * sigmaValuesT[:, None] ** 2 * np.pi) * T.exp(
            T.outer(1 / (2 * sigmaValuesT ** 2), - T.power((filterColumnsIndexShift - mu), 2)))
        # multiply the banks out
        filterBankReal = T.cast(coefficientsReal * gaussianWindows, theano.config.floatX)
        filterBankImag = T.cast(coefficientsImag * gaussianWindows, theano.config.floatX)

        # If I wasn't looping uncomment these 2 lines
        # filterBankRealFinal = filterBankReal[None, None, :, :]
        # filterBankImagFinal = filterBankImag[None, None, :, :]

        processedRows = 0
        for input_channel in range(input_channels):
            for output_channel in range(output_channels):
                if input_channel == output_channel:
                    filterBankRealFinal = T.set_subtensor(filterBankRealFinal[output_channel, input_channel, :, :],
                                                          filterBankReal[processedRows:processedRows + self.output_dim,
                                                          :])
                    filterBankImagFinal = T.set_subtensor(filterBankImagFinal[output_channel, input_channel, :, :],
                                                          filterBankImag[processedRows:processedRows + self.output_dim,
                                                          :])
                    processedRows += self.output_dim
        filterBankRealFinal.name = 'filterBankRealFinal'
        filterBankImagFinal.name = 'filterBankImagFinal'
        outputConvReal = theano.tensor.nnet.conv2d(input=inputSignalReshaped,
                                                   filters=filterBankRealFinal,
                                                   input_shape=input_shape,
                                                   filter_shape=filter_shape,
                                                   border_mode='full',
                                                   filter_flip=False)

        outputConvImag = theano.tensor.nnet.conv2d(input=inputSignalReshaped,
                                                   filters=filterBankImagFinal,
                                                   input_shape=input_shape,
                                                   filter_shape=filter_shape,
                                                   border_mode='full',
                                                   filter_flip=False)
        powerOut = T.power(outputConvReal, 2) + T.power(outputConvImag, 2)
        # (bs x output_channels x output_dim x timesteps + maxWindowSize -1)
        powerOut = powerOut[:, :, :, self.maxWindowSize / 2 - 1: - self.maxWindowSize / 2]
        # (bs x output_channels x output_dim x timesteps)
        powerOut = T.swapaxes(powerOut, 1, 3)
        # (bs x timesteps x output_dim x output_channels)
        powerOut = T.swapaxes(powerOut, 2, 3)
        # (bs x timesteps x output_channels x output_dim)
        powerOut = T.reshape(powerOut, (powerOut.shape[0], powerOut.shape[1], powerOut.shape[2] * powerOut.shape[3]))
        # (bs x timesteps x (output_channels * output_dim))

        return powerOut


if __name__ == '__main__':
    plt.close('all')
    signalLength = 44000
    input_channelsMain = 3
    maxWindowSizeMain = 1000
    minSigmasPerWindowMain = 8
    minCyclesPerOneSigmaMain = 3
    sigmaMax = maxWindowSizeMain / minSigmasPerWindowMain
    smallestK = minCyclesPerOneSigmaMain * minSigmasPerWindowMain
    input_dim = 1
    batch_sizeMain = 1

    # signal for everyone
    dt = 1 / 44000.0
    start = 0
    end = signalLength * dt
    print ("nyquist freq is {nyqist}".format(nyqist=1 / (2 * dt)))
    signalFreq = 100.0
    signalFreq2 = smallestK / float(maxWindowSizeMain * dt)
    x = np.arange(start, end, dt, dtype=theano.config.floatX)
    signal = np.sin(signalFreq * 2 * np.pi * x)
    signal2 = np.sin(signalFreq2 * 2 * np.pi * x)
    signal[signalLength / 2:] = signal2[signalLength / 2:]
    signal = signal.astype(theano.config.floatX)
    plt.figure(1)
    plt.plot(x, signal)
    plt.title("Signal")

    sigmaValuesMain = (np.array([sigmaMax, sigmaMax / 2.0] * input_channelsMain)).astype(theano.config.floatX)
    kValuesMain = (np.array([smallestK, smallestK * 2.0] * input_channelsMain)).astype(theano.config.floatX)
    bank_size = kValuesMain.shape[0] / input_channelsMain

    waveletLayer = WaveletTransformLayer(bank_size, maxWindowSizeMain,
                                         kValues=kValuesMain,
                                         sigmaValues=sigmaValuesMain,
                                         minCyclesPerOneSigma=minCyclesPerOneSigmaMain,
                                         minSigmasPerWindow=minSigmasPerWindowMain,
                                         name="Wavelet Layer")
    waveletLayer.build((batch_sizeMain, signalLength, input_channelsMain, input_dim))
    inputSignalT = T.tensor4("inputSignal", dtype=theano.config.floatX)
    powerOutT = waveletLayer.call(inputSignalT, None)

    waveletTransform = theano.function(inputs=[inputSignalT], outputs=[powerOutT])

    signal = np.repeat(signal[None, :], batch_sizeMain, 0)
    signal = np.repeat(signal[:, :, None], input_channelsMain, 2)
    signal = signal[:, :, :, None]
    powerOutMain = waveletTransform(signal)[0]

    powerOutMain = np.reshape(powerOutMain,
                              newshape=(batch_sizeMain, signalLength, input_channelsMain, bank_size))
    maxBatchesToShow = 9
    plotsX = np.ceil(np.sqrt(min(bank_size, maxBatchesToShow)))
    plotsY = plotsX
    totalPlots = min(bank_size, maxBatchesToShow)
    print(totalPlots)
    batchToShow = 0
    outputChannelToShow = 0
    for output_dimToShow in range(totalPlots):
        plt.figure(4)
        plt.subplot(plotsX, plotsY, output_dimToShow + 1)
        plt.semilogy(powerOutMain[batchToShow, :, outputChannelToShow, output_dimToShow])
        plt.ylim([np.min(powerOutMain), np.max(powerOutMain)])
    plt.show()
