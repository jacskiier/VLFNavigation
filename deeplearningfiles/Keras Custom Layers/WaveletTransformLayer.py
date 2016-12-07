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
    def __init__(self,
                 output_dim=None,
                 maxWindowSize=None,
                 kValues=None,
                 sigmaValues=None,
                 minSigmasPerWindow=None,
                 minCyclesPerOneSigma=None,
                 useConvolution=False,
                 **kwargs):
        """
        :param output_dim: number of Frequency-Sigma value banks per input channel
        :param maxWindowSize: size of filter window (if useConvolution=False must match input_dim)
        :param kValues: values of normalized frequency (must have one value for each input channel bank combination)
        :param sigmaValues: values of window drop off as gaussian variable sigma (must have one value for each input channel bank combination)
        :param minSigmasPerWindow: minimum sigmas in one filter window size
        :param minCyclesPerOneSigma: minimum cycles to fit inside a one sigma for any frequency
        :param useConvolution: if True will shift a sliding filter window for all input samples, if false does one window per timestep
        :param kwargs: extra layer args for Keras
        """
        assert output_dim is not None, "There must be an output_dim"
        assert maxWindowSize is not None, "There must be a window size"
        self.output_dim = output_dim
        self.input_dim = None
        self.timesteps = None
        self.input_channels = None
        self.batch_size = None
        self.useConvolution = useConvolution

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

    def get_output_shape_for(self, input_shape):
        # input_shape = (bs x timesteps x input_channels x input_dim)

        ret = input_shape[0], input_shape[1], input_shape[2] * self.output_dim
        # bs x timesteps x output_channels * output_dim
        return ret

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'timesteps': self.timesteps,
                  'input_channels': self.input_channels,
                  'batch_size': self.batch_size,
                  'useConvolution': self.useConvolution,
                  'kValues': self.initDict['kValues'],
                  'sigmaValues': self.initDict['sigmaValues'],
                  'minSigmasPerWindow': self.initDict['minSigmasPerWindow'],
                  'minCyclesPerOneSigma': self.initDict['minCyclesPerOneSigma'],
                  'input_dim': self.input_dim}
        base_config = super(WaveletTransformLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # input_shape = (bs x timesteps x input_channels x input_dim)
        self.input_dim = input_shape[3]
        self.input_channels = input_shape[2]
        self.timesteps = input_shape[1]
        self.batch_size = input_shape[0]

        if self.useConvolution is False:
            errorString = "If not using convolution input_dim ({0}) and window size ({1}) must be equal".format(self.input_dim, self.maxWindowSize)
            assert self.input_dim == self.maxWindowSize, errorString

        sigma_w = 8 if self.initDict['minSigmasPerWindow'] is None else self.initDict['minSigmasPerWindow']
        c_sigma = 3 / 8.0 if self.initDict['minSigmasPerWindow'] is None else 3 / float(self.initDict['minSigmasPerWindow'])
        assert self.maxWindowSize > 2 * (sigma_w * c_sigma), "minSigmasPerWindow and minSigmasPerWindow are too big for given maxWindowSize"
        defaultKs = np.random.randint(sigma_w * c_sigma, self.maxWindowSize - (sigma_w * c_sigma),
                                      (self.output_dim * self.input_channels,)).astype(theano.config.floatX)
        defaultSigmas = np.zeros((self.output_dim * self.input_channels,))
        for index in range(defaultSigmas.shape[0]):
            defaultSigmas[index] = np.random.randint(self.maxWindowSize * c_sigma / defaultKs[index],
                                                     int(self.maxWindowSize / float(sigma_w)), (1,)).astype(
                theano.config.floatX)
        defaultInitDict = {
            'kValues': defaultKs,
            'sigmaValues': defaultSigmas,
            'minSigmasPerWindow': 8,
            'minCyclesPerOneSigma': 3 / 8.0 if self.initDict['minSigmasPerWindow'] is None else 3 / float(self.initDict['minSigmasPerWindow']),
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
        self.non_trainable_weights.append(self.minSigmasPerWindow)

    def call(self, inputSignal, mask=None):
        # inputSignal = (bs x timesteps x input_channels x input_dim)

        kValuesT = self.kValues  # (output_dim * input_channels, )
        sigmaValuesT = self.sigmaValues  # (output_dim * input_channels, )

        if self.useConvolution:
            inputSignalReshaped = T.swapaxes(inputSignal, 1, 2)
            # (bs x  input_channels x timesteps x input_dim)

            isrShape = inputSignalReshaped.shape
            inputSignalReshaped = T.reshape(inputSignalReshaped, (isrShape[0], isrShape[1], isrShape[2] * isrShape[3], 1))
            # (bs x  input_channels x timesteps*input_dim, 1)

            inputSignalReshaped = T.swapaxes(inputSignalReshaped, 2, 3)
            # (bs x  input_channels x 1 x input_dim*timesteps)

            inputSignalReshaped.name = 'inputSignalReshaped'

            batch_size = self.batch_size
            input_channels = self.input_channels
            input_rows = 1  # I force reshape this to 1 by dumping input dim into the timestep dim
            input_columns = self.timesteps * self.input_dim
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
            # (bs x output_channels x output_dim x timesteps*input_dim + maxWindowSize -1)
            powerOut = powerOut[:, :, :, self.maxWindowSize / 2 - 1: - self.maxWindowSize / 2]
            # (bs x output_channels x output_dim x timesteps*input_dim)
            powerOut = T.reshape(powerOut, (self.batch_size, output_channels, self.output_dim, self.timesteps, self.input_dim))
            # (bs x output_channels x output_dim x timesteps x input_dim)
            powerOut = T.mean(powerOut, axis=4)
            # (bs x output_channels x output_dim x timesteps)
            powerOut = T.swapaxes(powerOut, 1, 3)
            # (bs x timesteps x output_dim x output_channels)
            powerOut = T.swapaxes(powerOut, 2, 3)
            # (bs x timesteps x output_channels x output_dim)
            powerOut = T.reshape(powerOut, (powerOut.shape[0], powerOut.shape[1], powerOut.shape[2] * powerOut.shape[3]))
            # (bs x timesteps x (output_channels * output_dim))
        else:
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

            # input_channel*filters x window_size
            filterBankRealFinal = T.reshape(filterBankReal, (self.input_channels, self.output_dim, self.maxWindowSize))
            filterBankImagFinal = T.reshape(filterBankImag, (self.input_channels, self.output_dim, self.maxWindowSize))
            # input_channel x filters x window_size

            filterBankRealFinal.name = 'filterBankRealFinal'
            filterBankImagFinal.name = 'filterBankImagFinal'

            filterBankRealFinal = T.swapaxes(filterBankRealFinal, 0, 1)
            filterBankImagFinal = T.swapaxes(filterBankImagFinal, 0, 1)
            # filters x input_channel x window_size

            inputSignalReshaped = inputSignal
            # (bs x timesteps x input_channels x input_dim)

            inputSignalReshaped = inputSignalReshaped[:, :, None, :, :]
            # (bs x timesteps x 1 x input_channels x input_dim)
            outputReal = inputSignalReshaped * filterBankRealFinal
            outputImag = inputSignalReshaped * filterBankImagFinal
            # bs x timesteps x filters x input_channels x input_dim
            # = (bs x timesteps x 1 x input_channels x input_dim) eleMult (filters x input_channels x window_size)

            outputReal = T.sum(outputReal, axis=4)
            outputImag = T.sum(outputImag, axis=4)
            # bs x timesteps x output_dim x output_channels

            powerOut = T.power(outputReal, 2) + T.power(outputImag, 2)
            # bs x timesteps x output_dim x output_channels
            powerOut = T.swapaxes(powerOut, 2, 3)
            # bs x timesteps x output_channels x output_dim
            powerOut = T.reshape(powerOut, (self.batch_size, self.timesteps, self.input_channels * self.output_dim))
            # (bs x timesteps x (output_channels*output_dim))
        return powerOut


def testNoConvolution():
    # test with no convolution
    plt.close('all')
    signalLength = 44100 * 60 * 30
    input_channelsMain = 3
    maxWindowSizeMain = 44100
    minSigmasPerWindowMain = 8
    minCyclesPerOneSigmaMain = 3
    sigmaMax = maxWindowSizeMain / minSigmasPerWindowMain
    smallestK = minCyclesPerOneSigmaMain * minSigmasPerWindowMain
    input_dim = 44100
    batch_sizeMain = 1

    # signal for everyone
    dt = 1 / 44100.0  # second/sample
    samplingRate = 1 / dt  # samples / second
    nyquistFreq = samplingRate / 2.0
    start = 0
    end = signalLength * dt
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

    print("Sampling Rate {samplingRate} Hz".format(samplingRate=samplingRate))
    print("Nyquist freq is {nyqist} Hz".format(nyqist=nyquistFreq))
    print("Sigma Max {sigmaMax} samples. Smallest K {smallestK} normFreq".format(sigmaMax=sigmaMax, smallestK=smallestK))
    print("Sigma Max {sigma} seconds. Smallest Freq {freq} Hz".format(sigma=sigmaMax * dt, freq=smallestK / float(maxWindowSizeMain) * samplingRate))
    secondsPerWindow = dt * maxWindowSizeMain
    print("Seconds per window {secondsPerWindow}".format(secondsPerWindow=secondsPerWindow))

    # input_channel x filters
    sigmaValuesMain = (np.array([sigmaMax, sigmaMax / 2.0] * input_channelsMain)).astype(theano.config.floatX)
    kValuesMain = (np.array([smallestK, smallestK * 2.0] * input_channelsMain)).astype(theano.config.floatX)
    bank_size = kValuesMain.shape[0] / input_channelsMain

    useConvolution = False
    signal = np.reshape(signal, (signal.shape[0] / input_dim, input_dim))
    assert maxWindowSizeMain == input_dim, "the window size and input_dim must be equal if not doing conv"

    signal = np.repeat(signal[None, :], batch_sizeMain, 0)
    signal = np.repeat(signal[:, :, None], input_channelsMain, 2)
    # inputSignal = (bs x timesteps x input_channels x input_dim)

    waveletLayer = WaveletTransformLayer(bank_size, maxWindowSizeMain,
                                         kValues=kValuesMain,
                                         sigmaValues=sigmaValuesMain,
                                         minCyclesPerOneSigma=minCyclesPerOneSigmaMain,
                                         minSigmasPerWindow=minSigmasPerWindowMain,
                                         useConvolution=useConvolution,
                                         name="Wavelet Layer")
    waveletLayer.build((batch_sizeMain, signal.shape[1], input_channelsMain, input_dim))
    inputSignalT = T.tensor4("inputSignal", dtype=theano.config.floatX)
    powerOutT = waveletLayer.call(inputSignalT, None)

    waveletTransform = theano.function(inputs=[inputSignalT], outputs=[powerOutT])

    powerOutMain = waveletTransform(signal)[0]

    powerOutMain = np.reshape(powerOutMain,
                              newshape=(batch_sizeMain, signal.shape[1], input_channelsMain, bank_size))
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


def testConvolution():
    # test with convolution
    plt.close('all')
    signalLength = 44100 * 5
    input_channelsMain = 3
    maxWindowSizeMain = 10000
    minSigmasPerWindowMain = 8
    minCyclesPerOneSigmaMain = 3
    sigmaMax = maxWindowSizeMain / minSigmasPerWindowMain
    smallestK = minCyclesPerOneSigmaMain * minSigmasPerWindowMain
    input_dim = 1
    batch_sizeMain = 1

    # signal for everyone
    dt = 1 / 44100.0  # second/sample
    samplingRate = 1 / dt  # samples / second
    nyquistFreq = samplingRate / 2.0
    start = 0
    end = signalLength * dt
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

    print("Sampling Rate {samplingRate} Hz".format(samplingRate=samplingRate))
    print("Nyquist freq is {nyqist} Hz".format(nyqist=nyquistFreq))
    print("Sigma Max {sigmaMax} samples. Smallest K {smallestK} normFreq".format(sigmaMax=sigmaMax, smallestK=smallestK))
    print("Sigma Max {sigma} seconds. Smallest Freq {freq} Hz".format(sigma=sigmaMax * dt, freq=smallestK / float(maxWindowSizeMain) * samplingRate))
    secondsPerWindow = dt * maxWindowSizeMain
    print("Seconds per window {secondsPerWindow}".format(secondsPerWindow=secondsPerWindow))

    # input_channel x filters
    sigmaValuesMain = (np.array([sigmaMax, sigmaMax / 2.0] * input_channelsMain)).astype(theano.config.floatX)
    kValuesMain = (np.array([smallestK, smallestK * 2.0] * input_channelsMain)).astype(theano.config.floatX)
    bank_size = kValuesMain.shape[0] / input_channelsMain

    print ("bank size {0}".format(bank_size))
    useConvolution = True
    signal = signal[:, None]
    signal = np.repeat(signal[None, :], batch_sizeMain, 0)
    signal = np.repeat(signal[:, :, None], input_channelsMain, 2)
    # inputSignal = (bs x timesteps x input_channels x input_dim)

    waveletLayer = WaveletTransformLayer(bank_size, maxWindowSizeMain,
                                         kValues=kValuesMain,
                                         sigmaValues=sigmaValuesMain,
                                         minCyclesPerOneSigma=minCyclesPerOneSigmaMain,
                                         minSigmasPerWindow=minSigmasPerWindowMain,
                                         useConvolution=useConvolution,
                                         name="Wavelet Layer")
    waveletLayer.build((batch_sizeMain, signal.shape[1], input_channelsMain, input_dim))
    inputSignalT = T.tensor4("inputSignal", dtype=theano.config.floatX)
    powerOutT = waveletLayer.call(inputSignalT, None)

    waveletTransform = theano.function(inputs=[inputSignalT], outputs=[powerOutT])

    powerOutMain = waveletTransform(signal)[0]

    powerOutMain = np.reshape(powerOutMain,
                              newshape=(batch_sizeMain, signal.shape[1], input_channelsMain, bank_size))
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


if __name__ == '__main__':
    # testNoConvolution()
    testConvolution()
