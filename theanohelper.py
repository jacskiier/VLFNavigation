from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.optimizers import SGD

import theano
import theano.tensor as T
import numpy as np
import pylab as plt

doTheanoShiz = False
doMatrixMultiplyFun = False
doScanExample = False
doScanPinv = False
doHardSigmoid = False
doResidualStuff = False
doSemiPositiveStuff = False
do2DConvStuff = True
doZipshiz = False

if doTheanoShiz:
    m = T.matrix(dtype=theano.config.floatX)
    m_normalized = m / m.sum(axis=1).reshape((m.shape[0], 1))
    m_normer = m.norm(1, axis=1)

    f = theano.function([m], m_normalized)
    f2 = theano.function([m], m_normer)
    a = np.exp(np.random.randn(10, 2)).astype(theano.config.floatX) - 5
    b = f(a)
    c = a / a.sum(axis=1)[:, np.newaxis]
    d = f2(a)

    a2 = np.exp(np.random.randn(20, 10)).astype(theano.config.floatX) - 5
    c2 = np.exp(np.random.randn(20, 1)).astype(theano.config.floatX) - 5

    mat = T.matrix(dtype=theano.config.floatX)
    vec = T.matrix(dtype=theano.config.floatX)
    # matInv = T.nlinalg.matrix_inverse(mat)
    matInv = T.nlinalg.pinv(mat)
    d2 = T.dot(matInv, vec)
    e3 = T.sum(d2)
    g_W = T.grad(cost=e3, wrt=vec)

    f3 = theano.function(inputs=[mat, vec], outputs=g_W)
    f4 = theano.function(inputs=[mat, vec], outputs=d2)
    b2 = f3(a2, c2)
    b3 = f4(a2, c2)
    from numpy.testing import assert_array_equal

    # assert_array_equal(b, d)
    print b2
    print b3

if doMatrixMultiplyFun:
    rgb = np.random.random_sample((5, 4, 3))
    rgb2 = np.random.random_sample((5, 4, 3))
    rgb3 = np.random.random_sample((5, 3))
    rgb4 = np.random.random_sample((4, 3, 5))
    rgb5 = np.random.random_sample((5, 4, 3))
    rgb6 = np.random.random_sample((5, 4, 3))

    M = np.random.random_sample((3, 6))
    M2 = np.random.random_sample((6, 4))
    M3 = np.random.random_sample((4, 3))
    M4 = np.random.random_sample((3, 6, 5))
    M5 = np.random.random_sample((5, 3, 6))
    M6 = np.random.random_sample((5, 3))

    slow_result1 = np.zeros((5, 4, 6))
    slow_result12 = np.zeros((5, 4, 6))
    slow_result13 = np.zeros((5, 4, 6))
    slow_result2 = np.zeros((5, 6, 3))
    slow_result3 = np.zeros((5, 4))
    slow_result4 = np.zeros((4, 6, 5))
    slow_result5 = np.zeros((5, 4, 6))
    slow_result6 = np.zeros((5, 4))

    for i in range(rgb.shape[0]):
        # for j in range(rgb.shape[1]):
        #     slow_result1[i, j, :] = np.dot(M, rgb[i, j, :])
        # slow_result13[i,:,:] = np.dot(M, rgb[i,:,:].T).T
        slow_result12[i, :, :] = np.dot(rgb[i, :, :], M)

    for i in range(rgb2.shape[0]):
        slow_result2[i, :, :] = np.dot(M2, rgb2[i, :, :])

    for i in range(rgb3.shape[0]):
        slow_result3[i, :] = np.dot(M3, rgb3[i, :])

    for i in range(rgb4.shape[2]):
        slow_result4[:, :, i] = np.dot(rgb4[:, :, i], M4[:, :, i])

    for i in range(rgb5.shape[0]):
        slow_result5[i, :, :] = np.dot(rgb5[i, :, :], M5[i, :, :])

    for i in range(rgb6.shape[0]):
        slow_result6[i] = np.dot(rgb6[i, :, :], M6[i])

    # M x N x O * O x P = M x N x P right multiply static matrix by a matrix list
    rgb_reshaped = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))
    # result1 = np.dot(M, rgb_reshaped.T).T.reshape(rgb.shape)
    result12 = np.dot(rgb_reshaped, M).reshape(slow_result12.shape)

    # M x N x O * P x N = M x P x O  left multiply static matrix by a matrix list
    rgb2Temp = rgb2.swapaxes(1, 2)
    rgb_reshaped2 = rgb2Temp.reshape((rgb2Temp.shape[0] * rgb2Temp.shape[1], rgb2Temp.shape[2]))
    result2 = np.dot(rgb_reshaped2, M2.T).reshape((5, 3, 6)).swapaxes(1, 2)

    # M x N * O x N = M x O  left multiply static matrix by vector list   A*b
    result3 = np.dot(M3, rgb3.swapaxes(0, 1)).swapaxes(0, 1)

    # N x O x M * O x P x M = N x P x M left multiply two matrix lists with batch at end
    result4temp = rgb4[:, :, None] * M4
    result4 = result4temp.sum(axis=1)

    # M x N x O * M x O x P left multiply two matrix lists
    result5temp = np.rollaxis(rgb5, 0, 3)[:, :, None] * np.rollaxis(M5, 0, 3)
    result5 = result5temp.sum(axis=1)
    result5 = np.rollaxis(result5, 2, 0)
    # without rollaxis
    temper = np.swapaxes(rgb5, 0, 1)
    temper = np.swapaxes(temper, 1, 2)
    temper = temper[:, :, None]
    temper2 = np.swapaxes(M5, 0, 1)
    temper2 = np.swapaxes(temper2, 1, 2)
    result52 = temper * temper2
    result52 = result52.sum(axis=1)
    result52 = np.squeeze(result52)
    result52 = np.swapaxes(result52, 2, 1)
    result52 = np.swapaxes(result52, 1, 0)

    # M x N x O * M x O = M x N left multiply a matrix list by a vector list
    result6temp = np.rollaxis(rgb6, 0, 3)[:, :, None] * np.rollaxis(M6[:, :, None], 0, 3)
    result6 = result6temp.sum(axis=1)
    result6 = np.squeeze(result6)
    result6 = np.rollaxis(result6, 1, 0)
    # without rollaxis
    temper = np.swapaxes(rgb6, 0, 1)
    temper = np.swapaxes(temper, 1, 2)
    temper2 = np.swapaxes(M6[:, :, None], 0, 1)
    temper2 = np.swapaxes(temper2, 1, 2)
    result6temp2 = temper[:, :, None] * temper2
    result62 = result6temp2.sum(axis=1)
    result62 = np.squeeze(result62)
    result62 = np.swapaxes(result62, 0, 1)

    print("Math Results")
    # print np.allclose(slow_result1, result1)
    # print np.allclose(slow_result13, result1)
    print np.allclose(slow_result12, result12)
    print np.allclose(slow_result2, result2)
    print np.allclose(slow_result3, result3)
    print np.allclose(slow_result4, result4)
    print np.allclose(slow_result5, result5)
    print np.allclose(slow_result5, result52)
    print np.allclose(slow_result6, result6)
    print np.allclose(slow_result6, result62)

if doScanExample:
    print("Scan Example")
    import numpy

    coefficients = theano.tensor.vector("coefficients")
    x = T.scalar("x")

    max_coefficients_supported = 10000

    # Generate the components of the polynomial
    components, updates = theano.scan(
        fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
        outputs_info=None,
        sequences=[coefficients, theano.tensor.arange(max_coefficients_supported)],
        non_sequences=x)
    # Sum them up
    polynomial = components.sum()

    # Compile a function
    calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

    # Test
    test_coefficients = numpy.asarray([1, 0, 2], dtype=numpy.float32)
    test_value = 3
    print(calculate_polynomial(test_coefficients, test_value))
    print(1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2))

if doScanPinv:
    print ("Scan Pinv")


    def H_Transposed_SInverseAtOneIndex(index, Sarg, Harg):
        return T.dot(T.transpose(Harg), T.nlinalg.pinv(Sarg[index, :, :]))


    S = T.tensor3(dtype=theano.config.floatX)
    H = T.matrix(dtype=theano.config.floatX)

    H_Transposed_SInverseVar, updates = theano.scan(fn=H_Transposed_SInverseAtOneIndex,
                                                    outputs_info=None,
                                                    sequences=theano.tensor.arange(S.shape[0]),
                                                    non_sequences=[S, H])

    f3 = theano.function(inputs=[S, H], outputs=H_Transposed_SInverseVar)

    S2 = np.exp(np.random.randn(5, 3, 3)).astype(theano.config.floatX) - 5
    H2 = np.exp(np.random.randn(3, 3)).astype(theano.config.floatX) - 5
    output = f3(S2, H2)
    print (output)

if doHardSigmoid:
    print ("plot hard sigmoid")

    xer = T.vector("xer")
    shiftH = 10
    scaleH = 1.0
    shiftV = 0.0
    scaleV = 1.0
    yer = scaleV * (T.nnet.hard_sigmoid(scaleH * T.log(xer) - shiftH)) - shiftV
    hardSigmoid = theano.function(inputs=[xer], outputs=[yer])

    xNums = np.arange(-10, 10, .01, dtype=np.float32)
    yNums = np.array(hardSigmoid(xNums), dtype=np.float32).squeeze()
    plt.semilogx(xNums, yNums)
    plt.plot([1, 1], [0, 1], color='k', linestyle='-', linewidth=2)
    plt.plot([0, 0], [0, 1], color='r', linestyle='-', linewidth=2)
    # plt.show()

if doResidualStuff:
    print ("\nDoing residual stuff")

    S = T.tensor3("S", dtype=theano.config.floatX)  # (bs x nm x nm)
    y = T.matrix("y", dtype=theano.config.floatX)  # (bs x nm)
    Kal = T.tensor3("Kal", dtype=theano.config.floatX)  # (bs x ns x nm)

    expectedSigma = T.sqrt(T.diagonal(S, offset=0, axis1=1, axis2=2))
    scaleH = 5.0
    shiftH = 2.5
    # residualMetric = T.nnet.hard_sigmoid(scaleH * (3 * expectedSigma / y) - shiftH)
    # residualMetric = expectedSigma > y
    residualMetric = T.prod(2 * expectedSigma > y, axis=1)
    KalResidual = Kal * residualMetric[:, None, None]
    getResidual = theano.function(inputs=[S, y, Kal], outputs=[KalResidual])

    S2 = np.exp(np.random.rand(5, 2, 2)).astype(theano.config.floatX) + 10
    y2 = np.exp(np.random.rand(5, 2)).astype(theano.config.floatX) + 5
    Kal2 = np.exp(np.random.rand(5, 4, 2)).astype(theano.config.floatX) + 5

    # finalAnswer = Kal2 * y2[:, None, :]
    finalAnswer = getResidual(S2, y2, Kal2)[0]

    print("Kal2")
    print (Kal2)
    print("Final Answer")
    print (finalAnswer)

if doSemiPositiveStuff:
    print("\nMaking sure its semi-positive")

    p = T.matrix(dtype=theano.config.floatX)

    # force to be symmetric
    pSymmetric = (p + T.transpose(p)) / 2.0
    # take the eigen values of this symmetric matrix
    (w, v) = T.nlinalg.eigh(pSymmetric)
    # force eigenvalues to be positive
    w *= T.cast(w >= 0., dtype='float32')
    # rebuild p with new eigenvalues
    pOut = T.dot(T.dot(v, T.diag(w)), T.nlinalg.matrix_inverse(v))

    fer = theano.function(inputs=[p], outputs=[pOut])

    Ser = np.exp(np.random.randn(3, 3)).astype(theano.config.floatX) * 0.00001 + np.eye(3).astype(
        theano.config.floatX) * 10.0
    Ser = Ser.astype(theano.config.floatX)
    print (Ser)
    outer = fer(Ser)
    print (outer)

if do2DConvStuff:
    signalLength = 44000
    batch_size = 1
    # signal for everyone
    dt = 1 / 44000.0
    start = 0
    end = signalLength * dt
    print ("nyquist freq is {nyqist}".format(nyqist=1 / (2 * dt)))
    signalFreq = 100.0
    signalFreq2 = 25.0
    x = np.arange(start, end, dt, dtype=theano.config.floatX)
    signal = np.sin(signalFreq * 2 * np.pi * x)
    signal2 = np.sin(signalFreq2 * 2 * np.pi * x)
    signal[signalLength / 2:] = signal2[signalLength / 2:]
    signal = signal.astype(theano.config.floatX)
    plt.figure(1)
    plt.plot(x, signal)
    plt.title("Signal")

    sigmaValues = (np.array([0.05, 0.05, 0.05, 0.05]) * signalLength).astype(theano.config.floatX)
    kValues = (np.array([25, 50, 100, 200]) * 2 / (signalLength * dt)).astype(theano.config.floatX)
    bank_size = kValues.shape[0]

    # the input will be the filter banks
    input_channels = 1
    input_rows = bank_size
    input_columns = signalLength * 2
    input_shape = (batch_size, input_channels, input_rows, input_columns)
    # the filter will be the signal
    output_channels = 1
    filter_rows = 1
    filter_columns = signalLength
    filter_shape = (output_channels, input_channels, filter_rows, filter_columns)

    print (np.array(input_shape) - np.array(filter_shape) + 1)
    # # theano full filter bank stuff
    # inputSignal = T.tensor3("inputSignalTheano")
    # # inputSignal = (bs x timesteps x input_dim)
    # inputSignalReshaped = T.swapaxes(inputSignal, 1, 2)  # (bs x input_dim x timesteps)
    # inputSignalReshaped = inputSignalReshaped[:, :, None, :]  # (bs x input_dim x 1 x timesteps)
    # kValuesT = theano.shared(kValues, "kValues")  # (output_dim, )
    # sigmaValuesT = theano.shared(sigmaValues, "sigmaValues")  # (output_dim, )
    #
    # filterColumnsIndexShift = T.arange(0, filter_columns)
    # coefficientsReal = T.cos(
    #     -2 * np.pi * T.outer(kValuesT, filterColumnsIndexShift) / filterColumnsIndexShift.shape[0])
    # coefficientsImag = T.sin(
    #     -2 * np.pi * T.outer(kValuesT, filterColumnsIndexShift) / filterColumnsIndexShift.shape[0])
    # coefficientsReal = coefficientsReal[:, None, None, :]
    # coefficientsImag = coefficientsImag[:, None, None, :]
    # mu = T.cast(filterColumnsIndexShift[filterColumnsIndexShift.shape[0] / 2], dtype=theano.config.floatX)
    # gaussianWindows = 1 / T.sqrt(2 * sigmaValuesT[:, None] ** 2 * np.pi) * T.exp(
    #     T.outer(1 / (2 * sigmaValuesT ** 2), - T.power((filterColumnsIndexShift - mu), 2)))
    # gaussianWindows = gaussianWindows[:, None, None, :]
    #
    # filterBankReal = T.cast(coefficientsReal * gaussianWindows, theano.config.floatX)
    # outputConvReal = theano.tensor.nnet.conv2d(input=inputSignalReshaped,
    #                                            filters=filterBankReal,
    #                                            input_shape=input_shape,
    #                                            filter_shape=filter_shape,
    #                                            border_mode='valid',
    #                                            filter_flip=True)
    # filterBankImag = T.cast(coefficientsImag * gaussianWindows, theano.config.floatX)
    # outputConvImag = theano.tensor.nnet.conv2d(input=inputSignalReshaped,
    #                                            filters=filterBankImag,
    #                                            input_shape=input_shape,
    #                                            filter_shape=filter_shape,
    #                                            border_mode='valid',
    #                                            filter_flip=True)
    # powerOut = T.power(outputConvReal, 2) + T.power(outputConvImag, 2)
    # powerOut = powerOut[:, :, 0, :]
    # powerOut = T.swapaxes(powerOut, 1, 2)
    #
    # waveletCall = theano.function(inputs=[inputSignal], outputs=[powerOut])
    #
    # powerOut = waveletCall(signal[None, :, None])
    # # (bs x timesteps x output_size)
    # maxBatchesToShow = 9
    # plotsX = np.ceil(np.sqrt(min(batch_size, maxBatchesToShow)))
    # plotsY = plotsX
    # totalPlots = int(plotsX * plotsY)
    # for indexer in range(totalPlots):
    #     plt.figure(9)
    #     plt.subplot(plotsX, plotsY, indexer + 1)
    #     plt.plot(np.arange(0, (end - start) / dt), powerOut[0, indexer, 0, :])
    #     plt.ylim([np.min(powerOut), np.max(powerOut)])
    #
    # conv function for numpy
    inputSignal2 = T.tensor4("inputSignal2", dtype=theano.config.floatX)
    filterBank = T.tensor4("filterBank", dtype=theano.config.floatX)
    convOut = theano.tensor.nnet.conv2d(input=inputSignal2,
                                        filters=filterBank,
                                        input_shape=input_shape,
                                        filter_shape=filter_shape,
                                        border_mode='valid',
                                        filter_flip=True)
    convFunc = theano.function(inputs=[inputSignal2, filterBank], outputs=[convOut])

    filterBankNP = np.zeros((bank_size, input_columns), dtype=theano.config.floatX)
    filterBankNPImag = np.zeros((bank_size, input_columns), dtype=theano.config.floatX)
    gaussianWindows = np.zeros((bank_size, input_columns), dtype=theano.config.floatX)

    # # make filterbanks and gaussian window in NP
    # for indexer in range(output_channels):
    #     useWavelet = False
    #     if useWavelet:
    #         sigma = (indexer + 1)
    #         filterColumnsIndexShift = (np.arange(filter_columns) - filter_columns / 2) / (20.0)
    #         c_sigma = (1 + np.exp(-sigma ** 2) - 2 * np.exp(-3 / 4.0 * sigma ** 2)) ** (-1 / 2.0)
    #         k_sigma = np.exp(-1 / 2.0 * sigma ** 2)
    #         psi_sigma = c_sigma * np.pi ** (-1 / 4.0) * np.exp(-1 / 2.0 * filterColumnsIndexShift ** 2) * \
    #                     (np.exp(1j * sigma * filterColumnsIndexShift) - k_sigma)
    #         filterBankNP[indexer, 0, 0, :] = np.real(psi_sigma)
    #         filterBankNPImag[indexer, 0, 0, :] = np.imag(psi_sigma)
    #     else:
    #         sigma = sigmaValues[indexer]
    #         k = kValues[indexer]
    #         filterColumnsIndexShift = np.arange(filter_columns)
    #         coefficientsReal = np.cos(-2 * np.pi * k * filterColumnsIndexShift / filterColumnsIndexShift.shape[0])
    #         coefficientsImag = np.sin(-2 * np.pi * k * filterColumnsIndexShift / filterColumnsIndexShift.shape[0])
    #         filterBankNP[indexer, 0, 0, :] = coefficientsReal
    #         filterBankNPImag[indexer, 0, 0, :] = coefficientsImag
    #         mu = float(filterColumnsIndexShift[filterColumnsIndexShift.shape[0] / 2])
    #         gaussianWindows[indexer, 0, 0, :] = 1 / np.sqrt(2 * sigma ** 2 * np.pi) * np.exp(
    #             - np.power((filterColumnsIndexShift - mu), 2) / (2 * sigma ** 2))
    #     plt.figure(2)
    #     plt.subplot(2, 2, indexer + 1)
    #     plt.plot(filterBankNP[indexer, 0, 0, :])
    #     plt.title("Real filter bank")
    #     plt.figure(3)
    #     plt.subplot(2, 2, indexer + 1)
    #     plt.plot(filterBankNPImag[indexer, 0, 0, :])
    #     plt.title("imag filter bank")
    #     plt.figure(4)
    #     plt.subplot(2, 2, indexer + 1)
    #     plt.plot(gaussianWindows[indexer, 0, 0, :])
    #     plt.title("gaussian window")

    # make in numpy without looping
    filterColumnsIndexShift = np.arange(input_columns)
    coefficientsReal = np.cos(
        -2 * np.pi * np.outer(kValues, filterColumnsIndexShift) / filterColumnsIndexShift.shape[0])
    coefficientsImag = np.sin(
        -2 * np.pi * np.outer(kValues, filterColumnsIndexShift) / filterColumnsIndexShift.shape[0])
    filterBankNP[:, :] = coefficientsReal
    filterBankNPImag[:, :] = coefficientsImag

    gaussianWindowColumnsIndex = np.arange(input_columns)
    mu = float(gaussianWindowColumnsIndex[gaussianWindowColumnsIndex.shape[0] / 2])
    gaussianWindows[:, :] = 1 / np.sqrt(2.0 * sigmaValues[:, None] ** 2 * np.pi) * np.exp(
        np.outer(1.0 / (2.0 * sigmaValues ** 2), - np.power((gaussianWindowColumnsIndex - mu), 2)))
    gaussianWindows = gaussianWindows.astype(theano.config.floatX)
    filterBankNP = filterBankNP[None, None, :, :]
    filterBankNPImag = filterBankNPImag[None, None, :, :]
    gaussianWindows = gaussianWindows[None, None, :, :]
    signal = signal[None, None, None, :]
    filterBankNP = gaussianWindows * filterBankNP
    filterBankNPImag = gaussianWindows * filterBankNPImag
    outputNP = convFunc(filterBankNP, signal)[0]
    outputNPImag = convFunc(filterBankNPImag, signal)[0]
    powerOut = np.sqrt(np.power(outputNP, 2) + np.power(outputNPImag, 2))

    maxBatchesToShow = 9
    totalPlots = min(bank_size, maxBatchesToShow)
    plotsX = np.ceil(np.sqrt(totalPlots))
    plotsY = plotsX

    # plt.figure(9)
    # thingy = np.convolve(signal[0, :], filterBankNP[0, 0, 0, :], mode="valid")
    # thingyImag = np.convolve(signal[0, :], filterBankNPImag[0, 0, 0, :], mode="valid")
    # plt.plot(np.square(thingy) + np.square(thingyImag))

    filterBankNP = filterBankNP[0, 0, :, :]
    filterBankNPImag = filterBankNPImag[0, 0, :, :]
    gaussianWindows = gaussianWindows[0, 0, :, :]
    outputNP = outputNP[0, 0, :, :]
    outputNPImag = outputNPImag[0, 0, :, :]
    powerOut = powerOut[0, 0, :, :]
    for indexer in range(totalPlots):
        plt.figure(2)
        plt.subplot(plotsX, plotsY, indexer + 1)
        plt.plot(filterBankNP[indexer, :])
        plt.title("Real filter bank")
        # plt.figure(3)
        # plt.subplot(plotsX, plotsY, indexer + 1)
        # plt.plot(filterBankNPImag[indexer, :])
        # plt.title("imag filter bank")
        plt.figure(4)
        plt.subplot(plotsX, plotsY, indexer + 1)
        plt.plot(gaussianWindows[indexer, :])
        plt.title("gaussian window")
        # plt.figure(5)
        # plt.subplot(plotsX, plotsY, indexer + 1)
        # plt.plot(outputNP[indexer, :])
        # plt.title("Real power output")
        # plt.figure(6)
        # plt.subplot(plotsX, plotsY, indexer + 1)
        # plt.plot(outputNPImag[indexer, :])
        # plt.title("Imag power output")
        plt.figure(7)
        plt.subplot(plotsX, plotsY, indexer + 1)
        plt.semilogy(powerOut[indexer, :])
        plt.ylim([np.min(powerOut), np.max(powerOut)])
        plt.title("final power output {0}".format(indexer))
    plt.show()

if doZipshiz:
    import zipfile
    import os
    import dateutil
    import matplotlib.dates
    import datetime
    import re

    rawDataFolder = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"  # VLF signals raw data folder
    zipfilepath = os.path.join(rawDataFolder, 'bad files', '20160927_124106_3.zip')

    zippy = zipfile.ZipFile(zipfilepath)
    cadenceArray = np.array([])
    speedArray = np.array([])
    locationArray = np.array([])
    for zippedfile in zippy.infolist():
        cadenceMatch = re.match('.*BikeCad.csv', zippedfile.filename)
        speedMatch = re.match('.*BikeSpd.csv', zippedfile.filename)
        locationMatch = re.match('.*Loc.csv', zippedfile.filename)
        heartMatch = re.match('.*.Heart.csv', zippedfile.filename)
        footpodMatch = re.match('.*Footpod.csv', zippedfile.filename)
        maMatch = re.match('.*MA.csv', zippedfile.filename)

        if cadenceMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('_ID', int), ('CrankRevolutions', int), ('Cadence', float), ('WorkoutActive', bool),
                          ('Timestamp', datetime.datetime)]
                cadenceArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype, converters={
                    'Timestamp': lambda s: datetime.datetime.utcfromtimestamp(int(s) / 1000.0)})
        if speedMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('_ID', int), ('WheelRevolutions', float), ('WheelCircumference', float),
                          ('SpeedInstant', float), ('WorkoutActive', bool), ('Timestamp', datetime.datetime)]
                speedArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype, converters={
                    'Timestamp': lambda s: datetime.datetime.utcfromtimestamp(int(s) / 1000.0)})
        if locationMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('Longitude', float), ('DistanceOffset', float), ('Accuracy', float), ('Altitude', float),
                          ('WorkoutActive', bool), ('Timestamp', np.int64), ('Latitude', float),
                          ('TotalDistance', float), ('GradeDeg', float), ('_ID', int), ('Speed', float)]
                locationArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype)

        if locationMatch:
            with zippy.open(zippedfile.filename) as csvFile:
                ndtype = [('Longitude', float), ('DistanceOffset', float), ('Accuracy', float), ('Altitude', float),
                          ('WorkoutActive', bool), ('Timestamp', datetime.datetime),
                          ('Latitude', float), ('TotalDistance', float), ('GradeDeg', float), ('_ID', int),
                          ('Speed', float)]
                locationArray = np.genfromtxt(csvFile, delimiter=',', names=True, dtype=ndtype,
                                              converters={'Timestamp': lambda s: datetime.datetime.utcfromtimestamp(
                                                  int(s) / 1000.0)})
    print ('cad')
    print (cadenceArray.dtype)
    print ('spd')
    print (speedArray.dtype)
    print ('loc')
    print (locationArray.dtype)

    loc2 = locationArray[['Latitude', 'Longitude', 'Altitude']].view(np.float64).reshape(locationArray.shape + (-1,))
    loc3 = (locationArray['Timestamp'] - locationArray['Timestamp'][0]) / 1000.0
    print (np.hstack((loc3[:, None], loc2)).shape)
