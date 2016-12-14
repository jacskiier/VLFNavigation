import theano
import theano.tensor as T
import numpy as np
import pylab as plt
import CoordinateTransforms
import CreateUtils

doTheanoShiz = False
doMatrixMultiplyFun = False
doScanExample = False
doScanPinv = False
doHardSigmoid = False
doResidualStuff = False
doSemiPositiveStuff = False
doZipshiz = False
doResoluitonComparison = False
doLatLonShiz = True

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

if doZipshiz:
    import zipfile
    import os
    import dateutil
    import matplotlib.dates
    import datetime
    import re

    rawDataFolder = CreateUtils.getRawDataFolder()
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

if doResoluitonComparison:
    def getPixelsPerSize(pixelHeight, pixelWidth, diagonal):
        totalPixels = pixelHeight * pixelWidth
        scaleFactor = diagonal / np.sqrt(pixelHeight ** 2 + pixelWidth ** 2)
        screenHeight = pixelHeight * scaleFactor
        screenWidth = pixelWidth * scaleFactor
        pixelsPerSize = totalPixels / (screenHeight * screenWidth)
        return pixelsPerSize


    def getNewDiag(pixelHeight, pixelWidth, diagonal, newPixelHeight=3840, newPixelWidth=2160):
        aspectRatio = pixelWidth / float(pixelHeight)
        pixelsPerSize = getPixelsPerSize(pixelHeight, pixelWidth, diagonal)
        newTotalPixels = newPixelHeight * newPixelWidth
        newArea = newTotalPixels / pixelsPerSize
        newDiag = np.sqrt((newArea * (1 + aspectRatio ** 2)) / aspectRatio)
        return newDiag


    pixelHeightMain = 1200
    pixelWidthMain = 1920
    # newPixelHeightMain = 3840
    # newPixelWidthMain = 2160
    newPixelHeightMain = 5120
    newPixelWidthMain = 2160
    diagonalMain = 24

    pixelsPerSize = getPixelsPerSize(pixelHeightMain, pixelWidthMain, diagonalMain)
    print ("pixels per size: {0}".format(pixelsPerSize))
    newDiag = getNewDiag(pixelHeightMain, pixelWidthMain, diagonalMain, newPixelHeightMain, newPixelWidthMain)
    print ("new diag: {0}".format(newDiag))

if doLatLonShiz:
    lat1 = 21.32525
    lat2 = 21.325038
    lon1 = -157.9439
    lon2 = -157.920485
    lat1 = lat1 * np.pi / 180.
    lat2 = lat2 * np.pi / 180.
    lon1 = lon1 * np.pi / 180.
    lon2 = lon2 * np.pi / 180.

    r = 6371.
    x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2.)
    y = (lat2 - lat1)
    d = np.sqrt(x * x + y * y) * r

    print(d)
    arr = np.array([[lat1, lon1, 0], [lat2, lon2, 0]])
    ecefCoords = CoordinateTransforms.LlhToEcef(arr)

    dist =np.sqrt(np.sum((ecefCoords[0,:] - ecefCoords[1,:]) ** 2))
    print (dist)