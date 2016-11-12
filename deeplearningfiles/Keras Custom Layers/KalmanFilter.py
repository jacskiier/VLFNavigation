import os

import keras.backend as K
import keras.callbacks
import keras.constraints
import matplotlib.pylab as plt
import numpy as np
import scipy.signal
import theano
import theano.tensor as T
import tqdm
from keras.engine.topology import Layer

import KerasClassifiers
import RunExperiment
import CreateUtils


class KalmanFilterLayer(Layer):
    def __init__(self, output_dim, statesXInitArg=None, PInitArg=None, phi=None, B=None, C=None, D=None, Q=None, H=None,
                 R=None,
                 trainMatrices=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = None
        self.batch_size = None
        self.ns = None
        self.ni = None
        self.trainMatrices = trainMatrices
        self.diagonalQ = (len(Q.shape) == 2 and (Q.shape[0] == 1 or Q.shape[1] == 1)) or len(Q.shape) == 1
        self.diagonalR = (len(R.shape) == 2 and (R.shape[0] == 1 or R.shape[1] == 1)) or len(R.shape) == 1
        self.updates = []

        self.initDict = {
            'statesX': statesXInitArg,  # bs, ns
            'PMatrix': PInitArg,  # bs, ns, ns
            'phiMatrix': phi,  # ns x ns
            'BMatrix': B,  # ns x ni
            'CMatrix': C,  # no x ns
            'DMatrix': D,  # no x ni
            'QMatrix': Q,  # ns x ns
            'HMatrix': H,  # nm x ns
            'RMatrix': R  # nm x nm
        }
        self.weightsDict = {}
        self.states = None
        self.P = None
        self.phi = None
        self.B = None
        self.C = None
        self.D = None
        self.Q = None
        self.H = None
        self.R = None

        self.y = None
        self.S = None

        super(KalmanFilterLayer, self).__init__(**kwargs)

    def doSizeAsserts(self, weightName, initValue):
        if weightName == 'statesX':
            assert initValue.shape[0] == self.batch_size, "States were initialized with bad batch size"
            assert initValue.shape[1] == self.ns, "States were initialized with a bad number of states"
        if weightName == 'PMatrix':
            assert initValue.shape[0] == self.batch_size, "P was initialized with a bad batch size"
            assert initValue.shape[1] == initValue.shape[2] == self.ns, "P was initialized with a bad number of states"
        if weightName == 'phiMatrix':
            assert initValue.shape[0] == initValue.shape[1] == self.ns, "phi has bas number of states"
        if weightName == 'BMatrix':
            assert initValue.shape[0] == self.ns, "B has bad ns"
            assert initValue.shape[1] == self.ni, "B has bad ni"
        if weightName == 'CMatrix':
            assert initValue.shape[0] == self.output_dim, "C has bad no"
            assert initValue.shape[1] == self.ns, "C has bad ns"
        if weightName == 'DMatrix':
            assert initValue.shape[0] == self.output_dim, "D has bad no"
            assert initValue.shape[1] == self.ni, "D has bad ni"
        if weightName == 'QMatrix':
            if len(initValue.shape) > 1:
                assert initValue.shape[0] == initValue.shape[1] == self.ns, "Q has bad ns"
            else:
                assert initValue.shape[0] == self.ns, "Q has bad ns"
        if weightName == 'HMatrix':
            assert initValue.shape[0] == self.input_dim, "H has bad nm"
            assert initValue.shape[1] == self.ns, "H has bad ns"
        if weightName == 'RMatrix':
            if len(initValue.shape) > 1:
                assert initValue.shape[0] == initValue.shape[1] == self.input_dim, "R has bad nm"
            else:
                assert initValue.shape[0] == self.input_dim, "R has bad nm"

    def getNumberOfStates(self):
        if self.initDict['statesX'] is not None:
            ns = self.initDict['statesX'].shape[-1]
        elif self.initDict['PMatrix'] is not None:
            ns = self.initDict['PMatrix'].shape[-1]
        elif self.initDict['phiMatrix'] is not None:
            ns = self.initDict['phiMatrix'].shape[0]
        elif self.initDict['BMatrix'] is not None:
            ns = self.initDict['BMatrix'].shape[0]
        elif self.initDict['CMatrix'] is not None:
            ns = self.initDict['CMatrix'].shape[1]
        elif self.initDict['QMatrix'] is not None:
            ns = self.initDict['QMatrix'].shape[0]
        elif self.initDict['HMatrix'] is not None:
            ns = self.initDict['HMatrix'].shape[0]
        else:
            raise ValueError("There is no way to know the number of states")
        return ns

    def getNumberOfInputs(self):
        if self.initDict['BMatrix'] is not None:
            ni = self.initDict['BMatrix'].shape[1]
        elif self.initDict['DMatrix'] is not None:
            ni = self.initDict['DMatrix'].shape[1]
        else:
            raise ValueError("No way to determine number of inputs")
        return ni

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.batch_size = input_shape[0]
        self.ns = self.getNumberOfStates()
        ns = self.ns
        self.ni = self.getNumberOfInputs()
        ni = self.ni
        no = self.output_dim
        nm = self.input_dim

        defaultInitDict = {
            'statesX': np.zeros((self.batch_size, ns)),  # bs, ns
            'PMatrix': np.zeros((self.batch_size, ns, ns)),  # bs, ns, ns
            'phiMatrix': np.eye(ns),  # ns x ns
            'BMatrix': np.zeros((ns, ni)),  # ns x ni
            'CMatrix': np.eye(np.max([no, ns]))[:no, :ns],  # no x ns
            'DMatrix': np.zeros((no, ni)),  # no x ni
            'QMatrix': np.eye(ns),  # ns x ns
            'HMatrix': np.eye(np.max([nm, ns]))[:nm, :ns],  # nm x ns
            'RMatrix': np.eye(nm),  # nm x nm
        }

        for (weightName, initValue) in self.initDict.iteritems():
            if initValue is None:
                finalInit = defaultInitDict[weightName]
            elif weightName == 'statesX':
                # statesInitArg.shape = (ns,) OR (bs, ns)
                if len(initValue.shape) < 2:
                    states_batch = np.repeat(initValue[np.newaxis, :], self.batch_size, axis=0)
                    if len(states_batch.shape) > 2:
                        states_batch = np.squeeze(states_batch)
                    statesXInit = states_batch
                elif initValue.shape[0] == self.batch_size:
                    statesXInit = initValue
                else:
                    errorString = "statesInit had bad batch size of {1} correct size should be {0}"
                    raise ValueError(errorString.format(self.batch_size, initValue.shape[0]))
                finalInit = statesXInit
            elif weightName == 'PMatrix':
                # PInitArg.shape = (ns, ns) OR (bs, ns , ns)
                if len(initValue.shape) < 3:
                    PInit = np.repeat(initValue[np.newaxis, :, :], self.batch_size, axis=0)
                elif initValue.shape[0] == self.batch_size:
                    PInit = initValue
                else:
                    errorString = "Pinitial had a bad batch size of {1} correct size should be {0}"
                    raise ValueError(errorString.format(self.batch_size, initValue.shape[0]))
                finalInit = PInit
            else:
                finalInit = initValue
            self.weightsDict[weightName] = K.variable(finalInit, name=weightName)
            self.doSizeAsserts(weightName, finalInit)

        self.non_trainable_weights = []
        if self.trainMatrices is None:
            self.non_trainable_weights = [self.states, self.P, self.phi, self.B, self.C, self.D, self.Q, self.H, self.R]
        else:
            for (weightName, trainWeight) in self.trainMatrices.iteritems():
                if not trainWeight:
                    self.non_trainable_weights.append(self.weightsDict[weightName])

        self.trainable_weights = []
        if self.trainMatrices is not None:
            for (weightName, weightInitValue) in self.initDict.iteritems():
                if self.trainMatrices[weightName]:
                    self.weightsDict[weightName] = K.variable(weightInitValue, name=weightName)
                    self.trainable_weights.append(self.weightsDict[weightName])

        self.states = self.weightsDict['statesX']
        self.P = self.weightsDict['PMatrix']
        self.phi = self.weightsDict['phiMatrix']
        self.B = self.weightsDict['BMatrix']
        self.C = self.weightsDict['CMatrix']
        self.D = self.weightsDict['DMatrix']
        self.Q = self.weightsDict['QMatrix']
        self.H = self.weightsDict['HMatrix']
        self.R = self.weightsDict['RMatrix']

        if self.diagonalQ:
            self.constraints[self.Q] = keras.constraints.nonneg()
        else:
            self.constraints[self.Q] = CovarianceMatrixConstraint()
        if self.diagonalR:
            self.constraints[self.R] = keras.constraints.nonneg()
        else:
            self.constraints[self.R] = CovarianceMatrixConstraint()

    def call(self, z, mask=None):
        # z = bs x nm

        bs = self.batch_size
        ns = self.ns
        nm = self.input_dim
        P_reshaped = T.reshape(self.P, (self.P.shape[0] * self.P.shape[1], self.P.shape[2]))
        P_reshaped.name = 'P_reshaped'
        # Propagate state estimate and covariance to this time

        # x = bs x ns  = ((ns x ns) * (bs x ns)')'
        statesAfterPlantUpdate = T.transpose(T.dot(self.phi, T.transpose(self.states)))
        statesAfterPlantUpdate.name = 'statesAfterPlantUpdate'

        # P*Phi'  = bs x ns x ns * (ns x ns)'
        P_phi_Transposed = T.reshape(T.dot(P_reshaped, T.transpose(self.phi)), self.P.shape)
        P_phi_Transposed.name = 'P_phi_Transposed'

        # Phi*(P*Phi') = ns x ns * (bs x ns x ns * (ns x ns)')
        P_temp = T.swapaxes(P_phi_Transposed, 1, 2)
        P_temp.name = 'P_temp'
        P_temp_reshaped = T.reshape(P_temp, (P_temp.shape[0] * P_temp.shape[1], P_temp.shape[2]))
        P_temp_reshaped.name = 'P_temp_reshaped'
        PAfterPlantUpdate = T.swapaxes(T.reshape(T.dot(P_temp_reshaped, T.transpose(self.phi)), P_temp.shape), 1, 2)
        # add in Q
        if self.diagonalQ:
            Q = T.nlinalg.diag(self.Q)
        else:
            Q = self.Q
        PAfterPlantUpdate = PAfterPlantUpdate + Q
        PAfterPlantUpdate.name = 'PAfterPlantUpdate'
        # Start Residual Update

        # y = bs x nm = ((nm x ns) * (bs x ns)')'
        y = z - T.transpose(T.dot(self.H, T.transpose(statesAfterPlantUpdate)))
        y.name = 'y residual'
        self.y = y

        # P*H' = bs x ns x nm = bs x ns x ns * (nm x ns)'
        # P*H' = bs x ns x nm = bs*ns x ns * (nm x ns)'
        P_H_Transposed = T.reshape(T.dot(P_reshaped, T.transpose(self.H)), (bs, ns, nm))
        P_H_Transposed.name = 'P_H_Transposed'

        # H*(P*H') = bs x nm x nm = nm x ns * (bs x ns x nm)
        P_temp2 = T.swapaxes(P_H_Transposed, 1, 2)  # (bs x ns x nm) => (bs x nm x ns)
        P_temp2.name = 'P_temp2'
        P_temp_reshaped2 = T.reshape(P_temp2, (P_temp2.shape[0] * P_temp2.shape[1], P_temp2.shape[2]))
        # (bs x nm x ns) => (bs*nm x ns)
        P_temp_reshaped2.name = 'P_temp_reshaped2'
        S = T.swapaxes(T.reshape(T.dot(P_temp_reshaped2, T.transpose(self.H)), (bs, nm, nm)), 1, 2)
        S.name = 'S matrix'

        # Add in R
        # S = H*(P*H') + R = bs x nm x nm + (nm x nm)
        if self.diagonalR:
            R = T.nlinalg.diag(self.R)
        else:
            R = self.R
        S = S + R
        self.S = S

        # H' * S^(-1) = bs x ns x nm =(nm x ns)' * bs x (nm x nm)^-1
        def H_Transposed_SInverseAtOneIndex(index, Sarg, Harg):
            return T.dot(T.transpose(Harg), T.nlinalg.matrix_inverse(Sarg[index, :, :]))

        H_Transposed_SInverseVar, updates = theano.scan(fn=H_Transposed_SInverseAtOneIndex,
                                                        outputs_info=None,
                                                        sequences=theano.tensor.arange(S.shape[0]),
                                                        non_sequences=[S, self.H])
        H_Transposed_SInverse = H_Transposed_SInverseVar
        H_Transposed_SInverse.name = 'H_Transposed_SInverse'

        # Kal = P * (H' * S^(-1)) = bs x ns x nm = bs x ns x ns * (bs x ns x nm)
        P_Temp_1 = np.swapaxes(PAfterPlantUpdate, 0, 1)
        P_Temp_1 = np.swapaxes(P_Temp_1, 1, 2)
        P_Temp_1 = T.shape_padaxis(P_Temp_1, 2)
        P_Temp_1.name = 'P_Temp_1'
        H_Transposed_SInverseTemp = np.swapaxes(H_Transposed_SInverse, 0, 1)
        H_Transposed_SInverseTemp = np.swapaxes(H_Transposed_SInverseTemp, 1, 2)
        H_Transposed_SInverseTemp.name = 'H_Transposed_SInverseTemp'
        resultTempKal = P_Temp_1 * H_Transposed_SInverseTemp
        resultTempKal = resultTempKal.sum(axis=1)
        resultTempKal = np.squeeze(resultTempKal)
        resultTempKal = np.swapaxes(resultTempKal, 2, 1)
        resultTempKal = np.swapaxes(resultTempKal, 1, 0)
        resultTempKal.name = 'resultTempKal'
        Kal = resultTempKal
        Kal.name = 'Kal'

        KalResidual = Kal
        # # residual monitoring
        # expectedSigma = T.sqrt(T.diagonal(S, offset=0, axis1=1, axis2=2))
        # expectedSigma.name = 'expectedSigma'
        # expectedSigma = T.clip(expectedSigma, 0, 1000)
        # # residualMetric = T.nnet.hard_sigmoid(5.0 * expectedSigma/y - 2.5)
        # residualMetric = T.prod(3.0*expectedSigma > y, axis=1, dtype=theano.config.floatX)
        # # residualMetric = T.ones((bs, nm))
        # KalResidual = Kal * residualMetric[:, None, None]

        # K * y = bs x ns = bs x ns x nm * bs x nm
        kalTemp = T.swapaxes(KalResidual, 0, 1)
        kalTemp = T.swapaxes(kalTemp, 1, 2)
        kalTemp = T.shape_padaxis(kalTemp, 2)
        kalTemp.name = 'kalTemp'
        yTemp = T.shape_padaxis(y, 2)
        yTemp = T.swapaxes(yTemp, 0, 1)
        yTemp = T.swapaxes(yTemp, 1, 2)
        yTemp.name = 'yTemp'
        resultTempxUpdate = kalTemp * yTemp
        resultTempxUpdate = resultTempxUpdate.sum(axis=1)
        resultTempxUpdate = T.squeeze(resultTempxUpdate)
        resultTempxUpdate = T.swapaxes(resultTempxUpdate, 0, 1)
        resultTempxUpdate.name = 'resultTempxUpdate'

        # x = x + (K * y) = bs x ns + (bs x ns)
        statesNew = statesAfterPlantUpdate + resultTempxUpdate
        statesNew = T.unbroadcast(statesNew)
        statesNew.name = 'statesNew'

        # I - K * H = bs x ns x ns = ns x ns - bs x ns x nm * nm x ns
        Kal_reshaped = T.reshape(KalResidual, (KalResidual.shape[0] * KalResidual.shape[1], KalResidual.shape[2]))
        Kal_reshaped.name = 'Kal_reshaped'
        Kal_H = T.reshape((T.dot(Kal_reshaped, self.H)), (bs, ns, ns))
        Kal_H.name = 'Kal_H'
        IdentityMatrix = T.eye(ns)
        IdentityMatrix.name = 'IdentityMatrix'
        I_Kal_H = IdentityMatrix - Kal_H
        I_Kal_H.name = 'I_Kal_H'

        # (I - K * H) * P = bs x ns x ns = (bs x ns x ns) * bs x ns x ns
        I_Kal_H_Temp = T.swapaxes(I_Kal_H, 0, 1)
        I_Kal_H_Temp = T.swapaxes(I_Kal_H_Temp, 1, 2)
        I_Kal_H_Temp = T.shape_padaxis(I_Kal_H_Temp, 2)
        I_Kal_H_Temp.name = 'I_Kal_H_Temp'
        P_temp3 = T.swapaxes(PAfterPlantUpdate, 0, 1)
        P_temp3 = T.swapaxes(P_temp3, 1, 2)
        P_temp3.name = 'P_temp3'
        resultPUpdate = I_Kal_H_Temp * P_temp3
        resultPUpdate = resultPUpdate.sum(axis=1)
        resultPUpdate = T.squeeze(resultPUpdate)
        resultPUpdate = T.swapaxes(resultPUpdate, 2, 1)
        resultPUpdate = T.swapaxes(resultPUpdate, 1, 0)
        resultPUpdate.name = 'resultPUpdate'
        PNew = resultPUpdate
        PNew = T.unbroadcast(PNew)
        PNew.name = 'PNew'
        self.updates = [(self.states, statesNew), (self.P, PNew)]

        # C * x = bs x ns = (no x ns * (bs x ns)')'
        output = T.transpose(T.dot(self.C, T.transpose(statesNew)))
        output.name = 'outputMatrix'
        return output

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim


class CovarianceMatrixConstraint(keras.constraints.Constraint):
    def __call__(self, p):
        # force to be symmetric
        pSymmetric = (p + K.transpose(p)) / 2.0
        # take the eigen values of this symmetric matrix
        (w, v) = T.nlinalg.eigh(pSymmetric)
        # force eigenvalues to be positive
        w *= K.cast(w >= 0., K.floatx())
        # rebuild p with new eigenvalues
        pOut = K.dot(K.dot(v, T.diag(w)), T.nlinalg.matrix_inverse(v))
        return pOut


def convertBatchSequencesToFullSequences(yArg, batch_sizeArg):
    slicesPerBatch = len(np.arange(batch_sizeArg, yArg.shape[0], batch_sizeArg))
    finalOut = np.zeros((batch_sizeArg, slicesPerBatch * yArg.shape[1]))
    for seqs in range(0, batch_sizeArg):
        ranger = np.arange(seqs, slicesPerBatch * batch_sizeArg, batch_sizeArg)
        filterRange = ranger
        oneSequence = yArg[filterRange, :].flatten()
        finalOut[seqs, :] = oneSequence
    return finalOut


def getQMatrix(gArg, qArg, fArg, Ts):
    gArg = np.matrix(gArg)
    qArg = np.matrix(qArg)
    fArg = np.matrix(fArg)
    (nx, nu) = gArg.shape
    (nq, mq) = qArg.shape

    assert (mq == nq) or (nu == mq), 'G and Q must be consistent.'

    # Check if q is positive semi - definite and symmetric
    assert np.all(np.linalg.eigvals(qArg) > 0), "q is not positive semi definite"
    assert np.allclose(qArg.transpose(), qArg), "q is not symmetric"

    # --- Compute discrete equivalent of continuous noise ---
    Zf = np.matrix(np.zeros((nx, nx)))
    Marg = np.concatenate((-fArg, gArg * qArg * gArg.transpose()), axis=1)
    Mlower = np.concatenate((Zf, fArg.transpose()), axis=1)
    Marg = np.concatenate((Marg, Mlower), axis=0)
    phi_aug = scipy.linalg.expm(Marg * Ts)
    phi12 = phi_aug[:nx, nx:]
    phi22 = phi_aug[nx:, nx:]
    qd = phi22.transpose() * phi12
    qd = (qd + qd.transpose()) / 2.0  # Make sure Qd is symmetric
    return qd


def getDiscreteSystem(Farg, Barg, Carg, Darg, Garg, Qarg, dtArg, matrixIsDiscreteArg):
    systemContinuous = (Farg, Barg, Carg, Darg)
    if 'plantMatrices' in matrixIsDiscreteArg and matrixIsDiscreteArg['plantMatrices']:
        (phiArg, BdArg, CdArg, DdArg) = systemContinuous
    else:
        # turn that into a discrete model
        (phiArg, BdArg, CdArg, DdArg, dtArg) = scipy.signal.cont2discrete(systemContinuous, dtArg)
    if 'QMatrix' in matrixIsDiscreteArg and matrixIsDiscreteArg['Qmatrix']:
        if (len(Qarg.shape) == 2 and Qarg.shape[1] == Qarg.shape[0]) or (
                    (len(Qarg.shape) == 2 and (Qarg.shape[1] == 1 or Qarg.shape[0] == 1)) or len(Qarg.shape) == 1):
            QdArg = Qarg
        else:
            raise ValueError("Bad Q matrix")
    else:
        if len(Qarg.shape) == 2 and Qarg.shape[1] == Qarg.shape[0]:
            QdArg = getQMatrix(Garg, Qarg, Farg, dtArg)
        elif (len(Qarg.shape) == 2 and (Qarg.shape[1] == 1 or Qarg.shape[0] == 1)) or len(Qarg.shape) == 1:
            QdArg = getQMatrix(Garg, np.diag(Qarg), Farg, dtArg)
            QdArg = np.diag(QdArg)
        else:
            raise ValueError("Bad Q matrix")
    return phiArg, BdArg, CdArg, DdArg, QdArg


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
    raise ValueError("bas OS")


def hard_sigmoid(x, leftZero=-2.5, rightOne=2.5, top=1, bottom=0):
    x = np.array(x)
    y = np.zeros_like(x)
    y[x < leftZero] = bottom
    y[x > rightOne] = top
    y[np.logical_and(x >= leftZero, x <= rightOne)] = x[np.logical_and(x >= leftZero, x <= rightOne)] * (
        (top - bottom) / (rightOne - leftZero)) + 0.5
    return y


def runMain():
    batch_size = 9
    timeSteps = 100

    makeUpData = False

    featureSetName = 'PatchShortTallAllFreq'
    datasetName = 'bikeneighborhoodPackFileNormCTDM'
    classifierType = 'LSTM'
    classifierSetName = 'RegressionAllClasses2LPlus2MLPStatefulAutoBatchDropRlrPWeight3RMSPropTD'
    modelStoreNameType = "bestLoss"
    whichSetArg = 0
    whichSetName = ['train', 'valid', 'test'][whichSetArg]
    (featureParameters,
     datasetParameters,
     classifierParameters) = RunExperiment.getParameters(rawDataFolder,
                                                         featureSetName,
                                                         datasetName,
                                                         classifierType,
                                                         classifierSetName)
    processedDataFolder = CreateUtils.convertPathToThisOS(datasetParameters['processedDataFolder'])
    datasetFile = os.path.join(processedDataFolder, featureParameters['featureSetName'] + '.hf')
    experimentsFolder = os.path.join(rawDataFolder, "Data Experiments", featureSetName, datasetName, classifierType,
                                     classifierSetName)

    yScaleFactor = datasetParameters['y value parameters']['yScaleFactor'] if 'yScaleFactor' in datasetParameters[
        'y value parameters'] else 1.0
    yBias = datasetParameters['y value parameters']['yBias'] if 'yBias' in datasetParameters[
        'y value parameters'] else 0.0
    xScaleFactor = datasetParameters['xScaleFactor'] if 'xScaleFactor' in datasetParameters else 1.0
    xBias = datasetParameters['xBias'] if 'xBias' in datasetParameters else 0.0

    system = 1
    if system == 1:
        # this system is a FOGM with 2D position updates and 2D position and velocity states
        dt = 0.1
        positionList = [0, 2]
        velocityList = [1, 3]
        # [n_dot     =F* [ n
        #  n_dot2          n_dot
        #  e_dot           e
        #  e_dot2]         e_dot]

        ns = 4
        nm = 2
        no = 2
        ni = 1
        sigma2_Plant = 0.0005
        sigma2_Meas = 10.0
        sigma_Initial = np.sqrt([5.25895608e-05, 0.001, 2.62760649e-05, 0.001])
        x_t0 = np.array([0.34633574, 0, 0.97964813, 0])
        P_t0 = np.diag(np.square(sigma_Initial)).astype(np.float32)
        # F = np.array([[0, 1, 0, 0],
        #               [0, 0, 0, 0],
        #               [0, 0, 0, 1],
        #               [0, 0, 0, 0], ], dtype=np.float32)  # ns x ns
        F = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0], ], dtype=np.float32) / 5.0  # ns x ns

        # n, n_dot, e, e_dot
        B = np.zeros((ns, ni), dtype=np.float32)  # ns x ni
        C = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])  # no x ns
        D = np.zeros((no, ni), dtype=np.float32)  # no x ni
        G = np.array([[0, 0],
                      [1, 0],
                      [0, 0],
                      [0, 1]], dtype=np.float32)  # ns x noisy states
        # QMatrix
        # [4.97102499e+00   9.99999997e-07   4.96971178e+00   9.99999997e-07]
        # RMatrix
        # [4.97038889  4.96807003]
        Q = np.diag([sigma2_Plant, sigma2_Plant])  # noisy states x noisy states
        # Q = np.array([4.97102499e-04, 4.96971178e-04])
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])  # nm x ns
        # R = np.array([[1, 0],
        #               [0, 1]], dtype=np.float32) * sigma2_Meas  # nm x nm
        # R = np.array([4.97038889e-04, 4.96807003e-04])  # nm x nm
        # R = np.array([0.00030531, 0.00019922])  # nm x nm
        R = np.array([[3.31367870e-04, 5.27556493e-06],
                      [5.27556493e-06, 1.10512432e-04]], dtype=np.float32)  # training set
        R = np.array([[0.00110002, 0.00048661],
                      [0.00048661, 0.0005926]], dtype=np.float32)  # validation set
        R = np.array([[0.01477426, 0.00034284],
                      [0.00034284, 0.00011283]], dtype=np.float32)  # validation set 2
        matrixIsDiscrete = {}
    elif system == 2:
        sigma2_Plant = 10.0 ** 2
        sigma2_Meas = 0.1
        sigma_Initial = [10, 0.01]
        dt = 0.1
        ns = 2
        nm = 1
        no = 2
        ni = 1
        x_t0 = np.zeros((ns,), dtype=np.float32)
        P_t0 = np.diag(np.square(sigma_Initial)).astype(np.float32)

        positionList = [0]
        velocityList = [1]
        # [n_dot     =F* [ n
        #  n_dot2          n_dot]

        F = np.array([[0, 1],
                      [0, 0]], dtype=np.float32)  # ns x ns
        B = np.zeros((ns, ni), dtype=np.float32)  # ns x ni
        C = np.eye(ns, dtype=np.float32)  # no x ns
        D = np.zeros((ns, ni), dtype=np.float32)  # no x ni
        G = np.array([[1, 0],
                      [0, 1]], dtype=np.float32)  # ns x noisy states
        Q = np.diag([sigma2_Plant, sigma2_Plant])  # noisy states x noisy states

        H = np.array([[1, 0]], dtype=np.float32)  # nm x ns
        R = np.array([[1]], dtype=np.float32) * sigma2_Meas  # nm x nm
        matrixIsDiscrete = {}
    elif system == 3:
        sigma2_Plant = 10.0 ** 2
        sigma2_Meas = 0.1
        sigma_Initial = [10, 0.01, 10.95, 3 * np.pi / 180.0, 10]
        dt = 0.1
        ns = 5
        nm = 1
        no = 5
        ni = 1
        x_t0 = np.zeros((ns,), dtype=np.float32)
        P_t0 = np.diag(np.square(sigma_Initial)).astype(np.float32)

        sigma_wind = 10.95
        Tau_wind = 10
        g = 9.8
        M = 2
        positionList = [0]
        velocityList = [1]
        # [P_dot            F* [ p
        #  velocity_dot          velocity
        # windforce_dot          wind_force
        # slope_dot              slope
        # laserbias_dot          laserbias]

        F = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 1 / M, -g, 0],
                      [0, 0, -1 / Tau_wind, 0, 0, ],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]], dtype=np.float32)  # ns x ns
        B = np.zeros((ns, ni), dtype=np.float32)  # ns x ni
        C = np.eye(ns, dtype=np.float32)  # no x ns
        D = np.zeros((ns, ni), dtype=np.float32)  # no x ni
        G = None
        Q = np.array([[2.5620248e-06, 1.9219974e-04, 1.5955061e-05, 0.0000000e+00, 0.0000000e+00],
                      [1.9219974e-04, 1.9223963e-02, 2.3932595e-03, 0.0000000e+00, 0.0000000e+00],
                      [1.5955061e-05, 2.3932595e-03, 4.7865206e-01, 0.0000000e+00, 0.0000000e+00],
                      [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                      [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],
                     dtype=np.float32)  # ns x ns

        H = np.array([[1, 0, 0, 0, 1]], dtype=np.float32)  # nm x ns
        R = np.array([[0.1]], dtype=np.float32) * sigma2_Meas  # nm x nm
        matrixIsDiscrete = {}
    else:
        raise ValueError("System number out of range")

    (phi, Bd, Cd, Dd, Qd,) = getDiscreteSystem(F, B, C, D, G, Q, dt, matrixIsDiscrete)
    phi = phi.astype(np.float32)
    Bd = Bd.astype(np.float32)
    Cd = Cd.astype(np.float32)
    Dd = Dd.astype(np.float32)
    Qd = Qd.astype(np.float32)

    if makeUpData:
        np.random.seed(0)
        trueStates = np.zeros((batch_size, timeSteps, ns), dtype=np.float32)

        trueStates[:, 0, :] = x_t0
        for timeStep in range(1, timeSteps):
            trueStates[:, timeStep, velocityList] = trueStates[:, timeStep - 1, velocityList] + np.random.randn(
                batch_size,
                len(
                    velocityList)) * np.sqrt(
                sigma2_Plant)
            trueStates[:, timeStep, positionList] = trueStates[:, timeStep - 1, positionList] + trueStates[:, timeStep,
                                                                                                velocityList] * dt
            # x[:, timeStep, 4] = x[:, timeStep - 1, 4] + x[:, timeStep, 1] * dt

        noisyMeasurements = np.zeros((batch_size, timeSteps, nm), dtype=np.float32)
        measList = [0]
        for timeStep in range(timeSteps):
            noisyMeasurements[:, timeStep, :] = trueStates[:, timeStep, measList] + \
                                                np.random.randn(batch_size,
                                                                len(measList)) * np.sqrt(sigma2_Meas)
    else:
        batch_size = classifierParameters['batch_size']
        timestepsPerSequence = datasetParameters[
            'timestepsPerSequence'] if 'timestepsPerSequence' in datasetParameters else None
        stateful = classifierParameters['stateful'] if 'stateful' in classifierParameters else False
        if stateful:
            packagedRowsPerSetDict = datasetParameters['packagedRowsPerSetDict']
            packagedRows = packagedRowsPerSetDict[whichSetName]
            auto_stateful_batch = classifierParameters[
                'auto_stateful_batch'] if 'auto_stateful_batch' in classifierParameters else False
            if auto_stateful_batch:
                batch_size = packagedRows
        (predicted_values_master,
         true_values_master,
         classLabelsMaster) = KerasClassifiers.getPred_Values_True_Labels(datasetFileName=datasetFile,
                                                                          experimentStoreFolder=experimentsFolder,
                                                                          whichSetArg=whichSetArg,
                                                                          datasetParameters=datasetParameters,
                                                                          classifierParameters=classifierParameters,
                                                                          modelStoreNameType=modelStoreNameType,
                                                                          shapeByRun=True)
        RMatrix = np.cov(np.reshape((predicted_values_master - true_values_master),
                                    (nm, np.prod(predicted_values_master.shape) / nm)))
        print("R: {0}".format(RMatrix))
        noisyMeasurements = np.array(predicted_values_master, dtype=np.float32)
        timeSteps = noisyMeasurements.shape[1]

        trueStates = np.zeros((batch_size, timeSteps, ns), dtype=np.float32)
        trueStates[:, :, positionList] = np.array(true_values_master, dtype=np.float32)
        trueStates[:, :, velocityList[0]] = np.gradient(trueStates[0, :, 0], edge_order=2) / dt
        trueStates[:, :, velocityList[1]] = np.gradient(trueStates[0, :, 1], edge_order=2) / dt

        # plt.subplot(2, 1, 1)
        # plt.plot(trueStates[0, :, 0], 'r')
        # plt.subplot(2, 1, 2)
        # plt.plot(np.gradient(trueStates[0, :, 0], edge_order=2)/dt, 'b')
        # plt.subplot(2, 1, 2)
        # plt.plot(np.diff(trueStates[0, :, 0])/dt, 'g')
        # plt.show()

    # make the kalman filter function
    kalmanLayer = KerasClassifiers.KalmanFilterLayer(no, statesXInitArg=x_t0, PInitArg=P_t0, phi=phi, B=Bd, C=Cd, D=Dd,
                                                     Q=Qd, H=H, R=R)
    measurement = T.matrix('measurement')
    kalmanLayer.build((batch_size, nm))
    outputs = kalmanLayer.call(measurement, None)
    doMeasurementUpdate = theano.function(inputs=[measurement], outputs=[outputs], updates=kalmanLayer.updates)
    # theano.pp(doMeasurementUpdate.maker.fgraph.outputs[0])
    # theano.printing.debugprint(outputs)

    dataArray = []

    doTheano = False
    # Start doing the estimation
    if doTheano:
        kalmanOutput = np.zeros((batch_size, timeSteps, no), dtype=np.float32)
        xEstTheano = np.zeros((batch_size, timeSteps, ns), dtype=np.float32)
        PEstTheano = np.zeros((batch_size, timeSteps, ns, ns), dtype=np.float32)
        xEstTheanoStd = np.zeros((batch_size, timeSteps, ns), dtype=np.float32)
        residualThenao = np.zeros((batch_size, timeSteps, nm), dtype=np.float32)
        STheano = np.zeros((batch_size, timeSteps, nm, nm), dtype=np.float32)
        for timeStep in tqdm.tqdm(range(timeSteps), desc="Theano time step Loop"):
            z_k = noisyMeasurements[:, timeStep, :]
            residualThenao[:, timeStep, :] = kalmanLayer.y.eval({measurement: z_k})
            STheano[:, timeStep, :, :] = kalmanLayer.S.eval()

            xNew = doMeasurementUpdate(z_k)
            kalmanOutput[:, timeStep, :] = xNew[0]
            xEstTheano[:, timeStep, :] = kalmanLayer.states.get_value()
            PEstTheano[:, timeStep, :, :] = kalmanLayer.P.get_value()

            for thisBatch in range(batch_size):
                xEstTheanoStd[thisBatch, timeStep, :] = np.sqrt(np.diag(PEstTheano[thisBatch, timeStep, :, :]))
        dataArray.append((xEstTheano, xEstTheanoStd, residualThenao, STheano))

    doNumpy = True
    if doNumpy:
        kalmanOutputNp = np.zeros((batch_size, timeSteps, no), dtype=np.float32)
        xEst = np.zeros((batch_size, timeSteps, ns), dtype=np.float32)
        xEstStd = np.zeros((batch_size, timeSteps, ns), dtype=np.float32)
        PEst = np.zeros((batch_size, timeSteps, ns, ns), dtype=np.float32)
        residual = np.zeros((batch_size, timeSteps, nm), dtype=np.float32)
        SNumpy = np.zeros((batch_size, timeSteps, nm, nm), dtype=np.float32)
        for timeStep in tqdm.tqdm(range(timeSteps), desc="Numpy Timestep Loop"):
            for thisBatch in range(batch_size):
                x_k_1 = xEst[thisBatch, timeStep - 1, :] if timeStep > 0 else x_t0
                x_k = np.dot(phi, x_k_1)
                P_k_1 = PEst[thisBatch, timeStep - 1, :, :] if timeStep > 0 else P_t0
                P_k = np.dot(phi, np.dot(P_k_1, phi.T)) + Qd

                z_k = noisyMeasurements[thisBatch, timeStep, :]

                y_k = z_k - np.dot(H, x_k)
                S_k = np.dot(H, np.dot(P_k, H.T)) + R
                # K_k = np.dot(P_k, np.dot(H.T, np.linalg.pinv(S_k)))
                K_k = np.dot(P_k, np.dot(H.T, np.linalg.inv(S_k)))

                if timeStep > 500 and y_k.all() > 0.0:
                    # residual monitoring
                    expectedSigma = np.sqrt(np.diagonal(S_k, offset=0, axis1=0, axis2=1))
                    expectedSigma = np.clip(expectedSigma, 0, 1000)
                    # residualMetric = hard_sigmoid(100000000.0 * expectedSigma / (1.0 * y_k), leftZero=0, rightOne=1)
                    residualMetric = np.prod(3.0 * expectedSigma > y_k, axis=0)
                    # residualMetric = T.ones((bs, nm))
                    K_k = K_k * residualMetric

                xEst[thisBatch, timeStep, :] = x_k + np.dot(K_k, y_k)
                PEst[thisBatch, timeStep, :, :] = np.dot((np.eye(ns) - np.dot(K_k, H)), P_k)

                kalmanOutputNp[thisBatch, timeStep, :] = np.dot(C, xEst[thisBatch, timeStep, :])
                xEstStd[thisBatch, timeStep, :] = np.sqrt(np.diag(PEst[thisBatch, timeStep, :, :]))

                residual[thisBatch, timeStep, :] = y_k
                SNumpy[thisBatch, timeStep, :, :] = S_k
        dataArray.append((xEst, xEstStd, residual, SNumpy))

    # print the output
    maxBatchesToShow = 1
    plotsX = np.ceil(np.sqrt(min(batch_size, maxBatchesToShow)))
    plotsY = plotsX
    totalPlots = int(plotsX * plotsY)
    doErrorPlots = True
    for (xEstTemp, xEstStdTemp, residualTemp, STemp) in dataArray:
        for thisBatch in range(totalPlots):
            rmse = np.sqrt(
                np.mean(np.square(xEstTemp[thisBatch, :, positionList] - trueStates[thisBatch, :, positionList])))
            print("RMSE Unscaled {0}".format(rmse))
            plt.figure(1)
            plt.plot(trueStates[thisBatch, :, positionList[1]], trueStates[thisBatch, :, positionList[0]], color='g')
            plt.plot(xEstTemp[thisBatch, :, positionList[1]], xEstTemp[thisBatch, :, positionList[0]], color='b')
            plt.plot(noisyMeasurements[thisBatch, :, 1], noisyMeasurements[thisBatch, :, 0], color='k')
            for thisPlotState in range(ns):
                absError = np.abs(xEstTemp[thisBatch, :, thisPlotState] - trueStates[thisBatch, :, thisPlotState])
                outside1Sigma = np.greater(xEstStdTemp[thisBatch, :, thisPlotState], absError)
                percentIn = np.mean(outside1Sigma)
                print("Estimator for state {1} was inside 1 sigma {0}%".format(percentIn, thisPlotState))
                plt.figure(thisPlotState + 2)
                plt.subplot(plotsX, plotsY, thisBatch % totalPlots + 1)
                if doErrorPlots:
                    for thisMeas in range(nm):
                        if H[thisMeas, thisPlotState] == 1:
                            plt.plot(
                                noisyMeasurements[thisBatch, :, thisMeas] - trueStates[thisBatch, :, thisPlotState],
                                color='k')
                            plt.plot(residualTemp[thisBatch, :, thisMeas], color='g', linestyle='--')
                            plt.plot(np.sqrt(STemp[thisBatch, :, thisMeas, thisMeas]), color='r', linestyle='--')
                            plt.plot(-np.sqrt(STemp[thisBatch, :, thisMeas, thisMeas]), color='r', linestyle='--')
                    plt.plot(xEstTemp[thisBatch, :, thisPlotState] - trueStates[thisBatch, :, thisPlotState],
                             color='b')
                    plt.plot(xEstStdTemp[thisBatch, :, thisPlotState], color='r', linestyle='-')
                    plt.plot(-xEstStdTemp[thisBatch, :, thisPlotState], color='r', linestyle='-')
                    plt.ylim([-0.2, 0.2])

                else:
                    plt.plot(trueStates[thisBatch, :, thisPlotState], color='g')
                    for thisMeas in range(nm):
                        if H[thisMeas, thisPlotState] == 1:
                            plt.plot(noisyMeasurements[thisBatch, :, thisMeas], color='k')
                    plt.plot(xEstTemp[thisBatch, :, thisPlotState], color='b')
                    plt.plot(xEstTemp[thisBatch, :, thisPlotState] + xEstStdTemp[thisBatch, :, thisPlotState],
                             color='r',
                             linestyle='-')
                    plt.plot(xEstTemp[thisBatch, :, thisPlotState] - xEstStdTemp[thisBatch, :, thisPlotState],
                             color='r',
                             linestyle='-')

    plt.show()


if __name__ == '__main__':
    runMain()
