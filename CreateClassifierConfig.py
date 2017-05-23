import os
import numpy as np
import CreateUtils
import math


def uniformLog(start, stop, base=10):
    startLog = math.log(start, base)
    stopLog = math.log(stop, base)
    randomNumberLog = np.random.uniform(startLog, stopLog)
    randomNumber = base ** randomNumberLog
    return randomNumber


def randomizeParameters(seed, classifierParameters, randRangeDict):
    np.random.seed(seed)
    for parameterNameRand, typeDict in randRangeDict.iteritems():
        if parameterNameRand in classifierParameters:
            if typeDict['type'] == 'log':
                classifierParameters[parameterNameRand] = uniformLog(typeDict['start'], typeDict['stop'])
            if typeDict['type'] == 'lin':
                classifierParameters[parameterNameRand] = np.random.uniform(typeDict['start'], typeDict['stop'])
            if typeDict['type'] == 'linD':
                classifierParameters[parameterNameRand] = np.random.randint(typeDict['start'], typeDict['stop'])
            if typeDict['type'] == 'linlin':
                classifierParameters[parameterNameRand] = np.random.uniform(typeDict['start'], typeDict['stop'],
                                                                            size=np.random.uniform(typeDict['listStart'],
                                                                                                   typeDict['listStop'])).tolist()
            if typeDict['type'] == 'linDlinD':
                randList = np.random.randint(low=typeDict['start'],
                                             high=typeDict['stop'],
                                             size=np.random.randint(typeDict['listStart'], typeDict['listStop']))
                classifierParameters[parameterNameRand] = randList.tolist()
            if typeDict['type'] == 'linDlinDInc':
                randList = np.array([10, 1])
                while not (np.diff(randList) < 0).any():
                    randList = np.random.randint(low=typeDict['start'],
                                                 high=typeDict['stop'],
                                                 size=np.random.randint(typeDict['listStart'], typeDict['listStop']))
                classifierParameters[parameterNameRand] = randList.tolist()
            if typeDict['type'] == 'linDlinDDec':
                randList = np.array([1, 10])
                while (np.diff(randList) > 0).any():
                    randList = np.random.randint(low=typeDict['start'],
                                                 high=typeDict['stop'],
                                                 size=np.random.randint(typeDict['listStart'], typeDict['listStop']))
                classifierParameters[parameterNameRand] = randList.tolist()
            if typeDict['type'] == 'list':
                values = typeDict['values']
                classifierParameters[parameterNameRand] = values[np.random.randint(0, len(values))]
    return classifierParameters


################################
# Parameters Begin ############
# ################################

if __name__ == '__main__':
    rootDataFolder = CreateUtils.getRootDataFolder()
    overwriteConfigFile = True

    classifierType = 'LSTM'
    classifierSetName = 'ClassificationAllClasses3LPlus3MLP1000StatefulAutoBatchDropRegRlrRMSPropTD'

    # classes are 0 indexed except when printed as a label!!!
    # rogueClasses = sorted(list(set(range(17)) - {1, 3, 4}))
    # rogueClasses = sorted((2, 8, 9, 16))
    # rogueClasses = sorted([1])
    rogueClasses = sorted([])
    rogueClasses = tuple(rogueClasses)

    configDict = {}
    # region Other Classifiers
    if classifierType == 'LogisticRegression':
        learning_rate = 0.13
        n_epochs = 1000
        batch_size = 600
        patience = 5000
        patience_increase = 2
        improvement_threshold = 0.995

        classifierGoal = 'classification'
        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'patience_increase': patience_increase,
            'improvement_threshold': improvement_threshold,
        }
    elif classifierType == 'MLP':
        classifierGoal = 'classification'
        learning_rate = 0.01
        n_epochs = 1000
        batch_size = 20
        patience = 100000
        patience_increase = 20
        improvement_threshold = 0.995

        rngSeed = 1234
        n_hidden = (500,)
        L1_reg = 0.00
        L2_reg = 0.0001

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'patience_increase': patience_increase,
            'improvement_threshold': improvement_threshold,
            'rngSeed': rngSeed,
            'n_hidden': n_hidden,
            'L1_reg': L1_reg,
            'L2_reg': L2_reg,
        }
    elif classifierType == 'ConvolutionalMLP':
        classifierGoal = 'classification'
        learning_rate = 0.1
        n_epochs = 200
        batch_size = 500
        patience = 10000
        patience_increase = 2
        improvement_threshold = 0.995
        rngSeed = 23455

        poolsize = (2, 2)
        filtersize = (5, 5)
        n_hidden = 500

        nkerns = (20, 50)

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'patience_increase': patience_increase,
            'improvement_threshold': improvement_threshold,
            'rngSeed': rngSeed,
            'nkerns': nkerns,

            'poolsize': poolsize,
            'filtersize': filtersize,
            'n_hidden': n_hidden,
        }
    elif classifierType == 'DBN':
        classifierGoal = 'classification'
        finetune_lr = 0.1
        pretraining_epochs = 100
        pretrain_lr = 0.01
        training_epochs = 1000
        k = 1
        batch_size = 10
        hidden_layers_sizes = [500, 500]
        patience_increase = 10.
        improvement_threshold = 0.990
        patienceMultiplier = 10
        rngSeed = 123
        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,
            'finetune_lr': finetune_lr,
            'pretraining_epochs': pretraining_epochs,
            'pretrain_lr': pretrain_lr,
            'k': k,
            'training_epochs': training_epochs,
            'batch_size': batch_size,
            'hidden_layers_sizes': hidden_layers_sizes,
            'patience_increase': patience_increase,
            'improvement_threshold': improvement_threshold,
            'patienceMultiplier': patienceMultiplier,
            'rngSeed': rngSeed,

        }
    elif classifierType == 'LinearRegression':
        classifierGoal = 'regression'
        learning_rate = 0.0009
        n_epochs = 400000
        batch_size = 600
        patience = 1200000
        patience_increase = 100
        improvement_threshold = 1.0

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'patience': patience,
            'patience_increase': patience_increase,
            'improvement_threshold': improvement_threshold,
        }
    elif classifierType == 'RandomForest':
        classifierGoal = 'regression'
        treeNumber = 2000
        rngSeed = 1234

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'treeNumber': treeNumber,
            'rngSeed': rngSeed,
        }
    elif classifierType == 'ADABoost':
        classifierGoal = 'regression'
        estimators = 50
        max_depth = 20
        rngSeed = 1234
        appendY = True

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'appendY': appendY,
            'max_depth': max_depth,
            'estimators': estimators,
            'rngSeed': rngSeed,
        }
    elif classifierType == 'GradientBoosting':
        classifierGoal = 'regression'
        estimators = 100
        rngSeed = 1234

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'estimators': estimators,
            'rngSeed': rngSeed,
        }
    elif classifierType == 'GaussianProcess':
        classifierGoal = 'regression'
        theta0 = 1e-2
        thetaL = 1e-4
        thetaU = 1e-1
        rngSeed = 1234

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'theta0': theta0,
            'thetaL': thetaL,
            'thetaU': thetaU,
            'rngSeed': rngSeed,
        }
    # endregion
    elif classifierType == 'LSTM':
        # Training
        classifierGoal = 'classification'
        learning_rate = 0.001
        epsilon = 1e-8
        decay = 0.0
        n_epochs = 2000
        batch_size = 0
        auto_stateful_batch = True
        reduceLearningRate = True
        rlrMonitor = 'loss'
        rlrFactor = 0.8
        rlrPatience = 10
        rlrCooldown = 10
        rlrEpsilon = 1e-4

        lossType = 'categorical_crossentropy'  # ['mse', 'categorical_crossentropy', 'falsePositiveRate']
        # ['root_mean_squared_error', 'root_mean_squared_error_unscaled', 'categorical_accuracy', 'falsePositiveRate']
        metrics = ['categorical_accuracy']
        optimizerType = 'rmsprop'

        # rmsprop specific
        rho = 0.9  # how much accumulation do you want? (using the RMS of the gradient)

        # adam specific
        beta_1 = 0.9  # how much accumulation do you want?
        beta_2 = 0.999  # how much accumulation do you want? (using the RMS of the gradient)

        # Wavelet Layer
        useWaveletTransform = False
        waveletBanks = 100
        maxWindowSize = 4410
        kValues = None
        sigmaValues = None

        # LSTM
        lstm_layers_sizes = [1000, 1000, 1000]
        dropout_W = 0.5
        dropout_U = 0.5
        dropout_LSTM = 0.0
        W_regularizer_l1_LSTM = 0.0001
        U_regularizer_l1_LSTM = 0.0001
        b_regularizer_l1_LSTM = 0.
        W_regularizer_l2_LSTM = 0.
        U_regularizer_l2_LSTM = 0.
        b_regularizer_l2_LSTM = 0.
        activations = 'tanh'
        inner_activations = 'hard_sigmoid'
        stateful = True
        consume_less = 'gpu'
        trainLSTM = True

        # MLP
        hidden_layers_sizes = [1000, 1000, 1000]
        hidden_activations = 'tanh'
        dropout_Hidden = 0.5
        W_regularizer_l1_hidden = 0.0001
        b_regularizer_l1_hidden = 0.
        W_regularizer_l2_hidden = 0.
        b_regularizer_l2_hidden = 0.
        finalActivationType = 'softmax'
        trainMLP = True

        # Maxout
        maxout_layers_sizes = []
        dropout_Maxout = 0.5
        W_regularizer_l1_maxout = 0.0001
        b_regularizer_l1_maxout = 0.
        W_regularizer_l2_maxout = 0.
        b_regularizer_l2_maxout = 0.
        trainMaxout = True

        # Model
        useTimeDistributedOutput = True
        onlyBuildModel = False
        useTeacherForcing = False
        teacherForcingDropout = 0.5

        # this will only load the previous weights for the hidden and lstm layers
        loadPreviousModelWeightsForTraining = False

        loadWeightsFilePath = CreateUtils.getPathRelativeToRoot(os.path.join(CreateUtils.getExperimentFolder(
            'PatchShortTallAllFreq',
            'bikeneighborhoodPackFileNormParticleTDM',
            'LSTM',
            'ClassificationAllClasses2LPlus2MLPStatefulAutoBatchDropReg2RlrRMSPropTD'),
            'best_modelWeights.h5'))

        # Append MLP Layers
        useAppendMLPLayers = False
        appendExpectedInput = 100
        append_layers_sizes = [2]
        append_activations = 'linear'
        dropout_Append = 0.0
        appendWeightsFile = CreateUtils.getPathRelativeToRoot(
            os.path.join(CreateUtils.getImageryFolder(), "bikeneighborhoodPackFileNormParticleTDMparticleLocationsFromDataset.csv"))
        trainAppend = True

        # Kalman Layer
        useKalman = False
        ns = 4
        nm = 2
        no = 2
        ni = 1
        sigma2_Plant = 0.00001
        sigma2_Meas = 10.0
        sigma_Initial = np.sqrt([5.25895608e-05, 0.001, 2.62760649e-05, 0.001])
        x_t0 = np.array([0.34633574, 0, 0.97964813, 0])
        P_t0 = np.diag(np.square(sigma_Initial)).astype(np.float32)
        F = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0], ], dtype=np.float32)  # ns x ns
        # n, n_dot, e, e_dot
        B = np.zeros((ns, ni), dtype=np.float32)  # ns x ni
        C = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])  # no x ns
        D = np.zeros((no, ni), dtype=np.float32)  # no x ni
        G = np.array([[0, 0],
                      [1, 0],
                      [0, 0],
                      [0, 1]], dtype=np.float32)  # ns x noisy states
        # Q = np.diag([sigma2_Plant, sigma2_Plant])  # noisy states x noisy states
        # Q = np.array([sigma2_Plant, sigma2_Plant])
        # Q = np.array([4.97102499e-04, 4.96971178e-04])
        Q = np.array([3.11591633e-04, 1.75137655e-04])
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])  # nm x ns
        # R = np.array([[1, 0],
        #               [0, 1]], dtype=np.float32) * sigma2_Meas  # nm x nm
        # R = np.array([sigma2_Meas, sigma2_Meas]) # nm x nm
        R = np.array([0.00030531, 0.00019922])  # nm x nm
        trainMatrices = {'statesX': False, 'PMatrix': False, 'phiMatrix': False, 'BMatrix': False, 'CMatrix': False,
                         'DMatrix': False, 'QMatrix': True, 'HMatrix': False, 'RMatrix': False}
        matrixIsDiscrete = {'plantMatrices': False, 'QMatrix': False}

        # random Specific
        addAllOptimizerParams = True
        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'rogueClasses': rogueClasses,

            'lstm_layers_sizes': lstm_layers_sizes,
            'activations': activations,
            'inner_activations': inner_activations,
            'stateful': stateful,
            'consume_less': consume_less,
            'trainLSTM': trainLSTM,

            'useTimeDistributedOutput': useTimeDistributedOutput,
            'onlyBuildModel': onlyBuildModel,
            'useTeacherForcing': useTeacherForcing,

            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,

            'lossType': lossType,
            'metrics': metrics,
            'optimizerType': optimizerType,

            'epsilon': epsilon,
        }
        if useTeacherForcing:
            configDict.update({'teacherForcingDropout': teacherForcingDropout})

        if dropout_W > 0:
            configDict.update({'dropout_W': dropout_W})
        if dropout_U > 0:
            configDict.update({'dropout_U': dropout_U})
        if dropout_LSTM > 0:
            configDict.update({'dropout_LSTM': dropout_LSTM})
        if W_regularizer_l1_LSTM > 0:
            configDict.update({'W_regularizer_l1_LSTM': W_regularizer_l1_LSTM})
        if U_regularizer_l1_LSTM > 0:
            configDict.update({'U_regularizer_l1_LSTM': U_regularizer_l1_LSTM})
        if b_regularizer_l1_LSTM > 0:
            configDict.update({'b_regularizer_l1_LSTM': b_regularizer_l1_LSTM})
        if W_regularizer_l2_LSTM > 0:
            configDict.update({'W_regularizer_l2_LSTM': W_regularizer_l2_LSTM})
        if U_regularizer_l2_LSTM > 0:
            configDict.update({'U_regularizer_l2_LSTM': U_regularizer_l2_LSTM})
        if b_regularizer_l2_LSTM > 0:
            configDict.update({'b_regularizer_l2_LSTM': b_regularizer_l2_LSTM})
        if stateful:
            configDict.update({'auto_stateful_batch': auto_stateful_batch, })
        if loadPreviousModelWeightsForTraining:
            previousDict = {
                'loadPreviousModelWeightsForTraining': loadPreviousModelWeightsForTraining,
                'loadWeightsFilePath': loadWeightsFilePath,
            }
            configDict.update(previousDict)
        if len(hidden_layers_sizes) > 0:
            hiddenDict = {
                'hidden_layers_sizes': hidden_layers_sizes,
                'hidden_activations': hidden_activations,
                'finalActivationType': finalActivationType,
                'trainMLP': trainMLP,
            }
            configDict.update(hiddenDict)
            if dropout_Hidden > 0:
                configDict.update({'dropout_Hidden': dropout_Hidden})
            if W_regularizer_l1_hidden > 0:
                configDict.update({'W_regularizer_l1_hidden': W_regularizer_l1_hidden})
            if b_regularizer_l1_hidden > 0:
                configDict.update({'b_regularizer_l1_hidden': b_regularizer_l1_hidden})
            if W_regularizer_l2_hidden > 0:
                configDict.update({'W_regularizer_l2_hidden': W_regularizer_l2_hidden})
            if b_regularizer_l2_hidden > 0:
                configDict.update({'b_regularizer_l2_hidden': b_regularizer_l2_hidden})
        if len(maxout_layers_sizes) > 0:
            maxoutDict = {
                "maxout_layers_sizes": maxout_layers_sizes,
                "trainMaxout": trainMaxout
            }
            configDict.update(maxoutDict)
            if dropout_Maxout > 0:
                configDict.update({'dropout_Maxout': dropout_Maxout})
            if W_regularizer_l1_maxout > 0:
                configDict.update({"W_regularizer_l1_maxout": W_regularizer_l1_maxout})
            if b_regularizer_l1_maxout > 0:
                configDict.update({"b_regularizer_l1_maxout": b_regularizer_l1_maxout})
            if W_regularizer_l2_maxout > 0:
                configDict.update({"W_regularizer_l2_maxout": W_regularizer_l2_maxout})
            if b_regularizer_l2_maxout > 0:
                configDict.update({"b_regularizer_l2_maxout": b_regularizer_l2_maxout})
        if useKalman:
            kalmanDict = {
                'useKalman': useKalman,
                'x_t0': x_t0,
                'P_t0': P_t0,
                'F': F,
                'B': B,
                'C': C,
                'D': D,
                'G': G,
                'Q': Q,
                'H': H,
                'R': R,
                'trainMatrices': trainMatrices,
                'matrixIsDiscrete': matrixIsDiscrete,
            }
            configDict.update(kalmanDict)
        if useWaveletTransform:
            waveletDict = {
                'useWaveletTransform': useWaveletTransform,
                'waveletBanks': waveletBanks,
                'kValues': kValues,
                'sigmaValues': sigmaValues,
                'maxWindowSize': maxWindowSize,
            }
            configDict.update(waveletDict)
        if useAppendMLPLayers:
            appendDict = {
                'useAppendMLPLayers': useAppendMLPLayers,
                'appendExpectedInput': appendExpectedInput,
                'append_layers_sizes': append_layers_sizes,
                'append_activations': append_activations,
                'dropout_Append': dropout_Append,
                'appendWeightsFile': appendWeightsFile,
                'trainAppend': trainAppend,
            }
            configDict.update(appendDict)
        if reduceLearningRate:
            rlrDict = {
                'reduceLearningRate': reduceLearningRate,
                'rlrMonitor': rlrMonitor,
                'rlrFactor': rlrFactor,
                'rlrPatience': rlrPatience,
                'rlrCooldown': rlrCooldown,
                'rlrEpsilon': rlrEpsilon,
            }
            configDict.update(rlrDict)
        if optimizerType == 'rmsprop' or addAllOptimizerParams:
            configDict.update({'rho': rho})
        if optimizerType == 'adam' or addAllOptimizerParams:
            adamDict = {
                'beta_1': beta_1,
                'beta_2': beta_2,
            }
            configDict.update(adamDict)
    ################################
    # Parameters End   ############
    ################################

    modelStoreFolder = CreateUtils.getModelFolder(classifierType=classifierType, classifierSetName=classifierSetName)
    if not os.path.exists(modelStoreFolder):
        os.makedirs(modelStoreFolder)
    configFileName = CreateUtils.getModelConfigFileName(classifierType, classifierSetName)
    doesFileExist = os.path.exists(configFileName)
    if not overwriteConfigFile:
        assert not doesFileExist, 'do you want to overwirte the config file?'
    if doesFileExist:
        configDictLoaded = CreateUtils.loadConfigFile(configFileName)
        dictDiffer = CreateUtils.DictDiffer(configDictLoaded, configDict)
        print(dictDiffer.printAllDiff())
    CreateUtils.makeConfigFile(configFileName, configDict)
    if doesFileExist:
        print("Overwrote file {0}".format(configFileName))
    else:
        print("Wrote file {0}".format(configFileName))
