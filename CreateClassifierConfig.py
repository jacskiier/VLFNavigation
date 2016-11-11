import yaml
import os
import numpy as np
import CreateUtils

rawDataFolder = CreateUtils.getRawDataFolder()

################################
# Parameters Begin ############
################################

if __name__ == '__main__':
    overwriteConfigFile = True

    classifierType = 'LSTM'
    classifierSetName = 'ClassificationAllClasses1LPlus1MLPStatefulWaveletAutoBatchRlrRMSPropTD'
    modelStoreFolder = os.path.join(rawDataFolder, "Processed Data Models", classifierType, classifierSetName)

    # classes are 0 indexed except when printed as a label!!!
    # rogueClasses = sorted(list(set(range(17)) - {1, 3, 4}))
    # rogueClasses = sorted((2, 8, 9, 16))
    # rogueClasses = sorted([1])
    rogueClasses = sorted([])
    rogueClasses = tuple(rogueClasses)

    configDict = {}
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
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
            'modelStoreFolder': modelStoreFolder,
            'rogueClasses': rogueClasses,

            'theta0': theta0,
            'thetaL': thetaL,
            'thetaU': thetaU,
            'rngSeed': rngSeed,
        }
    elif classifierType == 'LSTM':
        classifierGoal = 'classification'
        # LSTM
        lstm_layers_sizes = [100]
        dropout_W = 0.0
        dropout_U = 0.0
        dropout_LSTM = 0.0
        W_regularizer_l1_LSTM = 0.0
        U_regularizer_l1_LSTM = 0.0
        b_regularizer_l1_LSTM = 0.0
        W_regularizer_l2_LSTM = 0.0
        U_regularizer_l2_LSTM = 0.0
        b_regularizer_l2_LSTM = 0.0
        activations = 'tanh'
        inner_activations = 'hard_sigmoid'
        stateful = True
        consume_less = 'gpu'
        trainLSTM = True

        # MLP
        hidden_layers_sizes = [100]
        hidden_activations = 'tanh'
        dropout_Hidden = 0.0
        W_regularizer_l1_hidden = 0.0
        b_regularizer_l1_hidden = 0.0
        W_regularizer_l2_hidden = 0.0
        b_regularizer_l2_hidden = 0.0
        finalActivationType = 'softmax'
        trainMLP = True

        # Model
        useTimeDistributedOutput = True

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

        # Wavelet Layer
        useWaveletTransform = True
        waveletBanks = 100
        maxWindowSize = 10000
        kValues = None
        sigmaValues = None

        # Training
        learning_rate = 0.001
        epsilon = 1e-8
        decay = 0.0
        n_epochs = 2000
        batch_size = 13
        auto_stateful_batch = True
        reduceLearningRate = True
        rlrMonitor = 'loss'
        rlrFactor = 0.8
        rlrPatience = 10
        rlrCooldown = 10
        rlrEpsilon = 1e-4

        loadPreviousModelWeightsForTraining = False
        loadWeightsFilePath = os.path.join(rawDataFolder,
                                           'Data Experiments',
                                           'PatchShortTallAllFreq',
                                           'bikeneighborhoodSequenceOneFileNormCTDM',
                                           'LSTM',
                                           'RegressionAllClasses2LPlus2MLPStatefulAutoBatchDropRlrRMSPropTD2',
                                           'best_modelWeights.h5')
        lossType = 'categorical_crossentropy'  # ['mse', 'categorical_crossentropy', 'falsePositiveRate']
        # ['root_mean_squared_error_unscaled', 'categorical_accuracy', 'falsePositiveRate']
        metrics = ['categorical_accuracy']
        optimizerType = 'rmsprop'

        # rmsprop specific
        rho = 0.9  # how much accumulation do you want? (using the RMS of the gradient)

        # adam specific
        beta_1 = 0.9  # how much accumulation do you want?
        beta_2 = 0.999  # how much accumulation do you want? (using the RMS of the gradient)

        configDict = {
            'classifierSetName': classifierSetName,
            'classifierType': classifierType,
            'classifierGoal': classifierGoal,
            'modelStoreFolder': modelStoreFolder,
            'rogueClasses': rogueClasses,

            'lstm_layers_sizes': lstm_layers_sizes,
            'dropout_W': dropout_W,
            'dropout_U': dropout_U,
            'dropout_LSTM': dropout_LSTM,
            'W_regularizer_l1_LSTM': W_regularizer_l1_LSTM,
            'U_regularizer_l1_LSTM': U_regularizer_l1_LSTM,
            'b_regularizer_l1_LSTM': b_regularizer_l1_LSTM,
            'W_regularizer_l2_LSTM': W_regularizer_l2_LSTM,
            'U_regularizer_l2_LSTM': U_regularizer_l2_LSTM,
            'b_regularizer_l2_LSTM': b_regularizer_l2_LSTM,
            'activations': activations,
            'inner_activations': inner_activations,
            'stateful': stateful,
            'consume_less': consume_less,
            'trainLSTM': trainLSTM,

            'useTimeDistributedOutput': useTimeDistributedOutput,

            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'batch_size': batch_size,

            'lossType': lossType,
            'metrics': metrics,
            'optimizerType': optimizerType,

            'epsilon': epsilon,
        }
        if stateful:
            auto_stateful_batchDict = {
                'auto_stateful_batch': auto_stateful_batch,
            }
            configDict.update(auto_stateful_batchDict)
        if loadPreviousModelWeightsForTraining:
            previousDict = {
                'loadPreviousModelWeightsForTraining': loadPreviousModelWeightsForTraining,
                'loadWeightsFilePath': loadWeightsFilePath,
            }
            configDict.update(previousDict)
        if len(hidden_layers_sizes) > 0:
            hiddenDict = {
                'hidden_layers_sizes': hidden_layers_sizes,
                'dropout_Hidden': dropout_Hidden,
                'W_regularizer_l1_hidden': W_regularizer_l1_hidden,
                'b_regularizer_l1_hidden': b_regularizer_l1_hidden,
                'W_regularizer_l2_hidden': W_regularizer_l2_hidden,
                'b_regularizer_l2_hidden': b_regularizer_l2_hidden,
                'hidden_activations': hidden_activations,
                'finalActivationType': finalActivationType,
                'trainMLP': trainMLP,
            }
            configDict.update(hiddenDict)
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
        if optimizerType == 'rmsprop':
            configDict.update({'rho': rho})
        if optimizerType == 'adam':
            adamDict = {
                'beta_1': beta_1,
                'beta_2': beta_2,
            }
            configDict.update(adamDict)
    ################################
    # Parameters End   ############
    ################################

    if not os.path.exists(modelStoreFolder):
        os.makedirs(modelStoreFolder)
    configFileName = os.path.join(modelStoreFolder, "model set parameters.yaml")
    doesFileExist = os.path.exists(configFileName)
    if not overwriteConfigFile:
        assert not doesFileExist, 'do you want to overwirte the config file?'
    if doesFileExist:
        with open(configFileName, 'r') as myConfigFile:
            configDictLoaded = yaml.load(myConfigFile)
            dictDiffer = CreateUtils.DictDiffer(configDictLoaded, configDict)
            print(dictDiffer.printAllDiff())
    with open(configFileName, 'w') as myConfigFile:
        yaml.dump(configDict, myConfigFile, default_flow_style=False, width=1000)
        if doesFileExist:
            print("Overwrote file {0}".format(configFileName))
        else:
            print("Wrote file {0}".format(configFileName))
