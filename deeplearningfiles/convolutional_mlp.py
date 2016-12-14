"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import yaml
import cPickle
import mlp

import logistic_sgd
import ClassificationUtils
import CreateUtils


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, inputs, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.dtensor4
        :param inputs: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = inputs

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))

        if W is None:
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=self.input,
            image_shape=image_shape,
            filters=self.W,
            filter_shape=filter_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class LenetClassifier:
    def __init__(self,
                 inputs,
                 rng,
                 nkerns=(20, 50),
                 imageShape=(1, 28, 28),
                 imageShapeOrder=(0, 1, 2),
                 poolsize=(2, 2),
                 filtersize=(5, 5),
                 n_hidden=500,
                 n_out=10
                 ):
        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        self.input = inputs

        # Since input shape may not be correct for our network this will now give the corrected shape after the dim shuffle
        correctedImageShape = (imageShape[imageShapeOrder[0]], imageShape[imageShapeOrder[1]], imageShape[imageShapeOrder[2]])
        # First we reshape the input to its natural shape that was given to us
        self.layer0_input = self.input.reshape((self.input.shape[0], imageShape[0], imageShape[1], imageShape[2]), ndim=4)
        # Now we dimshuffle the natural shape to our expected shape  of (batch_size, image channel, width, height)
        self.layer0_input = self.layer0_input.dimshuffle(0, imageShapeOrder[0] + 1, imageShapeOrder[1] + 1, imageShapeOrder[2] + 1)

        # Construct the convolutional pooling layers:
        # filtering reduces the image size to (inputImageSize[0]-filter_shape[0]+1 , inputImageSize[0]-filter_shape[1]+1) = (24, 24)
        # For the default data (28-5+1 , 28-5+1) = (24, 24)
        # Maxpooling then reduces each image size by the pool
        # For the default data (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        # this repeats for each layer with the inputImageSize being the output image size of the last layer

        lastOutputLayerShape = (correctedImageShape[1], correctedImageShape[2])
        self.convolutionLayers = []
        for i in range(len(nkerns)):
            if i == 0:
                inputLayer = self.layer0_input
                inputnkern = correctedImageShape[0]
            else:
                inputLayer = self.convolutionLayers[-1].output
                inputnkern = nkerns[i - 1]
            outputnkern = nkerns[i]

            convolutionLayer = LeNetConvPoolLayer(
                rng,
                inputs=inputLayer,
                image_shape=(None, inputnkern, lastOutputLayerShape[0], lastOutputLayerShape[1]),
                filter_shape=(outputnkern, inputnkern, filtersize[0], filtersize[1]),
                poolsize=poolsize
            )
            self.convolutionLayers.append(convolutionLayer)
            lastOutputLayerShape = (
                (lastOutputLayerShape[0] - filtersize[0] + 1) / poolsize[0], (lastOutputLayerShape[1] - filtersize[1] + 1) / poolsize[1])
            # make sure I didn't shrink any image shape to 0
            assert all(lastOutputLayerShape), "The output for a layer was reduced to 0"

        # the MLP being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * lastOutputLayerShape[0] * lastOutputLayerShape[1]),
        # or in the default case (500, 50 * 4 * 4) = (500, 800) with the default values.
        # flatten to two dimensions and keep only the first the same
        hidden_layer_input = self.convolutionLayers[-1].output.flatten(2)

        # connect to a MLP classifier
        self.mlp = mlp.MLP(
            rng=rng,
            inputs=hidden_layer_input,
            n_in=nkerns[-1] * lastOutputLayerShape[0] * lastOutputLayerShape[1],
            n_hidden=n_hidden,
            n_out=n_out
        )

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.mlp.params
        for convolutionLayer in self.convolutionLayers:
            self.params += convolutionLayer.params

        self.y_pred = self.mlp.y_pred
        self.p_y_given_x = self.mlp.p_y_given_x

    def errors(self, y):
        return self.mlp.errors(y)

    def cost(self, y):
        return self.mlp.negative_log_likelihood(y)


def evaluate_lenet5(dataset='mnist.pkl.gz',
                    experimentStoreFolder='',
                    rogueClasses=(),
                    learning_rate=0.1,
                    n_epochs=200,
                    nkerns=(20, 50),
                    imageShape=(1, 28, 28),
                    imageShapeOrder=(0, 1, 2),
                    poolsize=(2, 2),
                    filtersize=(5, 5),
                    n_hidden=500,
                    batch_size=500,
                    patience=10000,
                    patience_increase=2,
                    improvement_threshold=0.995,
                    rngSeed=23455):
    """ Demonstrates lenet on MNIST dataset
    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type experimentStoreFolder: basestring
    :param experimentStoreFolder: the folder to store the model after it is trained

    :type rogueClasses: tuple of class outputs (int)
    :param rogueClasses: Tuple of classes to exclude from training to do rogue agalysis

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: tuple
    :param nkerns: number of kernels on each layer

    :type imageShape: tuple
    :param imageShape: shape of the input image (channels?, width, height)

    :type imageShapeOrder: char
    :param imageShapeOrder: the reorder to put the input features into a channel,width,height shaep

    :type poolsize: tuple or list of length 2
    :param poolsize: the downsampling (pooling) factor (#rows, #cols)

    :type filtersize: tuple or list of length 4
    :param filtersize: (number of filters, num input feature maps,
                              filter height, filter width)

    :type n_hidden: int
    :param n_hidden: number of hidden nodes in the MLP layer

    :type batch_size: int
    :param batch_size: size of batches to train on

    :type patience: int
    :param patience: look as this many examples regardless

    :type patience_increase: int
    :param patience_increase: wait this much longer when a new best is found

    :type improvement_threshold: float
    :param improvement_threshold:  a relative improvement of this much is considered significant

    :type rngSeed: int
    :param rngSeed: the random seed to use for the number generator
    """
    rng = numpy.random.RandomState(rngSeed)

    datasets, inputs, outputs, max_batch_size = ClassificationUtils.load_data(dataset, rogueClasses=rogueClasses)

    assert numpy.prod(imageShape) == inputs

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    classifier = LenetClassifier(x,
                                 rng,
                                 imageShape=imageShape,
                                 imageShapeOrder=imageShapeOrder,
                                 nkerns=nkerns,
                                 n_out=outputs,
                                 poolsize=poolsize,
                                 filtersize=filtersize,
                                 n_hidden=n_hidden)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # the cost we minimize during training is the NLL of the model
    cost = classifier.cost(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = classifier.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):

            current_iter = (epoch - 1) * n_train_batches + minibatch_index

            if current_iter % 100 == 0:
                print 'training @ iter = ', current_iter
            cost_ij = train_model(minibatch_index)

            if (current_iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, current_iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = current_iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                        ]
                    test_score = numpy.mean(test_losses)

                    # save the best model
                    modelStoreFilePathFullTemp = os.path.join(experimentStoreFolder, 'best_model.pkl')
                    with open(modelStoreFilePathFullTemp, 'wb') as f:
                        cPickle.dump(classifier, f)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= current_iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def convolutional_mlp_parameterized(featureParameters, datasetParameters, classifierParameters, forceRebuildModel=False):
    """
    Train a Logistic Regression model using the features, datset, and classifier parameters given

    :type featureParameters: dict
    :param featureParameters: parameters for the features

    :type datasetParameters: dict
    :param datasetParameters: parameters for the dataset

    :type classifierParameters: dict
    :param classifierParameters: parameters for the classifier

    :type forceRebuildModel: bool
    :param forceRebuildModel: forces to rebuild the model and train it again
    """

    assert classifierParameters['classifierType'] == 'ConvolutionalMLP', 'this config wasnt made for a Logistic Regression'

    datasetFile = CreateUtils.getDatasetFile(featureSetName=featureParameters['featureSetName'],datasetName= datasetParameters['datasetName'])

    experimentsFolder = CreateUtils.getExperimentFolder(featureParameters['featureSetName'],
                                                        datasetParameters['datasetName'],
                                                        classifierParameters['classifierType'],
                                                        classifierParameters['classifierSetName'])

    bestModelFilePath = os.path.join(experimentsFolder, 'best_model.pkl')
    if not os.path.exists(bestModelFilePath) or forceRebuildModel:
        evaluate_lenet5(dataset=datasetFile,
                        experimentStoreFolder=experimentsFolder,
                        learning_rate=classifierParameters['learning_rate'],
                        n_epochs=classifierParameters['n_epochs'],
                        batch_size=classifierParameters['batch_size'],
                        patience=classifierParameters['patience'],
                        patience_increase=classifierParameters['patience_increase'],
                        improvement_threshold=classifierParameters['improvement_threshold'],
                        rngSeed=classifierParameters['rngSeed'],
                        nkerns=classifierParameters['nkerns'],
                        imageShape=featureParameters['imageShape'],
                        imageShapeOrder=featureParameters['imageShapeOrder'],
                        poolsize=classifierParameters['poolsize'],
                        filtersize=classifierParameters['filtersize'],
                        n_hidden=classifierParameters['n_hidden'],
                        rogueClasses=classifierParameters['rogueClasses'],
                        )


if __name__ == '__main__':
    rawDataFolderMain = r"E:\\Users\\Joey\Documents\\Virtual Box Shared Folder\\"

    featureSetNameMain = 'DefaultPatchFeatures'
    datasetNameMain = 'staticLocations'
    classifierType = 'MLP'
    classifierSetNameMain = 'AllClassesDefault'

    (featureParametersDefault,
     datasetParametersDefault,
     modelSetParametersDefault) = CreateUtils.getParameters(featureSetName=featureSetNameMain,
                                                            datasetName=datasetNameMain,
                                                            classifierType=classifierType,
                                                            classifierSetName=classifierSetNameMain)

    convolutional_mlp_parameterized(featureParametersDefault, datasetParametersDefault, modelSetParametersDefault, forceRebuildModel=True)
    experimentsFolderMain = ""
    statisticsFolderMain = ""
    logistic_sgd.makeStatisticsForModel(experimentsFolderMain, statisticsFolderMain, featureParametersDefault, datasetParametersDefault,
                                        modelSetParametersDefault)
