# Import modules
import logging
import numpy as np
import os

# Import classes
from conv_net import ConvNet
from conv_neuron_layer import ConvNeuronLayer
from neon.layers import Conv, Dropout, Activation, Pooling, GeneralizedCost
from neon.initializers import Gaussian
from neon.transforms import Rectlin, Softmax
from neon.models import Model
from neon.data import CIFAR10
from neon.util.argparser import NeonArgparser

# Import functions
from six import iterkeys

logger = logging.getLogger("convolver")
logger.setLevel(logging.DEBUG)

parser = NeonArgparser(__doc__)
args = parser.parse_args()

'''
dataset = CIFAR10(path="~/nervana/data",
                  normalize=False,
                  contrast_normalize=True,
                  whiten=True,
                  pad_classes=True)
valid_set = dataset.valid_iter
'''
conv_net = ConvNet()

relu = Rectlin()
init_uni = Gaussian(scale=0.05)
conv = dict(init=init_uni, batch_norm=False, activation=relu)
convp1 = dict(init=init_uni, batch_norm=False, activation=relu, padding=1)
convp1s2 = dict(init=init_uni, batch_norm=False,
                activation=relu, padding=1, strides=2)

layers = [Dropout(keep=.8),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1),
          Conv((3, 3, 96), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1),
          Conv((3, 3, 192), **convp1s2),
          Dropout(keep=.5),
          Conv((3, 3, 192), **convp1),
          Conv((1, 1, 192), **conv),
          Conv((1, 1, 16), **conv),
          Pooling(8, op="avg"),
          Activation(Softmax())]

mlp = Model(layers=layers)
mlp.load_params("~/neon/examples/cifar10_allcnn_e350.p")

model_description = mlp.get_description(get_weights=True)
layers = model_description["model"]["config"]["layers"]

def ignore_dropout(layers, input_dims):
    if layers[0]["type"] == "neon.layers.layer.Dropout":
        logger.debug("\tIgnoring dropout layer:%s",
                     layers[0]["config"]["name"])
        return 1
    else:
        return 0

def convolution_neuron_layer(layers, input_dims):
    # If there aren't at least two more layers then there
    # can't be a convolution layer followed by a neuron layer!
    if len(layers) < 2:
        return 0

    # If first layer is a convolution layer and next is an activation layer
    if (layers[0]["type"] == "neon.layers.layer.Convolution" and
        layers[1]["type"] == "neon.layers.layer.Activation"):
        conv_config = layers[0]["config"]
        activation_config = layers[1]["config"]

        logger.info("\tBuilding ConvolutionNeuronLayer from convolution layer:%s and activation layer:%s",
            conv_config["name"], activation_config["name"])

        stride = conv_config["strides"] if "strides" in conv_config else 1
        padding = conv_config["padding"] if "padding" in conv_config else 0

        conv_f_shape = conv_config["fshape"]

        # Reshape weights into kernel_width, kernel_height, kernel_depth, num_filters
        weights = layers[0]["params"]["W"]
        assert conv_f_shape[0] * conv_f_shape[1] *  input_dims[2] == weights.shape[0]
        weights = np.reshape(weights, (conv_f_shape[0], conv_f_shape[1], input_dims[2], weights.shape[1]))


        # Apply stride and padding equation to input dimensions calculate output dimensions
        input_dims[0] = ((input_dims[0] - conv_f_shape[0] + (2 * padding)) // stride) + 1
        input_dims[1] = ((input_dims[1] - conv_f_shape[1] + (2 * padding)) // stride) + 1
        input_dims[2] = weights.shape[3]

        # Check activation function
        if activation_config["transform"]["type"] != "neon.transforms.activation.Rectlin":
            logger.warn("Only RectLin activation functions are supported not %s",
                        activation_config["transform"]["type"])

        # Add layer to conv net
        conv_net.layers.append(
            ConvNeuronLayer(layer_index=len(conv_net.layers),
                            output_width=input_dims[0],
                            output_height=input_dims[1],
                            padding=padding, stride=stride,
                            weights=weights,
                            parent_keyspace=conv_net.keyspace))
        return 2
    else:
        return 0

layer_processing_functions = [ignore_dropout,
                              convolution_neuron_layer]

def process_layers(layers, input_dims):
    # Loop through layer processing functions
    for l in layer_processing_functions:
        # Call layer processing function with layers
        n_input_layers_processed = l(layers, input_dims)

        if n_input_layers_processed != 0:
            return n_input_layers_processed

    return 0

logger.info("Layers")
input_dims = [32, 32, 3]
l = 0
while l < len(layers):
    # Slice out unprocessed layers of network
    subsequent_layers = layers[l:]

    # Attempt to map to output layers
    n_input_layers_processed = process_layers(
        subsequent_layers, input_dims)

    # If we failed to process any input layers
    if n_input_layers_processed == 0:
        logger.warn("Cannot map input layer name:%s, type:%s to SpiNNaker",
                    subsequent_layers[0]["config"]["name"], subsequent_layers[0]["type"])
        l += 1
    # Otherwise add newly created output layers to list
    else:
        l += n_input_layers_processed

conv_net.run("192.168.1.1")