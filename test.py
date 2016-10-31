# Import modules
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import logging
import math
import numpy as np
from os import path

from convolution_neuron_layer import ConvolutionNeuronLayer

# Import functions
from six import iteritems

logger = logging.getLogger("convolver")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

neuron_state_bytes = 2


proto_file = "regularised_cifar_10.prototxt"
model_file = "regularised_cifar_10_iter_4000.caffemodel"

caffe.set_mode_cpu()
net = caffe.Net(proto_file, model_file, caffe.TEST)

# Load proto buffer directly
proto_net = caffe_pb2.NetParameter()
with open(proto_file, "r") as f:
    pb.text_format.Merge(f.read(), proto_net)

def find_proto_layer(proto_net, name):
    for layer in proto_net.layer:
        if layer.name == name:
            return layer

    return None

def read_repeated_scalar_field(value, default):
    if len(value) == 1:
        return value[0]
    elif len(value) == 0:
        return default
    else:
        assert False
'''
# ----------------------------------------------------------------------------
# ConvolutionNeuronLayer
# ----------------------------------------------------------------------------
# Layer consisting of a layer of convolutional
# weights followed by a neural non-linearity
class ConvolutionNeuronLayer(object):
    # How large are weights (used to store kernel) and
    WeightBytes = 1
    StateBytes = 2

    def __init__(self, blob, params, proto_layer):
        logger.info("\tConvolutionNeuronLayer")

        # Check blob shape - num samples, depth, width, height
        assert len(blob.data.shape) == 4

        # Cache blob parameters
        self.width = blob.data.shape[2]
        self.height = blob.data.shape[3]
        self.output_depth = blob.data.shape[1]

        logger.debug("\t\tWidth:%u, height:%u, output depth:%u",
                     self.width, self.height, self.output_depth)

        # Check parameter shape - num kernels and; kernel depth, width height
        param_shape = params[0].data.shape
        assert len(param_shape) == 4
        assert param_shape[0] == self.output_depth

        # Cache kernels
        self.kernels = params[0].data
        logger.debug("\t\tMin kernel weight:%f, Max kernel weight:%f, Mean kernel weight:%f",
                     np.amin(self.kernels), np.amax(self.kernels), np.average(self.kernels))

        logger.debug("\t\tNum kernels:%u, kernel width:%u, kernel height:%u, kernel depth:%u",
                     param_shape[0], param_shape[2], param_shape[3], param_shape[1])

        # Check bias shape
        bias_shape = params[1].data.shape
        assert len(bias_shape) == 1
        assert bias_shape[0] == self.output_depth

        # Cache bias
        # **NOTE** these are repeated across every pixel
        self.bias = params[1].data

        # Check proto layer parameters
        convolution_params = proto_layer.convolution_param
        assert convolution_params.num_output == self.output_depth
        assert convolution_params.kernel_size[0] == param_shape[2]
        assert convolution_params.kernel_size[0] == param_shape[3]

        # Cache pad and stride
        self.pad = read_repeated_scalar_field(convolution_params.pad, 0)
        self.stride = read_repeated_scalar_field(convolution_params.stride, 1)
        logger.debug("\t\tPad:%u, Stride:%u", self.pad, self.stride)

        # Calculate memory required for the neurons driven by a single kernel
        neuron_bytes = self.StateBytes * self.width * self.height

        # Calculate memory required for a single kernel
        kernel_bytes = (self.WeightBytes * param_shape[2] * param_shape[3] *
                        param_shape[1])

        total_bytes = neuron_bytes + kernel_bytes + self.WeightBytes
        logger.debug("\t\tDTCM - neurons:%u bytes, kernel:%u bytes, total:%u bytes",
                     neuron_bytes, kernel_bytes, total_bytes)

        num_cores = math.ceil(float(total_bytes * self.output_depth) / 65536.0)
        logger.debug("\t\t%u cores", num_cores)
'''
def convolution_neuron_layer(layers, layer_names,
                             blobs, params, proto_net):
    # If there aren't at least two more layers then there
    # can't be a convolution layer followed by a neuron layer!
    if len(layers) < 2:
        return 0, None

    # If first layer is a convolution layer and next is a ReLU
    # **TODO** other types of neural non-linearity
    if layers[0].type == "Convolution" and layers[1].type == "ReLU":
        # Find parameters for convolution layer
        # **NOTE** ReLU layer is parameterless
        conv_proto_layer = find_proto_layer(proto_net, layer_names[0])
        conv_blob = blobs[layer_names[0]]
        conv_params = params[layer_names[0]]

        # Check proto layer parameters
        conv_proto_params = conv_proto_layer.convolution_param
        assert conv_proto_params.num_output == self.output_depth
        assert conv_proto_params.kernel_size[0] == param_shape[2]
        assert conv_proto_params.kernel_size[0] == param_shape[3]


        return 2, [ConvolutionNeuronLayer(output_width=blob.data.shape[2],
                                          output_height=blob.data.shape[3],
                                          padding=read_repeated_scalar_field(conv_proto_params.pad, 0),
                                          stride=read_repeated_scalar_field(conv_proto_params.stride, 1),
                                          weights=conv_params[0].data)]
    else:
        return 0, None

logger.info("Layers")
output_layers = []
l = 0
while l < len(net.layers):
    # Slice out unprocessed layers of network
    subsequent_layers = net.layers[l:]
    subsequent_layer_names = net._layer_names[l:]

    # Attempt to map to output layers
    n_input_layers_processed, new_output_layers =\
        convolution_neuron_layer(subsequent_layers, subsequent_layer_names,
                                 net.blobs, net.params, proto_net)

    # If we failed to process any input layers
    if n_input_layers_processed == 0:
        logger.warn("Cannot map input layer name:%s, type:%s to SpiNNaker",
                    subsequent_layer_names[0], subsequent_layers[0].type)
        l += 1
    # Otherwise add newly created output layers to list
    else:
        output_layers.extend(new_output_layers)
        l += n_input_layers_processed
