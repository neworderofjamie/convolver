# Import modules
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# ----------------------------------------------------------------------------
# ConvolutionNeuronLayer
# ----------------------------------------------------------------------------
# Layer consisting of a layer of convolutional
# weights followed by a neural non-linearity
class ConvolutionNeuronLayer(object):
    # How large are weights (used to store kernel) and
    WeightBytes = 1
    StateBytes = 2

    def __init__(self, output_width, output_height, padding, stride,
                 weights):
        # Cache dimensions
        self.output_width = output_width
        self.output_height = output_height
        self.padding = padding
        self.stride = stride

        logger.debug("\t\tOutput width:%u, output height:%u, padding:%u, stride:%u",
                     self.output_width, self.output_height, self.padding, self.stride)

        # Check blob shape - num samples, depth, width, height
        assert len(weights.shape) == 4

        # Cache kernels
        self.weights = weights
        logger.debug("\t\tMin kernel weight:%f, max kernel weight:%f, mean kernel weight:%f",
                     np.amin(self.weights), np.amax(self.weights), np.average(self.weights))

        logger.debug("\t\tKernel width:%u, kernel height:%u, kernel depth:%u, Num kernels:%u",
                     *self.weights.shape)

        # Check bias shape
        '''
        bias_shape = params[1].data.shape
        assert len(bias_shape) == 1
        assert bias_shape[0] == self.output_depth

        # Cache bias
        # **NOTE** these are repeated across every pixel
        self.bias = params[1].data
        '''
        # Calculate memory required for the neurons driven by a single kernel
        neuron_bytes = self.StateBytes * self.output_width * self.output_height

        # Calculate memory required for a single kernel
        kernel_bytes = (self.WeightBytes * self.weights.shape[0] * self.weights.shape[1] *
                        self.weights.shape[2])

        total_bytes = neuron_bytes + kernel_bytes + self.WeightBytes
        logger.debug("\t\tDTCM - neurons:%u bytes, kernel:%u bytes, total:%u bytes",
                     neuron_bytes, kernel_bytes, total_bytes)

        num_cores = np.ceil(float(total_bytes * self.weights.shape[3]) / (64 * 1024))
        logger.debug("\t\t%u cores", num_cores)

