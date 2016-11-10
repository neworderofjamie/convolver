# Import modules
import logging
import numpy as np
from rig import machine

logger = logging.getLogger("convolver")

# ----------------------------------------------------------------------------
# Vertex
# ----------------------------------------------------------------------------
class Vertex(object):
    def __init__(self, layer_index, vert_index, z_slice, parent_keyspace,
                 weights):
        # Build child keyspace
        self.keyspace = parent_keyspace(layer_index=layer_index,
                                        vert_index=vert_index)

        # Cache weights
        self.weights = weights
        self.z_slice = z_slice

        min_weight = np.amin(weights)
        max_weight = np.amax(weights)
        logger.debug("\t\t\t\tMin kernel weight:%f, max kernel weight:%f, mean kernel weight:%f",
                     min_weight, max_weight, np.average(weights))


        # Get MSB for maximum weight
        max_msb = np.floor(np.log2(max_weight)) + 1

        # If minimum weight isn't zero
        if min_weight != 0.0:
            # Get MSB of minimum weight
            min_msb = np.floor(np.log2(min_weight)) + 1

            # Check there's enough bits to represent this range
            if (max_msb - min_msb) >= 7:
                logger.warn("Insufficient range in 7-bit weight to represent "
                            "minimum weight:%f and maximum weight:%f",
                            min_weight, max_weight)

        # Calculate where the weight format fixed-point lies
        self.fixed_point_position = (7 - int(max_msb))
        logger.debug("\t\t\t\tFixed point position %u",
                     self.fixed_point_position)

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def routing_key(self):
        return self.keyspace.get_value(tag="routing")

    @property
    def routing_mask(self):
        return self.keyspace.get_value(tag="routing")

# ----------------------------------------------------------------------------
# ConvNeuronLayer
# ----------------------------------------------------------------------------
# Layer consisting of a layer of convolutional
# weights followed by a neural non-linearity
class ConvNeuronLayer(object):
    # How large are weights (used to store kernel) and
    WeightBytes = 1
    StateBytes = 2

    def __init__(self, layer_index, output_width, output_height, padding, stride,
                 weights, parent_keyspace):
        # Cache dimensions
        self.output_width = output_width
        self.output_height = output_height
        self.padding = padding
        self.stride = stride

        logger.debug("\t\tOutput width:%u, output height:%u, padding:%u, stride:%u",
                     self.output_width, self.output_height, self.padding, self.stride)

        # Check blob shape - num samples, depth, width, height
        assert len(weights.shape) == 4

        logger.debug("\t\tKernel width:%u, kernel height:%u, kernel depth:%u, Num kernels:%u",
                     *weights.shape)
        self.kernel_width = weights.shape[0]
        self.kernel_height = weights.shape[1]

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
        kernel_bytes = (self.WeightBytes * self.kernel_width * self.kernel_height *
                        weights.shape[2])

        total_bytes = neuron_bytes + kernel_bytes + self.WeightBytes
        logger.debug("\t\tDTCM - neurons:%u bytes, kernel:%u bytes, total:%u bytes",
                     neuron_bytes, kernel_bytes, total_bytes)

        # Calculate how many of these kernels can fit on each core
        num_kernels_per_core = (64 * 1024) // total_bytes
        logger.debug("\t\t%u kernels per core", num_kernels_per_core)

        # Loop through slices of kernels to assign to each core
        self.vertices = []
        for vert_index, z_slice_start in enumerate(range(0, weights.shape[3], num_kernels_per_core)):
            z_slice_stop = min(z_slice_start + num_kernels_per_core, weights.shape[3])
            z_slice = slice(z_slice_start, z_slice_stop)

            logger.debug("\t\t\tVertex %u: z slice: [%u, %u)",
                         vert_index, z_slice_start, z_slice_stop)

            # Build new vertex and add to list
            self.vertices.append(Vertex(layer_index, vert_index, z_slice,
                                        parent_keyspace, weights[:,:,:,z_slice]))

        logger.debug("\t\t%u vertices", len(self.vertices))

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def add_apps_and_resources(self, vertex_applications, vertex_resources):
        # Loop through vertices
        for v in self.vertices:
            # Add application
            vertex_applications[v] = ("binaries/convolution_neuron_%ux%u.aplx" %
                                      (self.kernel_width, self.kernel_height))

            # Add resources
            # **NOTE** SDRAM needs are minimal so don't bother
            vertex_resources[v] = {machine.Cores: 1}