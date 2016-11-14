# Import modules
import enum
import logging
import numpy as np
from rig import machine
import regions

# Import classes
from collections import defaultdict
from rig_cpp_common.regions import Profiler, Statistics, System
from rig_cpp_common.utils import Args

# Import functions
from rig_cpp_common.utils import load_regions

logger = logging.getLogger("convolver")

# ----------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------
class Regions(enum.IntEnum):
    """Region names, corresponding to those defined in `conv_layer.h`"""
    system = 0
    neurons = 1
    conv_kernel = 2
    input = 3
    profiler = 4
    statistics = 5

# ----------------------------------------------------------------------------
# Vertex
# ----------------------------------------------------------------------------
class Vertex(object):
    def __init__(self, vert_index, z_slice, parent_keyspace, weights):
        # Build child keyspace
        self.keyspace = parent_keyspace(vert_index=vert_index)

        # Cache weights
        self.weights = weights
        self.z_slice = z_slice

        # Create a temporary child keyspace to ensure the
        # z-field is large enough to contain all z-values
        max_z_keyspace = self.keyspace(z=self.z_slice.stop)

        max_weight = np.amax(np.fabs(weights))
        logger.debug("\t\t\t\tMax absolute kernel weight:%f", max_weight)


        # Get MSB for maximum weight
        max_msb = np.floor(np.log2(max_weight)) + 1

        # Calculate where the weight format fixed-point lies
        self.fixed_point_pos = (7 - int(max_msb))
        logger.debug("\t\t\t\tFixed point position %d",
                     self.fixed_point_pos)

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
    # Tag names, corresponding to those defined in conv_layer.h
    profiler_tag_names = {
        0:  "Convolve spike",
        1:  "Convolve image",
        2:  "Update neurons",
    }

    # Names of statistics
    statistic_names = (
        "input_buffer_overflows",
        "task_queue_full",
        "timer_event_overflows",
    )

    def __init__(self, start_vert_index, output_width, output_height,
                 padding, stride, weights, neuron_decay, neuron_threshold,
                 record_spikes, parent_keyspace, input_data,
                 vertex_applications, vertex_resources,
                 timer_period_us, sim_ticks):
         # Check blob shape - num samples, depth, width, height
        assert len(weights.shape) == 4

        # If we have input data, check it is in correct
        # format and matches convolution kernel
        if input_data is not None:
            assert len(input_data.shape) == 3
            assert input_data.shape[0] == weights.shape[2]

        # Create standard regions
        self.regions = {}
        self.regions[Regions.system] = System(timer_period_us, sim_ticks)
        self.regions[Regions.neurons] =\
            regions.Neurons(output_width, output_height,
                            neuron_decay, neuron_threshold, record_spikes,
                            sim_ticks)
        self.regions[Regions.conv_kernel] =\
            regions.ConvKernel(weights.shape[0], weights.shape[1],
                               weights.shape[2], stride)
        self.regions[Regions.input] = regions.Input(input_data, padding)

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
        neuron_bytes = self.regions[Regions.neurons].output_dim_dtcm_bytes

        # Calculate memory required for a single kernel
        kernel_bytes = self.regions[Regions.conv_kernel].kernel_dtcm_bytes

        # Calculate memory required for any input data
        input_bytes = self.regions[Regions.input].dtcm_bytes


        total_bytes = neuron_bytes + kernel_bytes + input_bytes
        logger.debug("\t\tDTCM - neurons:%u bytes, kernel:%u bytes, input:%u bytes, total:%u bytes",
                     neuron_bytes, kernel_bytes, input_bytes, total_bytes)

        # Calculate how many of these kernels can fit on each core
        num_kernels_per_core = (64 * 1024) // total_bytes
        logger.debug("\t\t%u kernels per core", num_kernels_per_core)

        kernel_width = self.regions[Regions.conv_kernel].kernel_width
        kernel_height = self.regions[Regions.conv_kernel].kernel_height
        vertex_application = ("binaries/convolution_neuron_%ux%u.aplx" %
                              (kernel_width, kernel_height))
        logger.debug("\t\tApplication: %s", vertex_application)

        # Loop through slices of kernels to assign to each core
        self.vertices = []
        for vert_index, z_slice_start in enumerate(range(0, weights.shape[3], num_kernels_per_core)):
            z_slice_stop = min(z_slice_start + num_kernels_per_core, weights.shape[3])
            z_slice = slice(z_slice_start, z_slice_stop)

            logger.debug("\t\t\tVertex %u: z slice: [%u, %u)",
                         vert_index, z_slice_start, z_slice_stop)

            # Create vertex
            v = Vertex(vert_index + start_vert_index, z_slice,
                       parent_keyspace, weights[:,:,:,z_slice])

            # Add vertex to list
            self.vertices.append(v)

            # Add application
            kernel_width = self.regions[Regions.conv_kernel].kernel_width
            kernel_height = self.regions[Regions.conv_kernel].kernel_height
            vertex_applications[v] = vertex_application

            # Add resources
            # **NOTE** SDRAM needs are minimal so don't bother
            vertex_resources[v] = {machine.Cores: 1}

        logger.debug("\t\t%u vertices", len(self.vertices))

    # ----------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------
    def read_recorded_spikes(self, machine_controller):
        region = self.regions[Regions.neurons]
        return np.concatenate(
            [region.read_recorded_spikes(v.z_slice,
                                         v.region_memory[Regions.neurons])
            for v in self.vertices], axis=3)

    def load(self, placements, allocations, machine_controller, z_mask):
        # Loop through vertices
        for v in self.vertices:
            # Get placement and allocation
            vertex_placement = placements[v]
            vertex_allocation = allocations[v]

            # Get core this vertex should be run on
            core = vertex_allocation[machine.Cores]
            assert (core.stop - core.start) == 1

            logger.debug("\t\t\tVertex %s (%u, %u, %u)",
                            v, vertex_placement[0], vertex_placement[1],
                            core.start)

            # Select placed chip
            with machine_controller(x=vertex_placement[0],
                                    y=vertex_placement[1]):
                # Create region arguments
                region_arguments = defaultdict(Args)

                # Add kwargs for regions that require them
                region_arguments[Regions.system].kwargs["application_words"] =\
                    [z_mask, v.z_slice.start, v.routing_key, v.fixed_point_pos]

                # Add neurons region kwargs
                region_arguments[Regions.neurons].kwargs["output_depth"] =\
                    (v.z_slice.stop - v.z_slice.start)
                region_arguments[Regions.neurons].kwargs["fixed_point_pos"] =\
                    v.fixed_point_pos

                # Add conv kernel region kwargs
                region_arguments[Regions.conv_kernel].kwargs["weights"] =\
                    v.weights
                region_arguments[Regions.conv_kernel].kwargs["fixed_point_pos"] =\
                    v.fixed_point_pos

                # Load regions
                v.region_memory = load_regions(self.regions, region_arguments,
                                               machine_controller, core,
                                               logger)


