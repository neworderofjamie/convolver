# Import modules
import logging
import math
import struct

# Import classes
from rig_cpp_common.regions import Region

# Import functions
from rig.type_casts import float_to_fp

logger = logging.getLogger("convolver")

def calc_bitfield_words(bits):
    return int(math.ceil(float(bits) / 32.0))

# ------------------------------------------------------------------------------
# Neurons
# ------------------------------------------------------------------------------
class Neurons(Region):
    # How many bytes does the state of each neuron require
    StateBytes = 2

    def __init__(self, output_width, output_height, decay, threshold,
                 record_spikes, sim_ticks):
        """Create a new neurons region.

        Parameters
        ----------
        output_width : int
            width of 3D output volume of neurons
        output_height : int
            height of 3D output volume of neurons
        """
        self.output_width = output_width
        self.output_height = output_height
        self.decay = decay
        self.threshold = threshold
        self.record_spikes = record_spikes
        self.sim_ticks = sim_ticks

        logger.debug("\t\tOutput width:%u, output height:%u, neuron decay:%f, neuron threshold:%f",
                     self.output_width, self.output_height,
                     self.decay, self.threshold)

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, output_depth, fixed_point_pos):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------
        output_depth : int
            depth of 3D output volume of neurons

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # If we're recording
        if self.record_spikes:
            # Calculate size of bitfield required to
            # represent one time step of spiking output
            num_neurons = self.output_width * self.output_height * output_depth
            recording_words = calc_bitfield_words(num_neurons)

            recording_bytes = (recording_words * 4 * self.sim_ticks)
        else:
            recording_bytes = 0

        return (6 * 4) + recording_bytes

    def write_subregion_to_file(self, fp, output_depth, fixed_point_pos):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        output_depth : int
            depth of 3D output volume of neurons
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        """
        # Create converter to correct fixed point format
        convert = float_to_fp(signed=True, n_bits=32, n_frac=fixed_point_pos)

        # Write structure
        fp.write(struct.pack("4I2i",
                             self.output_width, self.output_height,
                             output_depth, 1 if self.record_spikes else 0,
                             convert(self.threshold), convert(self.decay)))

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def output_dim_dtcm_bytes(self):
        num_dim_neurons = self.output_width * self.output_height

        # If we're recording calculate the number of bytes required to
        # store the spiking output of this dimension for one timestep
        # **NOTE** this is an overestimate
        if self.record_spikes:
            recording_bytes = calc_bitfield_words(num_dim_neurons) * 4
        else:
            recording_bytes = 0

        return (self.StateBytes * num_dim_neurons) + recording_bytes
