# Import modules
import logging
import struct

# Import classes
from rig_cpp_common.regions import Region

# Import functions
from rig.type_casts import float_to_fp

logger = logging.getLogger("convolver")

# ------------------------------------------------------------------------------
# Neurons
# ------------------------------------------------------------------------------
class Neurons(Region):
    # How many bytes does the state of each neuron require
    StateBytes = 2

    def __init__(self, output_width, output_height):
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

        logger.debug("\t\tOutput width:%u, output height:%u",
                     self.output_width, self.output_height)

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, output_depth, threshold, decay, fixed_point_pos):
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
        return (5 * 4)

    def write_subregion_to_file(self, fp, output_depth, threshold, decay,
                                fixed_point_pos):
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
        fp.write(struct.pack("3I2i", self.output_width, self.output_height,
                             output_depth, convert(threshold),
                             convert(decay)))

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def output_dim_dtcm_bytes(self):
        return self.StateBytes * self.output_width * self.output_height
