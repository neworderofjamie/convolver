# Import modules
import logging
import struct

# Import classes
from rig_cpp_common.regions import Region

logger = logging.getLogger("pynn_spinnaker")

# ------------------------------------------------------------------------------
# ConvKernel
# ------------------------------------------------------------------------------
class ConvKernel(Region):
    #  Size of a single weight
    WeightBytes = 1

    def __init__(self, kernel_width, kernel_height, kernel_depth, stride):
        """Create a new convolution kernel region.

        Parameters
        ----------
        kernel_width : int
            width of convolution kernel
        kernel_height : int
            height of convolution kernel
        kernel_depth : int
            depth of convolution kernel
        stride : int
            stride of convolution process
        """
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_depth = kernel_depth
        self.stride = stride

        logger.debug("\t\tKernel width:%u, kernel height:%u, kernel depth:%u, stride:%u",
                     self.kernel_width, self.kernel_height, self.kernel_depth,
                     self.stride)

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, weights, fixed_point_pos):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------
        application_words: list
            list of words to write to application-specific
            area of system region

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        return 0

    def write_subregion_to_file(self, fp, weights, fixed_point_pos):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        application_words: list
            list of words to write to application-specific
            area of system region
        """
        # Write structure
        #fp.write(struct.pack("%uI" % (2 + len(application_words)),
        #                     self.timer_period_us,
        #                     self.simulation_ticks,
         #                    *application_words))

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------
    @property
    def kernel_dtcm_bytes(self):
        return (self.WeightBytes * self.kernel_width * self.kernel_height *
                self.kernel_depth) + self.WeightBytes
