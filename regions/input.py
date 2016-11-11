# Import modules
import logging
import numpy as np
import struct

# Import classes
from rig_cpp_common.regions import Region


logger = logging.getLogger("convolver")

# ------------------------------------------------------------------------------
# Input
# ------------------------------------------------------------------------------
class Input(Region):
    # How many bytes are used to represent each input component
    InputBytes = 1

    def __init__(self, input_data):
        """Create a new input region.

        Parameters
        ----------
        input_data : ndarray
            array of input data
        """
        self.input_data = input_data

        if self.input_data is not None:
            # Calculate maximum absolute weight
            max_input = np.amax(np.fabs(self.input_data))

            # Get MSB for maximum weight
            max_msb = np.floor(np.log2(max_input)) + 1

            # Calculate where the weight format fixed-point lies
            self.input_fixed_point_pos = (7 - int(max_msb))

            logger.debug("\t\tInput fixed-point position:%d, data channels:%u, width:%u, height:%u",
                        self.input_fixed_point_pos, *self.input_data.shape)

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, application_words):
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
        # If input data is directly supplied to this layer
        if input_data is not None:
            assert len(input_data.shape) == 3
            assert input_data.shape[0] == weights.shape[2]

            # Calculate memory required for input
            input_bytes = (self.InputBytes * input_data.shape[0] *
                           input_data.shape[1] * input_data.shape[2])
        else:
            input_bytes = 0

    def write_subregion_to_file(self, fp, application_words):
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
    def dtcm_bytes(self):
        if self.input_data is None:
            return 0
        else:
            return (self.InputBytes * self.input_data.shape[0] *
                    self.input_data.shape[1] * self.input_data.shape[2])
