# Import modules
import logging
import numpy as np
import struct

# Import classes
from rig.type_casts import NumpyFloatToFixConverter
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
            self.fixed_point_pos = (7 - int(max_msb))

            logger.debug("\t\tInput fixed-point position:%d, data channels:%u, width:%u, height:%u",
                        self.fixed_point_pos, *self.input_data.shape)

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self):
        """Get the size requirements of the region in bytes.

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # If there's no input data, region will just contain a zero
        if self.input_data is None:
            return 4
        # Otherwise, count, num channels, width, height and image data
        else:
            return 16 + self.dtcm_bytes


    def write_subregion_to_file(self, fp):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        """
        # If there is no input data, write a zero
        if self.input_data is None:
            fp.write(struct.pack("I", 0))
        # Otherwise
        else:
            # Write header
            fp.write(struct.pack("4I", 1, *self.input_data.shape))

            # Write input data
            convert = NumpyFloatToFixConverter(signed=True, n_bits=8,
                                               n_frac=self.fixed_point_pos)
            fp.write(convert(self.input_data).tostring())

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
