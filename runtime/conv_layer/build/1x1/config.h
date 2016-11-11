#pragma once

// Rig CPP common
#include "rig_cpp_common/circular_buffer.h"

namespace ConvLayer
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef CircularBuffer<uint32_t, 256> SpikeInputBuffer;
typedef ConvKernelBase<int8_t, 3> ConvKernel;
typedef NeuronsBase<int16_t> Neurons;
};