#pragma once

// Rig CPP common
#include "rig_cpp_common/circular_buffer.h"

// Conv layer includes
#include "../../conv_kernel.h"
#include "../../input.h"
#include "../../neurons.h"

namespace ConvLayer
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef CircularBuffer<uint32_t, 256> SpikeInputBuffer;
typedef ConvKernelBase<int8_t, 3> ConvKernel;
typedef InputBase<int8_t> Input:
typedef NeuronsBase<int16_t> Neurons;
}