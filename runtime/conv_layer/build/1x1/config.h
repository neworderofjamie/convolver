#pragma once

// Model includes
#include "../../input_buffer.h"
#include "../../intrinsic_plasticity_models/stub.h"
#include "../../neuron_models/if_cond.h"
#include "../../synapse_models/exp.h"

namespace NeuronProcessor
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef NeuronModels::IFCond Neuron;
typedef SynapseModels::Exp Synapse;
typedef IntrinsicPlasticityModels::Stub IntrinsicPlasticity;

typedef InputBufferBase<uint32_t> InputBuffer;
};