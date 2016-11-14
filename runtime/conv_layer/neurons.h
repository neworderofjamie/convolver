#pragma once

// Rig CPP common includes
#include "rig_cpp_common/arm_intrinsics.h"
#include "rig_cpp_common/bit_field.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"

// Namespaces
using namespace Common;
using namespace Common::ARMIntrinsics;

//-----------------------------------------------------------------------------
// ConvLayer::NeuronsBase
//-----------------------------------------------------------------------------
namespace ConvLayer
{
template<typename State>
class NeuronsBase
{
public:
  NeuronsBase() : m_MembraneVoltage(NULL), m_Width(0), m_Height(0), m_Depth(0)
  {
  }

  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *region, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "NeuronsBase::ReadSDRAMData");

    // Read neuron slice dimensions
    m_Width = *region++;
    m_Height = *region++;
    m_Depth = *region++;
    LOG_PRINT(LOG_LEVEL_INFO, "\tWidth:%u, height:%u, depth:%u",
              m_Width, m_Height, m_Depth);

    // Read recording flag
    const bool record = (*region++ != 0);

    m_ThresholdVoltage = *reinterpret_cast<int32_t*>(region++);
    m_Decay = *reinterpret_cast<int32_t*>(region++);
    LOG_PRINT(LOG_LEVEL_INFO, "\tDecay:%d, threshold:%d",
              m_ThresholdVoltage, m_Decay);

    // Attempt to allocate memory for membrane voltages
    const unsigned int numNeurons = m_Width * m_Height * m_Depth;
    const unsigned int membraneVoltageBytes = numNeurons * sizeof(State);
    m_MembraneVoltage = (State*)spin1_malloc(membraneVoltageBytes);
    if(m_MembraneVoltage == NULL)
    {
      LOG_PRINT(LOG_LEVEL_INFO, "\tFailed to allocate %u bytes for membrane voltages",
                membraneVoltageBytes);

      return false;
    }
    else
    {
      LOG_PRINT(LOG_LEVEL_INFO, "\tAllocated %u bytes for membrane voltages",
                membraneVoltageBytes);

      // Zero membrane voltages
      State *membraneVoltage = m_MembraneVoltage;
      for(unsigned int n = 0; n < numNeurons; n++)
      {
        *membraneVoltage++ = 0;
      }
    }

    // If recording is enabled
    if(record)
    {
      // Cache pointer to rest of region to use for recording
      m_RecordingSDRAM = region;

      // Calculate number of recording words
      m_NumRecordingWords = BitField::GetWordSize(numNeurons);
      LOG_PRINT(LOG_LEVEL_INFO, "\tRecording using %u word bitfield",
                m_NumRecordingWords);

      // Allocate recording buffer
      m_RecordingBuffer = (uint32_t*)spin1_malloc(m_NumRecordingWords * sizeof(uint32_t));
      if(m_RecordingBuffer == NULL)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate local record buffer");
        return false;
      }

      // Zero recording buffer
      ResetRecording();
    }
    // Otherwise NULL all recording structures
    else
    {
      m_NumRecordingWords = 0;
      m_RecordingSDRAM = NULL;
      m_RecordingBuffer = NULL;
    }

    return true;
  }

  void AddInputCurrent(unsigned int x, unsigned int y, unsigned int z,
                       int inputCurrent)
  {
    // Calculate neuron index
    // **NOTE** n = z + depth * (y + (height * x))
    int32_t n = __smlabb((int32_t)x, (int32_t)m_Height, (int32_t)y);
    n = __smlabb(n, (int32_t)m_Depth, (int32_t)z);

    // Add input 'current' to it's 'voltage'
    m_MembraneVoltage[n] += inputCurrent;
  }

  template<typename E>
  void Update(E emitSpikeFunc, uint32_t fixedPointPosition)
  {
    // Loop through neuron volume
    // **THINK** might it be better to pad neurons to power of two and
    // have a single loop whose index is actually a valid spike key
    State *membraneVoltage = m_MembraneVoltage;
    for(unsigned int x = 0; x < m_Width; x++)
    {
      for(unsigned int y = 0; y < m_Height; y++)
      {
        for(unsigned int z = 0; z < m_Depth; z++)
        {
          int32_t neuronMembraneVoltage = *membraneVoltage;

          // If membrane voltage has crossed threshold
          if(neuronMembraneVoltage > m_ThresholdVoltage)
          {
            // Emit spike
            emitSpikeFunc(x, y, z);

            // If we're recording, set appropriate bit
            if(m_RecordingBuffer != NULL)
            {
              BitField::SetBit(m_RecordingBuffer, membraneVoltage - m_MembraneVoltage);
            }

            // Reset membrane voltage
            *membraneVoltage++ = 0;
          }
          else
          {
            // Decay membrane voltage
            neuronMembraneVoltage =  __smulbb(neuronMembraneVoltage, m_Decay);
            neuronMembraneVoltage >>= fixedPointPosition;

            // Update membrane voltage
            *membraneVoltage++ = neuronMembraneVoltage;
          }


        }
      }
    }
  }

  void ResetRecording()
  {
    if(m_RecordingBuffer != NULL)
    {
      BitField::Clear(m_RecordingBuffer, m_NumRecordingWords);
    }
  }

  void TransferBuffer(uint tag)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "\tTransferring record buffer to SDRAM:%08x",
      m_RecordingSDRAM);
#if LOG_LEVEL <= LOG_LEVEL_TRACE
    BitField::PrintBits(IO_BUF, m_RecordingBuffer, m_NumRecordingWords);
    io_printf(IO_BUF, "\n");
#endif

    // Use DMA to transfer record buffer to SDRAM
    if(m_NumRecordingWords > 0)
    {
      spin1_dma_transfer(tag, m_RecordingSDRAM,
                         m_RecordingBuffer, DMA_WRITE,
                         m_NumRecordingWords * sizeof(uint32_t));

      // Advance SDRAM pointer
      m_RecordingSDRAM += m_NumRecordingWords;
    }
  }

private:
  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Array of neuron's membrane voltage
  State *m_MembraneVoltage;

  // Neuron parameters
  int32_t m_ThresholdVoltage;
  int32_t m_Decay;

  // Neuron slice dimensions
  uint32_t m_Width;
  uint32_t m_Height;
  uint32_t m_Depth;

  uint32_t m_NumRecordingWords;
  uint32_t *m_RecordingBuffer;
  uint32_t *m_RecordingSDRAM;
};
} // ConvLayer