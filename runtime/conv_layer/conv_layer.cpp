#include "conv_layer.h"

// Rig CPP common includes
#include "rig_cpp_common/config.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/profiler.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/statistics.h"
#include "rig_cpp_common/utils.h"

// Conv layer includes
#include "conv_kernel.h"
#include "input.h"
#include "neurons.h"

// Configuration include
#include "config.h"

// Namespaces
using namespace Common;
using namespace Common::Utils;
using namespace ConvLayer;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
enum DMATag
{
  DMATagSpikeRecordingWrite,
};

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Config g_Config;
SpikeInputBuffer g_SpikeInputBuffer;
Statistics<StatWordMax> g_Statistics;
ConvKernel g_ConvKernel;
Neurons g_Neurons;
Input g_Input;

uint32_t g_AppWords[AppWordMax];

uint g_Tick = 0;

bool g_PacketPipelineBusy = false;

//-----------------------------------------------------------------------------
// Module functions
//-----------------------------------------------------------------------------
bool ReadSDRAMData(uint32_t *baseAddress, uint32_t flags)
{
  // Verify data header
  if(!g_Config.VerifyHeader(baseAddress, flags))
  {
    return false;
  }

  // Read system region
  if(!g_Config.ReadSystemRegion(
    Config::GetRegionStart(baseAddress, RegionSystem),
    flags, AppWordMax, g_AppWords))
  {
    return false;
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tZ mask:%08x, z start:%u, spike key:%08x, fixed point position:%u",
      g_AppWords[AppWordZMask], g_AppWords[AppWordOutputZStart],
      g_AppWords[AppWordSpikeKey], g_AppWords[AppWordFixedPointPosition]);
  }

  // Read conv kernel region
  if(!g_Neurons.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionNeurons), flags))
  {
    return false;
  }

  // Read conv kernel region
  if(!g_ConvKernel.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionConvKernel), flags))
  {
    return false;
  }

  // Read input region
  if(!g_Input.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionInput), flags))
  {
    return false;
  }

  // Read profiler region
  if(!Profiler::ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionProfiler),
    flags))
  {
    return false;
  }

  if(!g_Statistics.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionStatistics),
    flags))
  {
    return false;
  }

  return true;
}

//-----------------------------------------------------------------------------
// Event handler functions
//-----------------------------------------------------------------------------
void MCPacketReceived(uint key, uint)
{
  LOG_PRINT(LOG_LEVEL_TRACE, "Received spike %x at tick %u, packet pipeline busy = %u",
            key, g_Tick, g_PacketPipelineBusy);

  // If there was space to add spike to incoming spike queue
  if(g_SpikeInputBuffer.Push(key))
  {
    // If the packet pipeline is not already busy, start processing
    if(!g_PacketPipelineBusy)
    {
      spin1_trigger_user_event(0, 0);
      g_PacketPipelineBusy = true;
    }
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Cannot add spike to input buffer");
    g_Statistics[StatWordInputBufferOverflows]++;
  }
}
//-----------------------------------------------------------------------------
void DMATransferDone(uint, uint tag)
{
  LOG_PRINT(LOG_LEVEL_TRACE, "DMA transfer done tag:%u", tag);

  // If recording write back is complete, reset recording for next timestep
  if(tag == DMATagSpikeRecordingWrite)
  {
    g_Neurons.ResetRecording();
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "DMA transfer done with unknown tag %u", tag);
  }
}
//-----------------------------------------------------------------------------
void UserEvent(uint, uint)
{
  // Lambda function to add input current to neurons
  auto applyInput =
    [](unsigned int xNeuron, unsigned int yNeuron, unsigned int zNeuron,
       int input)
    {
      g_Neurons.AddInputCurrent(xNeuron, yNeuron, zNeuron, input);
    };

  // While there are spikes in input queue
  uint32_t spikeKey;
  while(g_SpikeInputBuffer.Pop(spikeKey))
  {
    // Extract x, y and z from spike
    // **THINK** if z was at bottom of key it be used to route
    const unsigned int xIn = (spikeKey & 0xFF);
    const unsigned int yIn = (spikeKey >> 8) & 0xFF;
    const unsigned int zIn = (spikeKey >> 16) & g_AppWords[AppWordZMask];

    // Convolve spike with convolution kernel
    g_ConvKernel.ConvolveSpike(xIn, yIn, zIn, applyInput);
  }

  // Pipeline no longer busy
  g_PacketPipelineBusy = false;
}
//-----------------------------------------------------------------------------
void TimerTick(uint tick, uint)
{
  // Cache tick
  // **NOTE** ticks start at 1
  g_Tick = (tick - 1);

  // If a fixed number of simulation ticks are specified and these have passed
  if(g_Config.GetSimulationTicks() != UINT32_MAX
    && g_Tick >= g_Config.GetSimulationTicks())
  {
    LOG_PRINT(LOG_LEVEL_INFO, "Simulation complete");

    // Finalise profiling
    Profiler::Finalise();

    // Copy diagnostic stats out of spin1 API
    g_Statistics[StatWordTaskQueueFull] = diagnostics.task_queue_full;
    g_Statistics[StatWordNumTimerEventOverflows] = diagnostics.total_times_tick_tic_callback_overran;

    // Finalise statistics
    g_Statistics.Finalise();

    // Exit simulation
    spin1_exit(0);
  }
  // Otherwise
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u", g_Tick);

    // If this vertex has any input to apply
    if(g_Input.HasInput())
    {
      // Lambda function to read input pixels
      auto getPixel =
        [](unsigned int x, unsigned int y)
        {
          return g_Input.GetPixel(x, y);
        };

      // Lambda function to add input current to neurons
      auto applyInput =
        [](unsigned int xNeuron, unsigned int yNeuron, unsigned int zNeuron,
          int input)
        {
          g_Neurons.AddInputCurrent(xNeuron, yNeuron, zNeuron, input);
        };

      // Convolve input image pixels, read using lambda function with kernel
      g_ConvKernel.ConvolveImage(g_Input.GetWidth(), g_Input.GetHeight(), g_Input.GetFixedPointPosition(),
        applyInput, getPixel);
    }

    // Lambda function to emit spike from specified neuron
    auto emitSpike =
      [](unsigned int x, unsigned int y, unsigned int z)
      {
        // Build neuron ID from x, y and z (offset by this core's starting slice)
        const unsigned int zOut = g_AppWords[AppWordOutputZStart] + z;
        const uint32_t n = (zOut  << 16) | (y << 8) | x;

        // Send spike
        while(!spin1_send_mc_packet(g_AppWords[AppWordSpikeKey] | n, 0, NO_PAYLOAD))
        {
          spin1_delay_us(1);
        }
      };

    // Update neural state using lambda function to emit spikes
    g_Neurons.Update(emitSpike, g_AppWords[AppWordFixedPointPosition]);

    // Write spike recording data to SDRAM
    g_Neurons.TransferBuffer(DMATagSpikeRecordingWrite);
  }
}
} // Anonymous namespace

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
extern "C" void c_main()
{
  // Get this core's base address using alloc tag
  uint32_t *baseAddress = Config::GetBaseAddressAllocTag();

  // If reading SDRAM data fails
  if(!ReadSDRAMData(baseAddress, 0))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Error reading SDRAM data");
    rt_error(RTE_ABORT);
    return;
  }

  // Set timer tick (in microseconds) in both timer and
  spin1_set_timer_tick(g_Config.GetTimerPeriod());

  // Register callbacks
  spin1_callback_on(MC_PACKET_RECEIVED, MCPacketReceived, -1);
  spin1_callback_on(DMA_TRANSFER_DONE,  DMATransferDone,   0);
  spin1_callback_on(USER_EVENT,         UserEvent,         0);
  spin1_callback_on(TIMER_TICK,         TimerTick,         2);

  // Start simulation
  spin1_start(SYNC_WAIT);
}