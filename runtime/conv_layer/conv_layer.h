#pragma once

namespace ConvLayer
{
//-----------------------------------------------------------------------------
// Enumerations
//-----------------------------------------------------------------------------
// Indices or regions
enum Region
{
  RegionSystem,
  RegionNeurons,
  RegionConvKernel,
  RegionInput,
  RegionSpikeRecording,
  RegionProfiler,
  RegionStatistics,
};

// Indexes of application words
enum AppWord
{
  AppWordZMask,
  AppWordOutputZStart,
  AppWordSpikeKey,
  AppWordFixedPointPosition,
  AppWordMax,
};

enum ProfilerTag
{
  ProfilerTagConvolveSpike,
  ProfilerTagConvolveImage,
  ProfilerTagUpdateNeurons,
};

// Indices of statistic words
enum StatWord
{
  StatWordInputBufferOverflows,
  StatWordTaskQueueFull,
  StatWordNumTimerEventOverflows,
  StatWordMax,
};

};  // namespace ConvLayer