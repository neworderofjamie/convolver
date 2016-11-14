#pragma once

// Standard includes
#include <tuple>

// Rig CPP common includes
#include "rig_cpp_common/arm_intrinsics.h"

// Namespaces
using namespace Common::ARMIntrinsics;

//--------------------------------------------------------------------------
// ConvLayer::ConvKernelBase
//--------------------------------------------------------------------------
namespace ConvLayer
{
template<typename Weight, unsigned int KernelSize>
class ConvKernelBase
{
private:
  //--------------------------------------------------------------------------
  // Constants
  //--------------------------------------------------------------------------
  static const unsigned int HalfKernelSize = KernelSize / 2;

public:
  //--------------------------------------------------------------------------
  // Public methods
  //--------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *region, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "ConvKernelBase::ReadSDRAMData");

    m_Stride = *region++;
    m_NumKernels = *region++;
    const uint32_t kernelDepth = *region++;

    LOG_PRINT(LOG_LEVEL_INFO, "\tStride:%u, num kernels:%u, kernel size:%u, kernel depth:%u",
      m_Stride, m_NumKernels, KernelSize, kernelDepth);

    // Allocate array to hold kernels
    const unsigned int kernelArrayBytes = m_NumKernels * sizeof(Weight*);
    m_KernelWeights = (Weight**)spin1_malloc(kernelArrayBytes);
    if(m_KernelWeights == NULL)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Cannot allocate %u byte array for kernels", kernelArrayBytes);
      return false;
    }

    // Loop through kernels
    uint8_t *kernelRegion = reinterpret_cast<uint8_t*>(region);
    const unsigned int kernelBytes = KernelSize * KernelSize * kernelDepth * sizeof(Weight);
    for(unsigned int k = 0; k < m_NumKernels; k++)
    {
      // Allocate kernel
      m_KernelWeights[k] = (Weight*)spin1_malloc(kernelBytes);
      if(m_KernelWeights[k] == NULL)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Cannot allocate %u bytes for kernel %u",
          kernelBytes, k);

        return false;
      }

      // Copy kernel into DTCM
      spin1_memcpy(m_KernelWeights[k], kernelRegion, kernelBytes);
      kernelRegion += kernelBytes;
    }

    return true;
  }

  template<typename A>
  void ConvolveSpike(int xIn, int yIn, int zIn,
    A applyFunc) const
  {
    const unsigned zStride = zIn * KernelSize;

    // Loop through kernel pixels
    // **TODO** stride
    for(int xKernel = 0; xKernel < (int)KernelSize; xKernel++)
    {
      for(int yKernel = 0; yKernel < (int)KernelSize; yKernel++)
      {
        // Calculate offset into kernel for this pixel
        const unsigned int kernelIndex = xKernel + (KernelSize * (yKernel + zStride));

        // Calculate corresponding output pixel
        const int xNeuron = xIn - xKernel + 1;
        const int yNeuron = yIn - yKernel + 1;

        // Loop through kernels and apply
        for(unsigned int k = 0; k < m_NumKernels; k++)
        {
          // Get current kernel
          const Weight *kernel = m_KernelWeights[k];

          applyFunc(xNeuron, yNeuron, k, kernel[kernelIndex]);
        }
      }
    }
  }

  // Convolve a (padded) input image with the convolution kernel
  template<typename A, typename I>
  void ConvolveImage(unsigned int imageWidth, unsigned int imageHeight,
                    unsigned int fixedPoint, A applyFunc, I getPixelFunc)
  {
    // Stride through image pixels
    for(unsigned int imageX = 0; imageX < (imageWidth - KernelSize); imageX += m_Stride)
    {
      for(unsigned int imageY = 0; imageY < (imageHeight - KernelSize); imageY += m_Stride)
      {
        // Loop through kernels
        for(unsigned int k = 0; k < m_NumKernels; k++)
        {
          // Get current kernel
          const Weight *kernel = m_KernelWeights[k];

          // Loop through kernel pixels
          int32_t value = 0;
          for(unsigned int kernelX = 0; kernelX < KernelSize; kernelX++)
          {
            for(unsigned int kernelY = 0; kernelY < KernelSize; kernelY++)
            {
              // Get image pixel
              auto imagePixel = getPixelFunc(imageX + kernelX, imageY + kernelY);

              // Read three colour components from image
              const int32_t kernelR = kernel[kernelX + (KernelSize * (kernelY + (0 * KernelSize)))];
              const int32_t kernelG = kernel[kernelX + (KernelSize * (kernelY + (1 * KernelSize)))];
              const int32_t kernelB = kernel[kernelX + (KernelSize * (kernelY + (2 * KernelSize)))];

              // Convolve kernel with image
              value = __smlabb(std::get<0>(imagePixel), kernelR, value);
              value = __smlabb(std::get<1>(imagePixel), kernelG, value);
              value = __smlabb(std::get<2>(imagePixel), kernelB, value);
            }
          }

          // Shift down to complete fixed point multiply-accumulate
          value >>= fixedPoint;

          // Apply value to pixel at centre of kernel
          applyFunc(imageX + HalfKernelSize, imageY + HalfKernelSize, k, value);
        }
      }
    }
  }

  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  unsigned int m_Stride;

  // Number of kernels
  unsigned int m_NumKernels;

  // Array of 3D kernels
  Weight **m_KernelWeights;
};
} // ConvLayer