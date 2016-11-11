#pragma once

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

  }

  template<typename A>
  void ConvolveSpike(int xIn, int yIn, int zIn,
    A applyFunc) const
  {
    // Get stack of 2D kernels to apply to inputs
    // from zIn for each output dimension
    const Weight ***zWeights = m_KernelWeights[zIn];
    for(int xKernel = 0; xKernel < KernelSize; xKernel++)
    {
      for(int yKernel = 0; yKernel < KernelSize; yKernel++)
      {
        // Calculate corresponding output pixel
        const int xNeuron = xIn - xKernel + 1;
        const int yNeuron = yIn - yKernel + 1;

        // Loop through kernels and apply
        for(unsigned int k = 0; k < m_NumKernels; k++)
        {
          applyFunc(xNeuron, yNeuron, k, zWeights[yKernel][xKernel][k]]);
        }
      }
    }
  }

  // Convolve a (padded) input image with the convolution kernel
  template<typename A>
  void ConvolveImage(const int8_t ***image, unsigned int imageWidth, unsigned int imageHeight,
                    unsigned int fixedPoint, A applyFunc)
  {
    // Stride through image pixels
    for(unsigned int imageX = 0; imageX < (imageWidth - KernelSize); imageX += m_Stride)
    {
      for(unsigned int imageY = 0; imageY < (imageHeight - KernelSize); imageY += m_Stride)
      {
        // Loop through kernels
        for(unsigned int k = 0; k < m_NumKernels; k++)
        {
          // Loop through kernel pixels
          int32_t value = 0;
          for(unsigned int kernelX = 0; kernelX < KernelSize; kernelX++)
          {
            for(unsigned int kernelY = 0; kernelY < KernelSize; kernelY++)
            {
              // Read three colour components from image
              const int32_t imageR = image[imageX + kernelX][imageY + kernelY][0];
              const int32_t imageG = image[imageX + kernelX][imageY + kernelY][1];
              const int32_t imageB = image[imageX + kernelX][imageY + kernelY][2];

              // Read three colour components from image
              const int32_t kernelR = m_KernelWeights[0][kernelY][kernelX][k];
              const int32_t kernelG = m_KernelWeights[1][kernelY][kernelX][k];
              const int32_t kernelB = m_KernelWeights[2][kernelY][kernelX][k];

              // Convolve kernel with image
              value = __smlabb(imageR, kernelR, value);
              value = __smlabb(imageG, kernelG, value);
              value = __smlabb(imageB, kernelB, value);
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

private:
  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  unsigned int m_Stride;

  // Number of kernels and array of 2D kernel weight
  unsigned int m_NumKernels;

  // 4D array kernel z, kernel y, kernel x, output dimension
  Weight ****m_KernelWeights;
};
} // ConvLayer