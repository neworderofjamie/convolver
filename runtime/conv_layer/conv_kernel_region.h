#pragma once

//--------------------------------------------------------------------------
// Kernel
//--------------------------------------------------------------------------
namespace ConvLayer
{
template<typename Weight, unsigned int KernelSize>
class ConvKernelBase
{
public:
  //--------------------------------------------------------------------------
  // Public methods
  //--------------------------------------------------------------------------
  template<typename A>
  void ConvolveSpike(int xIn, int yIn, int zIn,
    A applyFunc) const
  {
    // Loop through pixels of kernel
    const Weight *zWeights = m_KernelWeights[zIn];
    for(int xKernel = 0; xKernel < KernelSize; xKernel++)
    {
      for(int yKernel = 0; yKernel < KernelSize; yKernel++)
      {
        // Calculate corresponding output pixel
        const int xNeuron = xIn - xKernel + 1;
        const int yNeuron = yIn - yKernel + 1;

        // Loop through kernels
        for(unsigned int k = 0; k < m_NumKernels; k++)
        {
          applyFunc(xNeuron, yNeuron, k, *zWeights++);
        }
      }
    }
  }

private:
  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  // Size of kernel
  unsigned int m_KernelSize;

  // Number of kernels and array of 2D kernel weight
  unsigned int m_NumKernels;
  Weight *m_KernelWeights;
};
} // ConvLayer