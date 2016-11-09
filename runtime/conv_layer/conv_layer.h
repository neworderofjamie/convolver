#pragma once

//--------------------------------------------------------------------------
// Kernel
//--------------------------------------------------------------------------
template<typename Weight>
class Kernel
{
public:
  //--------------------------------------------------------------------------
  // Public methods
  //--------------------------------------------------------------------------
  template<typename ApplyFunction>
  void ConvolveInput(int x_in, int y_in, int d_in,
    ApplyFunction apply) const
  {
    // Loop through pixels
    const Weight *dimensionWeights = m_KernelWeights[d_in];
    for(int x_ker = 0; x_ker < m_KernelSize; x_ker++)
    {
      for(int y_ker = 0; y_ker < m_KernelSize; y_ker++)
      {
        // Calculate corresponding output pixel
        const int x_out = x_in - x_ker + 1;
        const int y_out = y_in - y_ker + 1;

        // Loop through kernels
        for(unsigned int k = 0; k < m_NumKernels; k++)
        {
          apply(x_out, y_out, k, *dimensionWeights++);
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

//--------------------------------------------------------------------------
// Neurons
//--------------------------------------------------------------------------
template<typename State, typename Weight>
class Neurons
{
public:
  void ApplyInput(unsigned int x_neuron, unsigned int y_neuron, Weight weight)
  {

  }

private:
  State *m_State;
};

//-----------------------------------------------------------------------------
// Event handler functions
//-----------------------------------------------------------------------------
void MCPacketReceived(uint key, uint)
{
  // 1) Extract global x and y from key
}