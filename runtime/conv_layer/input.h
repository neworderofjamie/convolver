#pragma once

// Standard includes
#include <tuple>

// Rig CPP common includes
#include "rig_cpp_common/arm_intrinsics.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"

// Namespaces
using namespace Common::ARMIntrinsics;

//--------------------------------------------------------------------------
// ConvLayer::Input
//--------------------------------------------------------------------------
namespace ConvLayer
{
template<typename Input>
class InputBase
{
public:
  //--------------------------------------------------------------------------
  // Public methods
  //--------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *region, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "InputBase::ReadSDRAMData");

    // Read input image count
    const unsigned int numInputImages = *region++;
    LOG_PRINT(LOG_LEVEL_INFO, "\t%u input images", numInputImages);

    if(numInputImages > 0)
    {
      // Read fixed point position
      m_FixedPointPosition = *region++;
      LOG_PRINT(LOG_LEVEL_INFO, "\tFixed point position:%u",
                m_FixedPointPosition);

      // Read neuron slice dimensions
      m_Width = *region++;
      m_Height = *region++;
      const uint32_t depth = *region++;
      LOG_PRINT(LOG_LEVEL_INFO, "\tWidth:%u, height:%u, depth:%u",
                m_Width, m_Height, depth);

      // Check depth is compatible
      if(depth != 3)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Only 3 channel input is currently supported");
        return false;
      }

      // Allocate array, large enough to hold a single image
      const unsigned int numBytes = m_Width * m_Height * depth * sizeof(Input);
      m_Input = (Input*)spin1_malloc(numBytes);
      if(m_Input == NULL)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Cannot allocate %u bytes for input images",
                  numBytes);
        return false;
      }

      // Copy first image into DTCM
      spin1_memcpy(m_Input, region, numBytes);

#if LOG_LEVEL <= LOG_LEVEL_TRACE
      for(unsigned int y = 0; y < m_Height; y++)
      {
        for(unsigned int x = 0; x < m_Width; x++)
        {
          auto pixel = GetPixel(x, y);
          io_printf(IO_BUF, "(%d, %d, %d),",
                    std::get<0>(pixel), std::get<1>(pixel), std::get<2>(pixel));
        }
        io_printf(IO_BUF, "\n");
      }

#endif
    }
    return true;
  }

  std::tuple<int32_t, int32_t, int32_t> GetPixel(unsigned int x, unsigned int y) const
  {
    // Get index in x-y
    // index_xy = (y * m_Width) + x
    const unsigned int index_xy = __smlabb((int32_t)y, (int32_t)m_Width,
                                           (int32_t)x);

    // Get pointer to start of the RGB values for this pixel
    const Input *input = &m_Input[3 * index_xy];

    // Read off R, G and B values and return in tuple
    const int32_t r = *input++;
    const int32_t g = *input++;
    const int32_t b = *input;
    return std::make_tuple(r, g, b);
  }

  uint32_t GetWidth() const
  {
    return m_Width;
  }

  uint32_t GetHeight() const
  {
    return m_Height;
  }

  uint32_t GetFixedPointPosition() const
  {
    return m_FixedPointPosition;
  }

  bool HasInput() const
  {
    return (m_Input != NULL);
  }

private:
  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  uint32_t m_FixedPointPosition;

  uint32_t m_Width;
  uint32_t m_Height;

  // 3D image
  Input *m_Input;
};
} // ConvLayer