#ifndef isi_mtt_two_line_extractor_h
#define isi_mtt_two_line_extractor_h

#include "OpenCVEnv.h"


namespace isi    
{
  namespace mtt  // Markerless Tool Tracking
  {
    // The results are given in the form of: ax + by + c=0
    // for both lines
    bool cvTwoLineExtractor(const cv::Mat& edges,
      float* line1, float* line2);

    bool cvTwoLineExtractorPts(const std::vector<cv::Point>& locs,
      float* line1, float* line2);

    bool cvTwoLineExtractor2(const cv::Mat& mask,
      float* line1, float* line2);
  }
}


#endif
