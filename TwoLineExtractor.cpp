#include "TwoLineExtractor.h"

#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>

#include "OpenCVEnv.h"


namespace isi
{
  namespace mtt
  {

    //------------------------------------------------------------------
    bool cvTwoLineExtractorPts(const std::vector<cv::Point>& locs,
      float* line1, float* line2)
    {
      // How many total points are there?
      const int N = static_cast<int>(locs.size());

      // If there's less than 4, return false here (can't do it)
      if (N < 4)
      {
        std::cerr << "cvTwoLineExtractorPts() -- ERROR!  Need at least 4 points!\n\n";
        return false;
      }

      // Run RANSAC to get the 2 best lines
      float bestScore = std::numeric_limits<float>::max();
      const int max_iters = 500;
      const float maxDist = 10.0f;
      for (int iter = 0; iter < max_iters; iter++)
      {
        // Randomly select 4 (unique) points to make 2 lines
        int i1 = rand() % N;
        int i2 = rand() % N;
        while (i2 == i1)
        {
          i2 = rand() % N;
        }
        int i3 = rand() % N;
        while ( (i3==i2) || (i3==i1) )
        {
          i3 = rand() % N;
        }
        int i4 = rand() % N;
        while( (i4==i3) || (i4==i2) || (i4==i1) )
        {
          i4 = rand() % N;
        }

        // Get the corresponding points
        const cv::Point2f& p1 = locs[i1];
        const cv::Point2f& p2 = locs[i2];
        const cv::Point2f& p3 = locs[i3];
        const cv::Point2f& p4 = locs[i4];

        // Construct the two lines in: y=mx+b form:
        const float slope1 = (p2.y-p1.y) / (p2.x-p1.x);  // line 1 slope
        const float yint1  = p1.y - slope1*p1.x;         // line 1 y-intercept 
        const float slope2 = (p4.y-p3.y) / (p4.x-p3.x);  // line 2 slope 
        const float yint2  = p3.y - slope2*p3.x;         // line 2 y-intercept 

        // Convert to the form: ax+by+c=0
        const float a1 = -slope1;
        const float b1 = 1.0f;
        const float c1 = -yint1;
        const float a2 = -slope2;
        const float b2 = 1.0f;
        const float c2 = -yint2;

        // For each point in the list, compute the perpendicular point-to-line
        // distance.  Take the smaller of the 2, so that when we have the true
        // boundary lines we correspond the points to the closest line.
        float score = 0.0f;
        for (int i = 0; i < N; i++)
        {
          const cv::Point2f p = locs[i];

          // Distance to line 1
          const float dto1 = abs(a1*p.x+b1*p.y+c1) / sqrt(a1*a1+b1*b1);
          // Distance to line 2
          const float dto2 = abs(a2*p.x+b2*p.y+c2) / sqrt(a2*a2+b2*b2);

          // Take the closer one
          if (dto1 < dto2)
          {
            score += std::min<float>(maxDist, dto1);  // <--- The M-SAC step
          }
          else
          {
            score += std::min<float>(maxDist, dto2);  // <--- The M-SAC step
          }
        }

        // If this is the best score so far, save it
        if (score < bestScore)
        {
          bestScore = score;

          line1[0] = a1;
          line1[1] = b1;
          line1[2] = c1;
          line2[0] = a2;
          line2[1] = b2;
          line2[2] = c2;
        }
      }

      // Normalize the points so that sqrt(a^2+b^2) = 1
      float norm = 1.0f/sqrt(line1[0]*line1[0] + line1[1]*line1[1]);
      line1[0] *= norm;  line1[1] *= norm;  line1[2] *= norm;
      norm = 1.0f/sqrt(line2[0]*line2[0] + line2[1]*line2[1]);
      line2[0] *= norm;  line2[1] *= norm;  line2[2] *= norm;

      // Return success
      return true;
    }

    //------------------------------------------------------------------
    bool cvTwoLineExtractor(const cv::Mat& edges,
      float* line1, float* line2)
    {
      // It's assumed that the input edges represents one *blob* in the
      // image and we will extract the two lines representing the boundaries
      // (i.e., of the cylindrical shaft of a tool)

      static bool doOnce = false;
      if (!doOnce)
      {
        doOnce = true;
        srand( (unsigned int)time(0) );
      }

      // Collect all pixel locations of the edge points
      std::vector<cv::Point> locs;
      for (int y = 0; y < edges.rows; y++)
      {
        const unsigned char* src = (unsigned char*)edges.ptr(y);
        for (int x = 0; x < edges.cols; x++, src++)
        {
          if (*src > 0)
          {
            locs.push_back(cv::Point(x,y));
          }
        }
      }

      return cvTwoLineExtractorPts(locs, line1, line2);
    }


    //------------------------------------------------------------------
    struct SegmentSortStruct
    {
      float length;
      int index;
    };
    static bool SegmentComparatorAscending(SegmentSortStruct& i, SegmentSortStruct& j) 
    { 
      return (i.length < j.length); 
    }


    //------------------------------------------------------------------
    bool cvTwoLineExtractor2(const cv::Mat& mask,
      float* line1, float* line2)
    {
      // Collect all of the marked pixels from the mask into a vector
      std::vector<cv::Point2f> points_vec;
      for (int y = 0; y < mask.rows; y++)
      {
        const unsigned char* pMask = (unsigned char*)mask.ptr(y);
        for (int x = 0; x < mask.cols; x++, pMask++)
        {
          if (*pMask > 0)
          {
            points_vec.push_back(cv::Point2f((float)x,(float)y));
          }
        }
      }

      // Now convert to a cv::Mat for the ellipse fitting
      cv::Mat points_mat(points_vec, false);

      // Fit an ellipse to the points
      cv::RotatedRect ellipse_fit = cv::fitEllipse(points_mat);

      // Get the 4 vertices of the rectangle
      cv::Point2f verts[4];
      ellipse_fit.points(verts);

      // These 4 vertices give 4 line segments--let's store them here.
      // (This assumes a clockwise or counter-clockwise ordering of the
      //  vertices.  Otherwise we need to do that here!)
      cv::Point2f segments[4][2];
      segments[0][0] = verts[0];  segments[0][1] = verts[1];
      segments[1][0] = verts[1];  segments[1][1] = verts[2];
      segments[2][0] = verts[2];  segments[2][1] = verts[3];
      segments[3][0] = verts[3];  segments[3][1] = verts[0];

      // Store lengths of each segment
      SegmentSortStruct segment_lengths[4];
      for (int i = 0; i < 4; i++)
      {
        float dx = segments[i][0].x-segments[i][1].x;
        float dy = segments[i][0].y-segments[i][1].y;
        segment_lengths[i].length = sqrt(dx*dx+dy*dy);
        segment_lengths[i].index = i;
      }

      // There should be only 2 different lengths amongst the 4 segments
      // (b/c it's a rectangle), equal to ellipse.size.width and ellipse.size.height.
      // Let's use the longer of these 2 as the line segments output here.
      std::sort(segment_lengths, segment_lengths+4, SegmentComparatorAscending);
      float slope1 = (segments[segment_lengths[2].index][0].y - segments[segment_lengths[2].index][1].y) / 
        (segments[segment_lengths[2].index][0].x - segments[segment_lengths[2].index][1].x);
      float yint1  = segments[segment_lengths[2].index][0].y - slope1*segments[segment_lengths[2].index][0].x;
      float slope2 = (segments[segment_lengths[3].index][0].y - segments[segment_lengths[3].index][1].y) / 
        (segments[segment_lengths[3].index][0].x - segments[segment_lengths[3].index][1].x);
      float yint2  = segments[segment_lengths[3].index][0].y - slope2*segments[segment_lengths[3].index][0].x;

      // And convert to ax+by+c=0 form
      line1[0] = -slope1;
      line1[1] = 1.0f;
      line1[2] = -yint1;
      line2[0] = -slope2;
      line2[1] = 1.0f;
      line2[2] = -yint2;

      // Normalize the points so that sqrt(a^2+b^2) = 1
      float norm = 1.0f/sqrt(line1[0]*line1[0] + line1[1]*line1[1]);
      line1[0] *= norm;  line1[1] *= norm;  line1[2] *= norm;
      norm = 1.0f/sqrt(line2[0]*line2[0] + line2[1]*line2[1]);
      line2[0] *= norm;  line2[1] *= norm;  line2[2] *= norm;

      // Return success
      return true;
    }

  }
}