#ifndef isi_mtt_epiline_h
#define isi_mtt_epiline_h

#include "OpenCVEnv.h"


namespace isi    
{
  namespace mtt  
  {
    class Epiline
    {
    public:
      Epiline();
      Epiline(const double& a, const double& b, const double& c, const int width, const int height);
      ~Epiline();

      void setLine(const double& a, const double& b, const double& c, const int width, const int height);
      void setLine2(const double& a, const double& b, const double& c, const int width, const int height);

      // Get the left-most endpoint
      inline const CvPoint& getLeftmost() const { return mLeftmost; }

      // Get the right-most endpoint
      inline const CvPoint& getRightmost() const { return mRightmost; }

      // Get the unit direction vector for the epiline, from mLeftmost to mRightmost
      inline const CvPoint2D32f& getDirection() const { return mDir; }

      // Get the normal to the unit direction vector for the epiline
      inline const CvPoint2D32f& getNormal() const { return mNormal; }

    private:
      // The parameters for the epipolar line: ax + by + c = 0
      double mA, mB, mC;

      // The dimensions of the image
      int mWidth, mHeight;

      // The left-most and right-most endpoints used for the epiline
      CvPoint mLeftmost, mRightmost;

      // The unit direction vector for the epiline
      CvPoint2D32f mDir;

      // The normal to the unit direction vector for the epiline
      CvPoint2D32f mNormal;
    };
  }
}

#endif
