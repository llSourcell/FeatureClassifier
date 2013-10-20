#include "Epiline.h"


namespace isi
{
  namespace mtt
  {

    //------------------------------------------------------------------------------------
    Epiline::Epiline()
    {
      mA = mB = mC = 0.0;
      mWidth = mHeight = 0;

      mLeftmost.x = mLeftmost.y = 0;
      mRightmost.x = mRightmost.y = 0;

      mDir.x = mDir.y = 0.0f;
      mNormal.x = mNormal.y = 0.0;
    }

    //------------------------------------------------------------------------------------
    Epiline::Epiline(const double& a, const double& b, const double& c, const int width, const int height)
    {
      setLine(a,b,c,width,height);
    }

    //------------------------------------------------------------------------------------
    Epiline::~Epiline()
    {
    }

    //------------------------------------------------------------------------------------
    void Epiline::setLine(const double& a, const double& b, const double& c, const int width, const int height)
    {
      mA = a;
      mB = b;
      mC = c;

      mWidth = width;
      mHeight = height;

      // The 2 endpoints of the epiline, within the image frame
      double x_0, y_0;
      double x_N, y_N;
      float mag;

      // Solve for y when x = 0
      x_0 = 0.0;
      y_0 = (-mA*x_0 - mC) / mB;

      // Do some error-checking
      if (y_0 > (mHeight-1))
      {
        // Solve for x when y = height-1
        y_0 = mHeight - 1;
        x_0 = (mB*y_0 + mC) / -mA;
        assert((x_0 >= 0.0) && (x_0 <= (mWidth-1)));
      }
      else if (y_0 < 0.0)
      {
        // Solve for x when y = 0
        y_0 = 0.0;
        x_0 = (mB*y_0 + mC) / -mA;
        assert((x_0 >= 0.0) && (x_0 <= (mWidth-1)));
      }
      // else { ...keep it... }

      // Solve for y when x = width-1
      x_N = (double)mWidth - 1.0;
      y_N = (-mA*x_N - mC) / mB;

      // Do some error-checking
      if (y_N > (mHeight-1))
      {
        // Solve for x when y = height-1
        y_N = mHeight - 1;
        x_N = (mB*y_N + mC) / -mA;
        assert((x_N >= 0.0) && (x_N <= (mWidth-1)));
      }
      else if (y_N < 0.0)
      {
        // Solve for x when y = 0
        y_N = 0.0;
        x_N = (mB*y_N + mC) / -mA;
        assert((x_N >= 0.0) && (x_N <= (mWidth-1)));
      }
      // else { ...keep it... }

      // Store the endpoints--round to nearest neighbor
      mLeftmost.x  = (int)(x_0 + 0.5);
      mLeftmost.y  = (int)(y_0 + 0.5);
      mRightmost.x = (int)(x_N + 0.5);
      mRightmost.y = (int)(y_N + 0.5);

      // Now compute the unit direction vector from (x_0,y_0) to (x_N,y_N)
      mDir.x = (float)(x_N - x_0);
      mDir.y = (float)(y_N - y_0);
      mag = ::sqrt(mDir.x*mDir.x + mDir.y*mDir.y);
      mDir.x /= mag;
      mDir.y /= mag;

      // And now the unit normal to the direction vector
      mNormal.x = -mDir.y / mDir.x;
      mNormal.y = 1.0f;
      mag = ::sqrt(mNormal.x*mNormal.x + mNormal.y*mNormal.y);
      mNormal.x /= mag;
      mNormal.y /= mag;
    }

    //------------------------------------------------------------------------------------
    void Epiline::setLine2(const double& a, const double& b, const double& c, const int width, const int height)
    {
      mA = a;
      mB = b;
      mC = c;

      mWidth = width;
      mHeight = height;

      // The 4 potential endpoints of the epiline, within the image frame
      double x[4], y[4];
      float mag;

      // Solve for y when x = 0
      x[0] = 0.0;
      y[0] = (-mA*x[0] - mC) / mB;

      // Solve for y when x = mWidth-1
      x[1] = mWidth-1;
      y[1] = (-mA*x[1] - mC) / mB;

      // Solve for x when y = 0;
      y[2] = 0.0;
      x[2] = -(mB*y[2] + mC) / mA;

      // Solve for x when y = mHeight-1
      y[3] = mHeight-1;
      x[3] = -(mB*y[3] + mC) / mA;

      // There are only 2 good points of intersection.  Those will be
      // the points which are BOTH within the image bounds
      double xint[2], yint[2];
      int idx = 0;
      if (x[0]>=0 && x[0]<=mWidth-1 && y[0]>=0 && y[0]<=mHeight-1)
      {
        xint[idx] = x[0];
        yint[idx] = y[0];
        ++idx;
      }
      if (x[1]>=0 && x[1]<=mWidth-1 && y[1]>=0 && y[1]<=mHeight-1)
      {
        xint[idx] = x[1];
        yint[idx] = y[1];
        ++idx;
      }
      if (x[2]>=0 && x[2]<=mWidth-1 && y[2]>=0 && y[2]<=mHeight-1)
      {
        if (idx >= 2)
        {
          // What to do here??  This doesn't make any sense to get here!
          std::cerr << "setLine2() -- WHAT HAPPENED HERE??\n" << std::endl;
          return;
        }
        xint[idx] = x[2];
        yint[idx] = y[2];
        ++idx;
      }
      if (x[3]>=0 && x[3]<=mWidth-1 && y[3]>=0 && y[3]<=mHeight-1)
      {
        if (idx >= 2)
        {
          // What to do here??  This doesn't make any sense to get here!
          std::cerr << "setLine2() -- WHAT HAPPENED HERE??\n" << std::endl;
          return;
        }
        xint[idx] = x[3];
        yint[idx] = y[3];
      }

      // Figure out leftmost from rightmost
      double x_0, y_0, x_N, y_N;
      if (xint[0] < xint[1])
      {
        x_0 = xint[0];
        y_0 = yint[0];
        x_N = xint[1];
        y_N = yint[1];
      }
      else
      {
        x_0 = xint[1];
        y_0 = yint[1];
        x_N = xint[0];
        y_N = yint[0];
      }

      // Store the endpoints--round to nearest neighbor
      mLeftmost.x  = (int)(x_0 + 0.5);
      mLeftmost.y  = (int)(y_0 + 0.5);
      mRightmost.x = (int)(x_N + 0.5);
      mRightmost.y = (int)(y_N + 0.5);

      // Now compute the unit direction vector from (x_0,y_0) to (x_N,y_N)
      mDir.x = (float)(x_N - x_0);
      mDir.y = (float)(y_N - y_0);
      mag = ::sqrt(mDir.x*mDir.x + mDir.y*mDir.y);
      mDir.x /= mag;
      mDir.y /= mag;

      // And now the unit normal to the direction vector
      mNormal.x = -mDir.y / mDir.x;
      mNormal.y = 1.0f;
      mag = ::sqrt(mNormal.x*mNormal.x + mNormal.y*mNormal.y);
      mNormal.x /= mag;
      mNormal.y /= mag;
    }

  }
}



