#pragma once

#include <cmath>


template <typename T>
T* rowptr(T* start, int x, int y, int w)
{
	return start + y*w + x;
}


void compute_11_features(unsigned char* color,
						 unsigned char* gray,
						 const int w,
						 const int h)
						 // float* covar_mat)
{
	// Compute 11 features

	const int N = w*h;

	// Allocate the feature images
	float* red = new float [N]; 
	float* green = new float [N];
	float* blue = new float [N];
	float* xlocs = new float [N];
	float* ylocs = new float [N];
	float* gx = new float [N];
	float* gy = new float [N];
	float* gx2 = new float [N];
	float* gy2 = new float [N];
	float* mag = new float [N];
	float* ori = new float [N];

	float* ftr_ptrs[] = {red, green, blue, xlocs, ylocs, gx, gy, gx2, gy2, mag, ori};

	// Let's start with splitting the color channels into 
	// individual images
	clock_t t1, t2;
	t1 = clock();
	for (int y = 0; y < h; y++)
	{
		// Get row ptr from the color image
        //creates pointer of img = img + (y * w) + constant
		const unsigned char* src = rowptr<unsigned char>(color, 0, y, w);

		// Get row ptrs for the destination channel features
		float* rptr = rowptr<float>(red, 0, y, w);
		float* gptr = rowptr<float>(green, 0, y, w);
		float* bptr = rowptr<float>(blue, 0, y, w);

		for (int x = 0; x < w; x++)
		{
			*rptr++ = (float)*src++;
			*gptr++ = (float)*src++;
			*bptr++ = (float)*src++;
		}
	}
	 t2 = clock();
    //printf("before process time: %d", t1); 
	//printf("Time to calculate color channel on the host is %d", diff); 

	// Next the pixel locations
	for (int y = 0; y < h; y++)
	{
		// Get row ptrs to the x and y feature images
		float* xptr = rowptr<float>(xlocs, 0, y, w);
		float* yptr = rowptr<float>(ylocs, 0, y, w);

		for (int x = 0; x < w; x++)
		{
			*xptr++ = (float)x;
			*yptr++ = (float)y;
		}
	}

	// Let's do first order gradients in x and y
	memset(gx, 0, N*sizeof(float));
	memset(gy, 0, N*sizeof(float));
	memset(gx2, 0, N*sizeof(float));
	memset(gy2, 0, N*sizeof(float));
	// Gx
	for (int y = 0; y < h; y++)
	{
		// ptr to gray image 1 pixel ahead
		const unsigned char* ahead = rowptr<unsigned char>(gray, 2, y, w);
		// ptr to gray image 1 pixel behind
		const unsigned char* behind = rowptr<unsigned char>(gray, 0, y, w);

		// gx ptr
		float* gxptr = rowptr<float>(gx, 1, y, w);

		for (int x = 1; x < w-1; x++, ahead++, behind++, gxptr++)
		{
			*gxptr = (float)*ahead - (float)*behind;
		}
	}
	// Gy
	for (int y = 1; y < h-1; y++)
	{
		// ptr to gray image 1 row ahead
		const unsigned char* ahead = rowptr<unsigned char>(gray, 0, y+1, w);
		// ptr to gray image 1 row behind
		const unsigned char* behind = rowptr<unsigned char>(gray, 0, y-1, w);

		// gy ptr
		float* gyptr = rowptr<float>(gy, 0, y, w);

		for (int x = 0; x < w; x++, ahead++, behind++, gyptr++)
		{
			*gyptr = (float)*ahead - (float)*behind;
		}
	}
	// Gx2
	for (int y = 0; y < h; y++)
	{
		// ptr to gx image 1 pixel ahead
		const float* ahead = rowptr<float>(gx, 2, y, w);
		// ptr to gx image 1 pixel behind
		const float* behind = rowptr<float>(gx, 0, y, w);

		// gx2 ptr
		float* gx2ptr = rowptr<float>(gx2, 1, y, w);

		for (int x = 1; x < w-1; x++, ahead++, behind++, gx2ptr++)
		{
			*gx2ptr = *ahead - *behind;
		}
	}
	// Gy2
	for (int y = 1; y < h-1; y++)
	{
		// ptr to gy image 1 row ahead
		const float* ahead = rowptr<float>(gy, 0, y+1, w);
		// ptr to gy image 1 row behind
		const float* behind = rowptr<float>(gy, 0, y-1, w);

		// gy2 ptr
		float* gy2ptr = rowptr<float>(gy2, 0, y, w);

		for (int x = 0; x < w; x++, ahead++, behind++, gy2ptr++)
		{
			*gy2ptr = *ahead - *behind;
		}
	}

	// Finally the gradient magnitude and orientation
	for (int y = 0; y < h; y++)
	{
		// ptrs to gx and gy
		const float* gxptr = rowptr<float>(gx, 0, y, w);
		const float* gyptr = rowptr<float>(gy, 0, y, w);

		// ptrs to mag and ori
		float* magptr = rowptr<float>(mag, 0, y, w);
		float* oriptr = rowptr<float>(ori, 0, y, w);

		for (int x = 0; x < w; x++, gxptr++, gyptr++)
		{
			*magptr++ = (*gxptr)*(*gxptr) + (*gyptr)*(*gyptr);
			*oriptr++ = atan2(* gyptr, *gxptr);
		}
	}

//////////////////////////////
	 //
      // COMPUTE COVARIANCE MATRIX
      //

      // features is [NumberOfFeatures x N]

      // Compute the mean vector
      float* mean = new float [NumberOfFeatures];
      const float meanNorm = 1.0f / static_cast<float>(N);
      for (int i = 0; i < NumberOfFeatures; i++)
      {
        mean[i] = 0.0f;

        //const float* src = (float*)features.ptr(i);// cv mat is features. 
		const float* src = ftr_ptrs[i];
        for (int j = 0; j < N; j++, src++)
        {
          mean[i] += *src;
        }

        mean[i] *= meanNorm;
      }

	  float* feature_matrix[][];
      // Take the mean out of the data
      for (int i = 0; i < NumberOfFeatures; i++)
      {
        float* src = ftr_ptrs[i];
		feature_matrix[i][] = *src;

        for (int j = 0; j < N; j++, src++)
        {
          *src -= mean[i];
		  feature_matrix[][j] = *src;
        }
      }

	  
    
	
      // Compute the scatter matrix
      cv::Mat scatter = features * features.t();
 
      // Finally the covariance matrix
      const float covarNorm = 1.0f / static_cast<float>(N-1);
      covar_mat = covarNorm * scatter;

	/// delete everything
	delete [] red;
	delete [] green;
	delete [] blue;
	delete [] xlocs;
	delete [] ylocs;
	delete [] gx;
	delete [] gy;
	delete [] gx2;
	delete [] gy2;
	delete [] mag;
	delete [] ori;
	
}