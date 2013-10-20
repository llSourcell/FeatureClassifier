//test.cu


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>
#include <iostream>
#include <limits>
#include <direct.h>

#define N      256
#define NUMROW   N
#define NUMCOL   N
#define PIXSIZE  3
#define REDOFF   0
#define GREENOFF 1
#define BLUEOFF  2
#define nTPB    16
#define GRNVAL   5
#define REDVAL   7
#define BLUVAL   9

template <typename T> __device__
T* rowptr(T* start, int x, int y, int w) 
{ 
	return start + y*w + x; 
}

__global__ void color_channels_calc(const unsigned w, const unsigned h, unsigned char* color, float* d_red, float* d_green, float* d_blue)
{

  //Calculates rgb pointers using CUDA loop 
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w) && (idy < h))
  {
    unsigned char *src = rowptr<unsigned char>(color, 0, idy, w);
    float *rptr        = rowptr<float>(d_red, 0, idy, w);
    float *gptr        = rowptr<float>(d_red, 0, idy, w);
    float *bptr        = rowptr<float>(d_red, 0, idy, w);


    *rptr++ = (float)*src++;
    *gptr++ = (float)*src++;
    *bptr++ = (float)*src++;
  }
}

__global__ void pixel_locations_calc(const unsigned w, const unsigned h, float* xlocs, float* ylocs)
{

  //Calculates XY locations using CUDA loop 
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w) && (idy < h))
  {
    float *xptr = rowptr<float>(xlocs, 0, idy, w);
    float *yptr = rowptr<float>(ylocs, 0, idy, w);
    
    *xptr++ = (float)idx;
	*yptr++ = (float)idy;
  }
}

__global__ void gx_gradient_calc(const unsigned w, const unsigned h, unsigned char* gray, float* d_gx)
{

  //Calculates gx gradients
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w-1) && (idy < h))
  {
    // ptr to gray image 1 pixel ahead
	unsigned char* ahead = rowptr<unsigned char>(gray, 2, idy, w);
	
	// ptr to gray image 1 pixel behind
	unsigned char* behind = rowptr<unsigned char>(gray, 0, idy, w);
	
	// gx ptr
	float* gxptr = rowptr<float>(d_gx, 1, idy, w);
    
    *gxptr++ = (float)*ahead++ - (float)*behind++;
  }
}

__global__ void gy_gradient_calc(const unsigned w, const unsigned h, unsigned char* gray, float* d_gy)
{

  //Calculates gy gradients
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w) && (idy < h-1))
  {
    // ptr to gray image 1 row ahead
	unsigned char* ahead = rowptr<unsigned char>(gray, 0, idy+1, w);
	
	// ptr to gray image 1 row behind
	unsigned char* behind = rowptr<unsigned char>(gray, 0, idy-1, w);
	
	// gy ptr
	float* gyptr = rowptr<float>(d_gy, 1, idy, w);
    
    *gyptr++ = (float)*ahead++ - (float)*behind++;
  }
}

__global__ void gx2_gradient_calc(const unsigned w, const unsigned h, float* d_gx, float* d_gx2)
{

  //Calculates gx2 gradients
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w-1) && (idy < h))
  {
    // ptr to gx image 1 pixel ahead
	float* ahead = rowptr<float>(d_gx, 2, idy, w);
	
	// ptr to gx image 1 pixel behind
	float* behind = rowptr<float>(d_gx, 0, idy, w);
	
	// gx2 ptr
	float* gx2ptr = rowptr<float>(d_gx2, 1, idy, w);
    
    *gx2ptr++ = (float)*ahead++ - (float)*behind++;
  }
}

__global__ void gy2_gradient_calc(const unsigned w, const unsigned h, float* d_gy, float* d_gy2)
{

  //Calculates gy2 gradients
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w) && (idy < h-1))
  {
    // ptr to gy image 1 row ahead
	float* ahead = rowptr<float>(d_gy, 0, idy+1, w);
	
	// ptr to gy image 1 row behind
	float* behind = rowptr<float>(d_gy, 0, idy-1, w);
	
	// gy2 ptr
	float* gy2ptr = rowptr<float>(d_gy2, 0, idy, w);
    
    *gy2ptr++ = (float)*ahead++ - (float)*behind++;
  }
}

__global__ void mag_ori_calc(const unsigned w, const unsigned h, float* d_gx, float* d_gy, float* d_mag, float* d_ori)
{

  //Calculates gy2 gradients
  unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
  unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);
  
  if ((idx < w) && (idy < h))
  {
    // ptrs to gx and gy
	float* gxptr = rowptr<float>(d_gx, 0, idy, w);
	float* gyptr = rowptr<float>(d_gy, 0, idy, w);
	
	// ptrs to mag and ori
	float* magptr = rowptr<float>(d_mag, 0, idy, w);
	float* oriptr = rowptr<float>(d_ori, 0, idy, w);
    
    *magptr++ = (*gxptr)*(*gxptr) + (*gyptr)*(*gyptr);
	*oriptr++ = atan2(* gyptr, *gxptr);
	gxptr++;
	gyptr++;
  }
}



__global__ void covariance_calc(int n, int NumberOfFeatures, float* d_red, float* d_green, float* d_blue, float* d_xlocs, float* d_ylocs, float* d_gx, float* d_gy, float* d_gx2, float* d_gy2, 
float* d_mag, float* d_ori, float* d_mean, const float d_meanNorm, float *d_src, float* d_FeatureMatrix, int pitch
, float* d_MatrixTranspose, int pitch2)
{
 
//Collect the pointers
float* ftr_ptrs[] = {d_red, d_green, d_blue, d_xlocs, d_ylocs, d_gx, d_gy, d_gx2, d_gy2, d_mag, d_ori};



unsigned idx = threadIdx.x + (blockDim.x*blockIdx.x);
unsigned idy = threadIdx.y + (blockDim.y*blockIdx.y);

//Step 1 mean vector calc
   if ((idx < NumberOfFeatures) && (idy < n))
   {
       d_mean[idx] = 0.0f;
       d_src = ftr_ptrs[idx];
	   d_mean[idx] += *d_src;
	   d_src++;
	   d_mean[idx] = (d_mean[idx]) * (d_meanNorm);    //ask austin wtf 
   }


//Step 2 Take the mean out of the data

for (int idx = 0; idx < NumberOfFeatures; ++idx) 
     {
        float* row = (float*)((char*)d_FeatureMatrix + idx * pitch);
			   row = ftr_ptrs[idx];
        for (int idy = 0; idy < n; ++idy) 
        {
             *row -= d_mean[idy];
             //float element = row[idy];
        }
     }
     
     
     
//Step 3 Compute the Transpose
for (int idx = 0; idx < NumberOfFeatures; ++idx) 
     {
        float* row = (float*)((char*)d_MatrixTranspose + idx * pitch2);
			   row = ftr_ptrs[idx];
        for (int idy = 0; idy < n; ++idy) 
        {
             *row -= d_mean[idy];
             //float element = row[idy];
        }
     }
     
//50 ms computational time so far
     
  /*   
//Step 4 Compute Scatter Matrix 

float *d_ScatterMatrix;
int row = threadIdx.y;
int col = threadIdx.x;
float P_val = 0;
for (int k = 0; k < n; ++k) 
	{
		float M_elem = d_FeatureMatrix[row * n + k];
		float N_elem = d_MatrixTranspose[k * n + col];
		P_val += M_elem * N_elem;
	}
		d_ScatterMatrix[row*n+col] = P_val;


//Step 5 Compute Covariance Matrix 

const float covarNorm = 1.0f / static_cast<float>(n-1);
float *d_CovarianceMatrix;
for (int idx = 0; idx < NumberOfFeatures; ++idx) //Q6 Austin length and width of Scatter/Cov Matrix?
     {
        float* row = (float*)((char*)d_ScatterMatrix + idx);
			   row = ftr_ptrs[idx];
        for (int idy = 0; idy < n; ++idy) 
        {
             *row -= d_mean[idy];
             //float element = row[idy];
        }
     }
*/
 
}




void compute_11_features_CUDA(unsigned char* color, unsigned char* gray, int w, int h)
{

  //Constants
  const int n =w*h;                          // covariance and scatter is 11x11
  const int NumberOfFeatures = 11;               
  
  //host vars declared
  float *h_red =(float*)malloc(n*sizeof(float));
  float *h_green = (float*)malloc(n*sizeof(float));
  float *h_blue = (float*)malloc(n*sizeof(float));
  float *h_xlocs = (float*)malloc(n*sizeof(float));
  float *h_ylocs = (float*)malloc(n*sizeof(float));
  float *h_gx = (float*)malloc(n*sizeof(float));
  float *h_gy = (float*)malloc(n*sizeof(float));
  float *h_gx2 = (float*)malloc(n*sizeof(float));
  float *h_gy2 = (float*)malloc(n*sizeof(float));
  float *h_mag = (float*)malloc(n*sizeof(float));
  float *h_ori = (float*)malloc(n*sizeof(float));
  
  //device vars declared 
  float *d_red;
  float *d_green;
  float *d_blue;
  float *d_xlocs;
  float *d_ylocs;
  float *d_gx;
  float *d_gy;
  float *d_gx2;
  float *d_gy2;
  float *d_mag;
  float *d_ori; 
  unsigned char *d_color;
  unsigned char *d_gray;
  

  //device vars alloc'd
  cudaMalloc((void **)&d_color, (n*PIXSIZE)*sizeof(unsigned char));        
  cudaMalloc((void **)&d_gray, (n)*sizeof(unsigned char));
  cudaMalloc((void **)&d_red, (n)*sizeof(float));
  cudaMalloc((void **)&d_green, (n)*sizeof(float));
  cudaMalloc((void **)&d_blue, (n)*sizeof(float));
  cudaMalloc((void **)&d_xlocs, (n)*sizeof(float));
  cudaMalloc((void **)&d_ylocs, (n)*sizeof(float));
  cudaMalloc((void **)&d_gx, (n)*sizeof(float));
  cudaMalloc((void **)&d_gy, (n)*sizeof(float));
  cudaMalloc((void **)&d_gx2, (n)*sizeof(float));
  cudaMalloc((void **)&d_gy2, (n)*sizeof(float));
  cudaMalloc((void **)&d_mag, (n)*sizeof(float));
  cudaMalloc((void **)&d_ori, (n)*sizeof(float));
  
  //Device Memsets the first order gradients 
  cudaMemset(d_gx, 0, n*sizeof(float));
  cudaMemset(d_gy, 0, n*sizeof(float));
  cudaMemset(d_gx2, 0, n*sizeof(float));
  cudaMemset(d_gy2, 0, n*sizeof(float));
  
  //Copies host image to device image.
  cudaMemcpy(d_color, color, (n*PIXSIZE)*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gray,  gray,  (n*PIXSIZE)*sizeof(unsigned char),  cudaMemcpyHostToDevice);
  

  //covariance variables
  float *d_mean; 
  const float d_meanNorm = 1.0f / static_cast<float>(n);
  float* d_src;
  
  //covariance allocs 
  cudaMalloc((void **)&d_mean, (NumberOfFeatures)*sizeof(float));
  cudaMalloc((void **)&d_meanNorm, sizeof(const float));           
  cudaMalloc((void **)&d_src, sizeof(const float)); 

  //2D Feature Matrix 		
  float* d_FeatureMatrix;
  float* h_FeatureMatrix[NumberOfFeatures][11];
  size_t pitch;
  cudaMallocPitch((void**)&d_FeatureMatrix, &pitch, n * sizeof(float), NumberOfFeatures); //width, then height. switch?
  cudaMemcpy2D(d_FeatureMatrix, pitch, h_FeatureMatrix, n*sizeof(int), n*sizeof(int), NumberOfFeatures, cudaMemcpyHostToDevice);
 
  //2D Matrix Transpose
  float* d_MatrixTranspose;
  float* h_MatrixTranspose[11][NumberOfFeatures];
  size_t pitch2;
  cudaMallocPitch((void**)&d_MatrixTranspose, &pitch2, NumberOfFeatures, n * sizeof(float)); //width, then height. switch?
  cudaMemcpy2D(d_MatrixTranspose, pitch2, h_MatrixTranspose, NumberOfFeatures, NumberOfFeatures, n*sizeof(int), cudaMemcpyHostToDevice);


  //Calculates GPU Usage 
  dim3 block(nTPB, nTPB);
  dim3 grid(((w+nTPB-1)/nTPB),((h+nTPB-1)/nTPB));
  
  //Starts Timer 1 
  cudaEvent_t start, stop;
  cudaEventCreate(&start);     	
  cudaEventCreate(&stop);
  float elapsed_time_ms; 
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);  		
////////////////////////////////////////////////////////////////////////////////////
  //Calls CUDA rgb pointer calc kernel 
  color_channels_calc<<<grid,block>>>(w, h, d_color, d_red, d_green, d_blue);
  
  //Calls CUDA XY pointer calc kernel
  pixel_locations_calc<<<grid,block>>>(w, h, d_xlocs, d_ylocs);
 
  //Calls CUDA gx gradient calc kernel
  gx_gradient_calc<<<grid,block>>>(w, h, d_gray, d_gx);
  
  //Calls CUDA gy gradient calc kernel
  gx_gradient_calc<<<grid,block>>>(w, h, d_gray, d_gy);
  
  //Calls CUDA gx2 gradient calc kernel
  gx2_gradient_calc<<<grid,block>>>(w, h, d_gx, d_gx2);
  
  //Calls CUDA gy2 gradient calc kernel
  gy2_gradient_calc<<<grid,block>>>(w, h, d_gy, d_gy2);
  
  //Calls CUDA gradient magnitude and orientation calc kernel
  mag_ori_calc<<<grid,block>>>(w, h, d_gx, d_gy, d_mag, d_ori);
  
//Covariance kernel
   
covariance_calc<<<grid,block>>>(n, NumberOfFeatures, d_red, d_green, d_blue, d_xlocs, d_ylocs, 
d_gx, d_gy, d_gx2, d_gy2, d_mag, d_ori, d_mean, d_meanNorm, d_src, d_FeatureMatrix, pitch
,d_MatrixTranspose, pitch2 );

////////////////////////////////////////////////////////////////////////////////////  
  //Ends Timer 1
  cudaEventRecord(stop, 0);     		
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop );
  printf("GPU Time: %f ms.\n", elapsed_time_ms);  // print out execution time


 //cudaMemcpy2D( h_FeatureMatrix, pitch, d_FeatureMatrix,  n*sizeof(int), n*sizeof(int), NumberOfFeatures, cudaMemcpyDeviceToHost );
 //again for transpose
 
 //Download features images to host 

  cudaMemcpy(color, d_color, (n*PIXSIZE)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaMemcpy(gray,  d_gray,  (n*PIXSIZE)*sizeof(unsigned char),  cudaMemcpyDeviceToHost); 
  
  cudaMemcpy(h_red, d_red, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_green,  d_green,  (n)*sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(h_blue, d_blue, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_xlocs,  d_xlocs,  (n)*sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ylocs, d_ylocs, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gx,  d_gx,  (n)*sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gy, d_gy, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gx2,  d_gx2,  (n)*sizeof(float),  cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gy2, d_gy2, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mag, d_mag, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ori, d_ori, (n)*sizeof(float), cudaMemcpyDeviceToHost);
  
float* ftr_ptrs[] = {h_red, h_green, h_blue, h_xlocs, h_ylocs, h_gx, h_gy, h_gx2, h_gy2, h_mag, h_ori};
 
  //deallocs everything
  cudaFree(d_red); 
  cudaFree(d_green); 
  cudaFree(d_blue);
  cudaFree(d_xlocs);
  cudaFree(d_ylocs);
  cudaFree(d_gx);
  cudaFree(d_gy);
  cudaFree(d_gx2);
  cudaFree(d_gy2);
  cudaFree(d_mag);
  cudaFree(d_ori); 
  cudaFree(d_mean); 
  cudaFree(d_src); 
  cudaFree(d_FeatureMatrix); 

  
  
  


  printf("Success!\n");
 // return 0;
}