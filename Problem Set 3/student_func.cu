/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]

  Your task is to calculate this cumulative distribution.


  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  author: Timothy C. Arlen
          tca3@psu.edu

  date: 21 June 2015

  This function was completed as a part of project 3: Tone mapping, and employs
  three fundamental GPU algorithms: reduce, histogram, and (exclusive) scan.
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

*/


#include <cmath>
#include <cstdlib>
//#include <math_functions.h>
#include "utils.h"
#include <cuda_runtime.h>


const int MAX_THREADS_PER_BLOCK = 512;


//----------------------- START KERNELS ------------------------
__global__ void reduce_min_max_kernel(float* d_out, float* d_in, int array_len,
                                      bool use_min)
/*
  Does a reduction of d_in into one final d_out array, where d_out is
  an array with length of the number of blocks in the kernel.

  Then only expectation is that blocks and threads are one dimensional.

  \Params:
    * array_len - length of d_in array
    * use_min - boolean to use minimum reduction operator if true, else use maximium.
*/
{

  // Set up shared memory
  extern __shared__ float input_array[];

  int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int th_idx = threadIdx.x;

  // If this thread index takes us outside of the array, fill it with
  // the first value of the actual global array.
  if (global_idx >= array_len) input_array[th_idx] = d_in[0];
  else input_array[th_idx] = d_in[global_idx];
  __syncthreads(); // syncs up all threads within the block.

  // Do reduction in shared memory. All elements in input_array are
  // filled (some with duplicate values from first element of global
  // input array which wont effect final result).
  for (int power = 0; power < int(logf(array_len)/logf(2.0)); power++) {
    int neighbor = int(pow(2.0,power));
    int skip = 2*neighbor;
    if ((th_idx % skip) == 0) {
      if ((th_idx + neighbor) < blockDim.x) {
        float extrema = 0.0;
        if (use_min) extrema = min(input_array[th_idx],input_array[th_idx + neighbor]);
        else extrema = max(input_array[th_idx],input_array[th_idx + neighbor]);
        input_array[th_idx] = extrema;
      }
    }
    __syncthreads();
  }

  // only thread 0 writes result for this block to d_out:
  if(th_idx == 0) {
    d_out[blockIdx.x] = input_array[0];
  }

}

__global__ void atomic_histo(unsigned int* d_histo, const float* const d_inputArray,
                             const float minimum, const float range, const int numBins)
{
  int array_idx = threadIdx.x + blockDim.x*blockIdx.x;
  int bin_idx = int(numBins*(d_inputArray[array_idx] - minimum) / range);

  bin_idx = min(numBins - 1, bin_idx);
  atomicAdd(&(d_histo[bin_idx]), 1);
}

__global__ void blelloch_scan_single_block(unsigned int* d_in_array, const size_t numBins)
/*
  Computes the blelloch exclusive scan for a cumulative distribution function of a
  histogram, as long as the number of Bins of the histogram is not more than twice
  the number of threads per block.

  Also, if numBins < 2*num_threads, then it will full the end of the
  input array with zeros.

  \Params:
    * d_in_array - input array of histogram values in each bin. Gets converted
      to cdf by the end of the function.
    * numBins - number of bins in the histogram (Must be < 2*MAX_THREADS_PER_BLOCK)
*/
{

  int thid = threadIdx.x;
  int global_id = blockIdx.x*blockDim.x + thid;
  // In this function, above two expressions should be equal, so let's
  // not use global_id!!

  extern __shared__ float temp_array[];

  // Make sure that we do not read from undefined part of array if it
  // is smaller than the number of threads that we gave defined. If
  // that is the case, the final values of the input array are
  // extended to zero.
  if (thid < numBins) temp_array[thid] = d_in_array[thid];
  else temp_array[thid] = 0;
  if( (thid + numBins/2) < numBins)
    temp_array[thid + numBins/2] = d_in_array[thid + numBins/2];
  else temp_array[thid + numBins/2] = 0;

  __syncthreads();

  // Part 1: Up Sweep, reduction
  // Iterate log_2(numBins) times, and each element adds value 'stride'
  // elements away to its own value.
  int stride = 1;
  for (int d = numBins>>1; d > 0; d>>=1) {

    if (thid < d) {
      int neighbor = stride*(2*thid+1) - 1;
      int index = stride*(2*thid+2) - 1;

      temp_array[index] += temp_array[neighbor];
    }
    stride *=2;
    __syncthreads();
  }
  // Now set last element to identity:
  if (thid == 0)  temp_array[numBins-1] = 0;

  // Part 2: Down sweep
  // Iterate log(n) times. Each thread adds value stride elements away to
  // its own value, and sets the value stride elements away to its own
  // previous value.
  for (int d=1; d<numBins; d *= 2) {
    stride >>= 1;
    __syncthreads();

    if(thid < d) {
      int neighbor = stride*(2*thid+1) - 1;
      int index = stride*(2*thid+2) - 1;

      float t = temp_array[neighbor];
      temp_array[neighbor] = temp_array[index];
      temp_array[index] += t;
    }
  }

  __syncthreads();

  if (thid < numBins) d_in_array[thid] = temp_array[thid];
  if ((thid + numBins/2) < numBins)
    d_in_array[thid + numBins/2] = temp_array[thid + numBins/2];

}
//----------------------- END KERNELS ------------------------

void reduce_min_max(const float* const d_input_array, unsigned num_elem_in, float& optimum,
                    bool if_min)
/*
  Split up array into blocks of MAX_THREADS_PER_BLOCK length each, and
  reduce (find extrema) of each block, writing the output of the block
  to a new d_out array. Then the new d_out array becomes the input array
  to perform reduction on, until the length of the d_out array is 1 element
  and extremum is found.
 */
{

  // We can't change original array, so copy it here on device so that
  // we can modify it:
  float* d_in;
  checkCudaErrors(cudaMalloc((void**)&d_in, num_elem_in*sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_in, d_input_array, num_elem_in*sizeof(float),
                             cudaMemcpyDeviceToDevice));

  int nthreads = MAX_THREADS_PER_BLOCK;
  int num_elem_out = (num_elem_in - 1) / MAX_THREADS_PER_BLOCK + 1;
  int nblocks = num_elem_out;

  //unsigned iloop = 0;
  while(true) {

    float* d_out;
    checkCudaErrors(cudaMalloc((void**)&d_out, num_elem_out*sizeof(float)));

    reduce_min_max_kernel<<<nblocks, nthreads, nthreads*sizeof(float)>>>
      (d_out, d_in, num_elem_in, if_min);

    checkCudaErrors(cudaFree(d_in));

    if (num_elem_out <= 1) {
      // Copy output to h_out
      float* h_out = (float*)malloc(sizeof(float));
      checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
      optimum = h_out[0];

      free(h_out);
      checkCudaErrors(cudaFree(d_out));
      break;
    }

    // Now output array becomes new input array:
    num_elem_in = num_elem_out;
    num_elem_out = (num_elem_in - 1) / MAX_THREADS_PER_BLOCK + 1;
    nblocks = num_elem_out;
    d_in = d_out;

  }

  return;

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
/*
  d_logLuminance - array/ptr of log(Luminance) values from image.
  d_cdf - cumulative distribution function of histogram array
  min_logLum - pointer to hold minimum logLuminance value
  max_logLum - pointer to hold max logLuminance value
  numRows - number of rows in image of d_logLuminance
  numCols - number of cols in image of d_logLuminance
  numBins - number of histogram bins
 */
{

  /*CHECKING */
  float* h_logLuminance = (float*)malloc(numRows*numCols*sizeof(float));
  checkCudaErrors(cudaMemcpy(h_logLuminance,d_logLuminance,numRows*numCols*sizeof(float),
                             cudaMemcpyDeviceToHost));

  /*
  // This particular image had 240 cols and 294 rows
  for(int c = 0; c<numCols; c++) {
    for(int r = 0; r<numRows; r++) {
      if (*(h_logLuminance + r*numCols + c) + 4.0f > 0.1f) {
        //printf("col: %d, row: %d, logLum: %f\n",c,r,*(h_logLuminance +r*numCols + c));
        printf("col: %d, row: %d, logLum: %f\n",c,r,h_logLuminance[r*numCols + c]);
      }
    }
  }
  printf("\n\nREFERENCE CALCULATION (Min/Max): \n");
  float logLumMin = h_logLuminance[0];
  float logLumMax = h_logLuminance[0];
  for (size_t i = 1; i < numCols * numRows; ++i) {
    logLumMin = min(h_logLuminance[i], logLumMin);
    logLumMax = max(h_logLuminance[i], logLumMax);
    //logLumMin = (h_logLuminance[i] < logLumMin) ? (h_logLuminance[i]) : (logLumMin);
    //logLumMax = (h_logLuminance[i] > logLumMax) ? (h_logLuminance[i]) : (logLumMax);
  }
  printf("  Min logLum: %f\n  Max logLum: %f\n",logLumMin,logLumMax);
  free(h_logLuminance);
  */

  // Step 1: Parallel implementation of reduce (minimum/maximum)
  // Reduce the array, finding the optimium within each block and
  // writing to the array element corresponding to the block number,
  // until there is only one block left.

  // 1a) Find mimimum:
  int arrayLen = numRows*numCols;
  bool ifMin = true;
  reduce_min_max(d_logLuminance, arrayLen, min_logLum, ifMin);

  // 1b) Find maximum:
  ifMin = false;
  reduce_min_max(d_logLuminance, arrayLen, max_logLum, ifMin);
  //printf("\nPARALLEL CALCULATION: \n");
  //printf("  -->Min of logLum: %f\n  -->Max of logLum: %f\n", min_logLum, max_logLum);


  // Step 2: Find range:
  float logLumRange = max_logLum - min_logLum;


  // Step 3: Generate a histogram of values in logLuminance channel
  // using formula: bin = (lum[i] - lumMin)/lumRange * numBins
  // Use atomicAdd for now

  unsigned int h_histo[numBins];
  // Set to zero everywhere
  for(int i=0; i< numBins; i++) h_histo[i] = 0;

  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc((void**)&d_histo, numBins*sizeof(unsigned)));
  checkCudaErrors(cudaMemcpy(d_histo, h_histo, numBins*sizeof(unsigned),
                             cudaMemcpyHostToDevice));

  int nthreads = MAX_THREADS_PER_BLOCK;
  int nblocks   = (arrayLen-1) / MAX_THREADS_PER_BLOCK + 1;
  atomic_histo<<<nblocks, nthreads>>>(d_histo, d_logLuminance, min_logLum, logLumRange,
                                      numBins);
  checkCudaErrors(cudaMemcpy(h_histo, d_histo, numBins*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));

  // Parallel calc histo checking:
  //printf("\nPARALLEL - REFERENCE Calc Histogram:\n");
  //printf("  Differences: \n");
  //for (int i=0; i<numBins; i++) {
  //  int diff = h_histo[i] - h_histo_ref[i];
  //  if (diff != 0) printf("  %d) diff: %u\n",i,diff);
  //}
  //printf("\n");


  // Step 4: Perform exclusive scan (prefix sum) on the histogram to
  // get the cumulative distribution of luminance values (assign to
  // the incoming d_cdf pointer which already has been pre-allocated)

  // NOTE: This method only works if numBins is even, and if the histogram can
  // be scanned in one block. It is possible to modify this later, but for now,
  // we make this restriction.
  checkCudaErrors(cudaMemcpy(d_cdf, d_histo, numBins*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));
  nthreads = MAX_THREADS_PER_BLOCK;
  nblocks = (numBins/2 - 1) / nthreads + 1;
  if (nblocks > 1) {
    printf("\n  number of blocks: %d\n",nblocks);
    printf("ERROR: current version of cdf can only be implmented in 1 block!");
    exit(0);
  }
  blelloch_scan_single_block<<<nblocks,nthreads,numBins*sizeof(int)>>>(d_cdf, numBins);

  checkCudaErrors(cudaFree(d_histo));

}
