//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <algorithm>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

*/


const int MAX_THREADS_PER_BLOCK = 512;


void print_to_file(unsigned int* const d_inputVals,
                   unsigned int* const d_inputPos,
                   unsigned int* const d_outputVals,
                   unsigned int* const d_outputPos,
                   const size_t numElems)
/*
  Easiest to print out to file quickly for inspection
*/
{

  unsigned* h_inputVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
  unsigned* h_inputPos = (unsigned*)malloc(numElems*sizeof(unsigned));
  checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
  unsigned* h_outputVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  checkCudaErrors(cudaMemcpy(h_outputVals, d_outputVals, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
  unsigned* h_outputPos = (unsigned*)malloc(numElems*sizeof(unsigned));
  checkCudaErrors(cudaMemcpy(h_outputPos, d_outputPos, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));

  // TESTING: Dump the values to a file:
  FILE *f = fopen("inVals.out","w");
  for (unsigned ii = 0; ii<numElems; ii++) {
    fprintf(f,"%u \n",h_inputVals[ii]);
  }
  fclose(f);

  f = fopen("inPos.out", "w");
  for(unsigned ii=0; ii<numElems; ii++) {
    fprintf(f,"%u \n",h_inputPos[ii]);
  }
  fclose(f);

  f = fopen("outPos.out", "w");
  for(unsigned ii=0; ii<numElems; ii++) {
    fprintf(f,"%u \n",h_outputPos[ii]);
  }
  fclose(f);

  f = fopen("outVals.out", "w");
  for(unsigned ii=0; ii<numElems; ii++) {
    fprintf(f,"%u \n",h_outputVals[ii]);
  }
  fclose(f);

  delete h_inputVals;
  delete h_inputPos;
  delete h_outputVals;
  delete h_outputPos;

}


__global__ void atomic_histo(unsigned int* d_histo, const unsigned int* const d_inArray,
                             const unsigned mask, const unsigned bit,
                             const size_t numElems)
{
  int array_idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (array_idx >= numElems) return;
  
  unsigned bin_idx = (d_inArray[array_idx] & mask) >> bit;
  atomicAdd(&(d_histo[bin_idx]), 1);
}


__global__ void determine_offset(const unsigned int* const d_inVals,
                                 const unsigned int* const d_inPos,
                                 unsigned int* const d_outVals,
                                 unsigned int* const d_outPos,
                                 unsigned int* const d_cdf,
                                 const unsigned mask, const unsigned bit,
                                 const size_t numElems)
{

  int array_idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (array_idx >= numElems) return;

  /* This is wrong!
  unsigned bin_idx = (d_inVals[array_idx] & mask) >> bit;
  d_outVals[d_cdf[bin_idx]] = d_inVals[array_idx];
  d_outPos[d_cdf[bin_idx]] = d_inPos[array_idx];

  atomicAdd(&(d_cdf[bin_idx]), 1);
  */

  // This should actually be correct...AND IT IS!!! Now how do I
  // parallelize this stuff.
  if (array_idx != 0) return;
  for (unsigned int j = 0; j<numElems; ++j) {
    unsigned int bin = (d_inVals[j] & mask) >> bit;
    d_outVals[d_cdf[bin]] = d_inVals[j];
    d_outPos[d_cdf[bin]] = d_inPos[j];
    d_cdf[bin]++;
  }

}



void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
/*
  Beginning (example) values:
    * d_inputVals  - large integers (10 digit, 32 bit unsigned)
    * d_inputPos   - [0 1 2 ... 220477 220478 220479]
    * d_outputVals - [0 0 0 ... 0 0 0 ]
    * d_outputPos  - [0 0 0 ... 0 0 0 ]


 */
{

  //print_to_file(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

  // For checking on host:
  unsigned* h_inputVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));

  //-----Set up-----
  const int numBits = 1;
  const int numBins = 1 << numBits;
  printf("numBits: %u, numBins: %u, numElems: %lu \n",numBits, numBins, numElems);


  // Since these values will be changing at each bit shift, define
  // here then copy back to their final values in function
  // Here we set up a (device) pointer to the (device) pointer.
  unsigned int* d_inVals  = d_inputVals;
  unsigned int* d_inPos   = d_inputPos;
  unsigned int* d_outVals = d_outputVals;
  unsigned int* d_outPos  = d_outputPos;

  unsigned int* h_inVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_outVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_inPos = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_outPos = (unsigned*)malloc(numElems*sizeof(unsigned));


  unsigned int* d_histo;  // for each bit - 0 or 1
  checkCudaErrors(cudaMalloc((void**)&d_histo, numBins*sizeof(unsigned)));
  unsigned int* d_cdf;    // for exclusive prefix scan
  checkCudaErrors(cudaMalloc((void**)&d_cdf, numBins*sizeof(unsigned)));
  unsigned int* h_cdf = new unsigned int[numBins];


  // Parallel radix sort - Do a pass of: 1) Histogram 2) Exclusive Scan 3) Reorder
  // for each bit.
  for(unsigned ibit = 0; ibit < 8 * sizeof(unsigned); ibit+=numBits) {
    //printf("Calculating for bit index: %u\n",ibit);

    checkCudaErrors(cudaMemset(d_histo,0,numBins*sizeof(unsigned)));
    checkCudaErrors(cudaMemset(d_cdf,0,numBins*sizeof(unsigned)));
    memset(h_cdf, 0, numBins*sizeof(unsigned));

    // Step 1: Generate histogram of values that match mask
    unsigned int mask = (numBins - 1) << ibit;
    // NOTE: mask = [1 2 4 8 ... 2147483648]
    //              [2^0, 2^1,...2^31]

    int nthreads = MAX_THREADS_PER_BLOCK;
    int nblocks = (numElems - 1)/nthreads + 1;
    //printf("Launching kernel with %i total threads.\n",nthreads*nblocks);
    atomic_histo<<<nblocks, nthreads>>>(d_histo, d_inVals, mask, ibit, numElems);
    // Testing purposes:
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int h_histo[numBins];
    checkCudaErrors(cudaMemcpy(h_histo, d_histo, numBins*sizeof(unsigned),
                               cudaMemcpyDeviceToHost));


    // Step 2: Exclusive Prefix Sum of histogram:
    // Since we are writing this for only 1 bit in each bucket, we
    // know that this serial version will be (much) faster than
    // launching a kernel to do it.
    for (unsigned int j = 1; j < numBins; ++j)
      h_cdf[j] = h_cdf[j - 1] + h_histo[j - 1];
    //checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, numBins*sizeof(unsigned),
    //                cudaMemcpyHostToDevice));


    // EVERYTHING CORRECT TO THIS POINT...
    // The problem is that I don't know what to parallelize below!

    // Step 3: Determine relative offset of each digit.
    //determine_offset<<<nblocks, nthreads>>>(d_inVals, d_inPos, d_outVals,
    //                                        d_outPos, d_cdf, mask, ibit, numElems);

    checkCudaErrors(cudaMemcpy(h_inVals, d_inVals, numElems*sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_outVals, d_outVals, numElems*sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_inPos, d_inPos, numElems*sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_outPos, d_outPos, numElems*sizeof(unsigned),
                               cudaMemcpyDeviceToHost));

    // Compute the locations serially, but do the scattering (inVals
    // -> outVals) in kernel/parallel??
    for (unsigned int j = 0; j<numElems; ++j) {
      unsigned int bin = (h_inVals[j] & mask) >> ibit;
      h_outVals[h_cdf[bin]] = h_inVals[j];
      h_outPos[h_cdf[bin]] = h_inPos[j];
      h_cdf[bin]++;
    }

    checkCudaErrors(cudaMemcpy(d_inVals, h_inVals, numElems*sizeof(unsigned),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_outVals, h_outVals, numElems*sizeof(unsigned),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_inPos, h_inPos, numElems*sizeof(unsigned),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_outPos, h_outPos, numElems*sizeof(unsigned),
                               cudaMemcpyHostToDevice));
    
    // Need to parallelize this??
    // Have to understand exactly why this is?
    std::swap(d_outVals, d_inVals);
    std::swap(d_outPos,  d_inPos);

  }

  // NOT SURE WHY WE NEED THIS, IF AT ALL...
  // need to copy from input buffer into output
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));

  // You have others to free...right?...
  checkCudaErrors(cudaFree(d_histo));

  free(h_cdf);
  free(h_inVals);
  free(h_outVals);
  free(h_inPos);
  free(h_outPos);
  
}
