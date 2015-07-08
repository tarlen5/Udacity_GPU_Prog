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


/////////////////////////////////////////////////////////////////
//------------------ Useful for Debugging ---------------------//
/////////////////////////////////////////////////////////////////

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


void compare_arrays(const unsigned int* h_array, const unsigned int* d_array,
                    const size_t numElems)
{

  unsigned int h_copy[numElems];
  checkCudaErrors(cudaMemcpy(h_copy, d_array, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
  printf("\ncomparing arrays...\n");
  for (int ii=0; ii<numElems; ii++) {
    if(h_copy[ii] != h_array[ii]) {
      printf("DISAGREEMENT AT: %u!  host: %u, device: %u \n",ii,h_array[ii],
             h_copy[ii]);
      exit(1);
    }
  }
  printf("DONE!\n");

}
/*---------------------------------------------------------------------------------*/


///////////////////////////////////////////////////////
//--------------------- KERNELS ---------------------//
///////////////////////////////////////////////////////
__global__ void split_array(unsigned int* d_inputVals, unsigned int* d_splitVals,
                            const size_t numElems, unsigned int mask,
                            unsigned int ibit)
{

  int array_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (array_idx >= numElems) return;

  // Split based on whether inputVals digit is 1 or 0:
  d_splitVals[array_idx] = !(d_inputVals[array_idx] & mask);

}


__global__ void blelloch_scan_single_block(unsigned int* d_in_array,
                                           const size_t numBins,
                                           unsigned normalization=0)
/*
  Computes the blelloch exclusive scan for a cumulative distribution function of a
  histogram, one block at a time.

  \Params:
    * d_in_array - input array of histogram values in each bin. Gets converted
      to cdf by the end of the function.
    * numBins - number of bins in the histogram (Must be < 2*MAX_THREADS_PER_BLOCK)
    * normalization - constant value to add to all bins
      (when doing full exclusive sum scan over multiple blocks).
*/
{

  int thid = threadIdx.x;

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

  if (thid < numBins) d_in_array[thid] = temp_array[thid] + normalization;
  if ((thid + numBins/2) < numBins)
    d_in_array[thid + numBins/2] = temp_array[thid + numBins/2] + normalization;

}


__global__ void compute_outputPos(const unsigned int* d_inputVals,
                       unsigned int* d_outputVals,
                       unsigned int* d_outputPos, unsigned int* d_tVals,
                       const unsigned int* d_splitVals,
                       const unsigned int* d_cdf, const unsigned int totalFalses,
                       const unsigned int numElems)
{

  int thid = threadIdx.x;
  int global_id = blockIdx.x*blockDim.x + thid;
  if (global_id >= numElems) return;

  d_tVals[global_id] = global_id - d_cdf[global_id] + totalFalses;

  unsigned int scatter = (!(d_splitVals[global_id]) ?
                          d_tVals[global_id] : d_cdf[global_id] );
  d_outputPos[global_id] = scatter;

}


__global__ void do_scatter(unsigned int* d_outputVals, const unsigned int* d_inputVals,
                           unsigned int* d_outputPos,
                           unsigned int* d_inputPos,
                           unsigned int* d_scatterAddr,
                           const unsigned int numElems)
{

  int global_id = blockIdx.x*blockDim.x + threadIdx.x;
  if(global_id >= numElems) return;

  d_outputVals[d_outputPos[global_id]]  = d_inputVals[global_id];
  d_scatterAddr[d_outputPos[global_id]] = d_inputPos[global_id];
  
}


///////////////////////////////////////////////////////////
//--------------------- END KERNELS ---------------------//
///////////////////////////////////////////////////////////



void full_blelloch_exclusive_scan(unsigned int* d_binScan, const size_t totalNumElems)
/*
  NOTE: blelloch_scan_single_block() does an exclusive sum scan over
  an array (balanced tree) of size 2*MAX_THREADS_PER_BLOCK, by
  performing the up and down sweep of the scan in shared memory (which
  is limited in size).

  In order to scan over an entire array of size >
  2*MAX_THREADS_PER_BLOCK, we employ the following procedure:

    1) Compute total number of blocks of size 2*MAX_THREADS_PER_BLOCK
    2) Loop over each block and compute a partial array of number
    of bins: 2*MAX_THREADS_PER_BLOCK
    3) Give this partial array to blelloch_scan_single_block() and let
       it return the sum scan.
    4) Now, one has a full array of partial sum scans, and then we take the
       last element of the j-1 block and add it to each element of the jth
       block.

  \Params:
    * d_binScan - starts out as the "histogram" or in this case, the
      split_array that we will perform an exclusive scan over.
    * totalNumElems - total number of elements in the d_binScan array to
      perform an exclusive scan over.
*/
{

  int nthreads = MAX_THREADS_PER_BLOCK;
  int nblocksTotal = (totalNumElems/2 - 1) / nthreads + 1;
  int partialBins = 2*nthreads;
  int smSize = partialBins*sizeof(unsigned);

  // Need a balanced d_binScan array so that on final block, correct
  // values are given to d_partialBinScan.
  // 1. define balanced bin scan
  // 2. set all values to zero
  // 3. copy all of binScan into binScanBalanced.
  unsigned int* d_binScanBalanced;
  unsigned int balanced_size = nblocksTotal*partialBins*sizeof(unsigned);
  checkCudaErrors(cudaMalloc((void**)&d_binScanBalanced, balanced_size));
  checkCudaErrors(cudaMemset(d_binScanBalanced, 0, balanced_size));
  checkCudaErrors(cudaMemcpy(d_binScanBalanced, d_binScan,
                             totalNumElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));

  unsigned int* d_partialBinScan;
  checkCudaErrors(cudaMalloc((void**)&d_partialBinScan, partialBins*sizeof(unsigned)));

  unsigned int* normalization = (unsigned*)malloc(sizeof(unsigned));
  unsigned int* lastVal = (unsigned*)malloc(sizeof(unsigned));
  for (unsigned iblock = 0; iblock < nblocksTotal; iblock++) {
    unsigned offset = iblock*partialBins;

    // Copy binScan Partition into partialBinScan
    checkCudaErrors(cudaMemcpy(d_partialBinScan, (d_binScanBalanced + offset),
                               smSize, cudaMemcpyDeviceToDevice));

    if (iblock > 0) {
      // get normalization - final value in last cdf bin + last value in original
      checkCudaErrors(cudaMemcpy(normalization, (d_binScanBalanced + (offset-1)),
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(lastVal, (d_binScan + (offset-1)),
                                 sizeof(unsigned), cudaMemcpyDeviceToHost));
      *normalization += (*lastVal);
    } else *normalization = 0;

    blelloch_scan_single_block<<<1, nthreads, smSize>>>(d_partialBinScan,
                                                        partialBins,
                                                        *normalization);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Copy partialBinScan back into binScanBalanced:
    checkCudaErrors(cudaMemcpy((d_binScanBalanced+offset), d_partialBinScan, smSize,
                               cudaMemcpyDeviceToDevice));

  }

  // ONE BLOCK WORKING HERE!!!
  // binScanBalanced now needs to be copied into d_binScan!
  checkCudaErrors(cudaMemcpy(d_binScan,d_binScanBalanced,totalNumElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));

  free(normalization);
  free(lastVal);
  checkCudaErrors(cudaFree(d_binScanBalanced));
  checkCudaErrors(cudaFree(d_partialBinScan));

}


void compute_scatter_addresses(const unsigned int* d_inputVals,
                               unsigned int* d_outputVals,
                               unsigned int* d_inputPos,
                               unsigned int* d_outputPos,
                               unsigned int* d_scatterAddr,
                               const unsigned int* const d_splitVals,
                               const unsigned int* const d_cdf,
                               const unsigned totalFalses,
                               const size_t numElems)
/*
  Modifies d_outputVals and d_outputPos
*/
{

  unsigned int* d_tVals;
  checkCudaErrors(cudaMalloc((void**)&d_tVals, numElems*sizeof(unsigned)));

  int nthreads = MAX_THREADS_PER_BLOCK;
  int nblocks  = (numElems - 1) / nthreads + 1;
  compute_outputPos<<<nblocks, nthreads>>>(d_inputVals, d_outputVals, d_outputPos,
                                           d_tVals, d_splitVals, d_cdf, totalFalses,
                                           numElems);
  // Testing purposes - REMOVE in production
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  do_scatter<<<nblocks, nthreads>>>(d_outputVals, d_inputVals, d_outputPos,
                                    d_inputPos, d_scatterAddr, numElems);
  // Testing purposes - REMOVE in production
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(d_tVals));

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElemsReal)
/*
  Beginning (example) values:
    * d_inputVals  - large integers (10 digit, 32 bit unsigned)
    * d_inputPos   - [0 1 2 ... 220477 220478 220479]
    * d_outputVals - [0 0 0 ... 0 0 0 ]
    * d_outputPos  - [0 0 0 ... 0 0 0 ]

*/
{

  //const size_t numElems = 8;
  const size_t numElems = numElemsReal;
  //print_to_file(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);


  //-----Set up-----
  const int numBits = 1;
  unsigned int* d_splitVals;
  checkCudaErrors(cudaMalloc((void**)&d_splitVals, numElems*sizeof(unsigned)));
  unsigned int* d_cdf;
  checkCudaErrors(cudaMalloc((void**)&d_cdf, numElems*sizeof(unsigned)));

  // d_scatterAddr keeps track of the scattered original addresses at every pass
  unsigned int* d_scatterAddr;
  checkCudaErrors(cudaMalloc((void**)&d_scatterAddr, numElems*sizeof(unsigned)));
  checkCudaErrors(cudaMemcpy(d_scatterAddr, d_inputPos, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));

  // Need a global device array for blelloch scan:
  const int nBlellochBins = 1 << unsigned(log(numElems)/log(2) + 0.5);
  unsigned int* d_blelloch;
  checkCudaErrors(cudaMalloc((void**)&d_blelloch, nBlellochBins*sizeof(unsigned)));
  //printf("  numElems: %lu, numBlellochBins: %d \n",numElems, nBlellochBins);

  unsigned int* d_inVals = d_inputVals;
  unsigned int* d_inPos = d_inputPos;
  unsigned int* d_outVals = d_outputVals;
  unsigned int* d_outPos = d_outputPos;

  // Testing purposes - also free'd at end
  unsigned int* h_splitVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_cdf = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_inVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_outVals = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_inPos = (unsigned*)malloc(numElems*sizeof(unsigned));
  unsigned int* h_outPos = (unsigned*)malloc(numElems*sizeof(unsigned));


  // Parallel radix sort - For each pass (each bit):
  //   1) Split values based on current bit
  //   2) Scan values of split array
  //   3) Compute scatter output position
  //   4) Scatter output values using inputVals and outputPos
  for(unsigned ibit = 0; ibit < 8 * sizeof(unsigned); ibit+=numBits) {

    checkCudaErrors(cudaMemset(d_splitVals, 0, numElems*sizeof(unsigned)));
    checkCudaErrors(cudaMemset(d_cdf,0,numElems*sizeof(unsigned)));
    checkCudaErrors(cudaMemset(d_blelloch,0,nBlellochBins*sizeof(unsigned)));


    // Step 1: Split values on True if bit matches 0 in the given bit
    // NOTE: mask = [1 2 4 8 ... 2147483648]
    //              [2^0, 2^1,...2^31]
    unsigned int mask = 1 << ibit;
    int nthreads = MAX_THREADS_PER_BLOCK;
    int nblocks = (numElems - 1)/nthreads + 1;
    split_array<<<nblocks, nthreads>>>(d_inVals, d_splitVals, numElems,
                                       mask, ibit);
    // Testing purposes - REMOVE in production
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(d_cdf, d_splitVals, numElems*sizeof(unsigned),
                               cudaMemcpyDeviceToDevice));

    // Step 2: Scan values of split array:
    // Uses Blelloch exclusive scan
    full_blelloch_exclusive_scan(d_cdf, numElems);
    // STEP 2 --> WORKING!!! VERIFIED FOR ALL STEPS!


    // Step 3: compute scatter addresses
    // Get totalFalses:
    unsigned totalFalses = 0;
    checkCudaErrors(cudaMemcpy(h_splitVals, d_splitVals + (numElems-1), sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_cdf, d_cdf + (numElems -1), sizeof(unsigned),
                               cudaMemcpyDeviceToHost));
    totalFalses = h_splitVals[0] + h_cdf[0];
    compute_scatter_addresses(d_inVals, d_outVals, d_inPos, d_outPos, d_scatterAddr,
                              d_splitVals, d_cdf, totalFalses, numElems);

    // swap pointers:
    std::swap(d_inVals, d_outVals);
    std::swap(d_inPos, d_scatterAddr);

  }

  // Do we need this?
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));

  // Put scatter addresses (->inPos) into d_outputVals;
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inPos, numElems*sizeof(unsigned),
                             cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaFree(d_splitVals));
  checkCudaErrors(cudaFree(d_cdf));
  checkCudaErrors(cudaFree(d_blelloch));

  free(h_splitVals);
  free(h_cdf);
  free(h_inVals);
  free(h_outVals);
  free(h_inPos);
  free(h_outPos);

}

