#ifndef __CUDA_REDUCTION_HPP__
#error "do not include reduction.tcc directly, use reduction.h instead"
#endif

#ifndef __REDUCTION_KERNEL_TCC__
#define __REDUCTION_KERNEL_TCC__

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/


#include <stdio.h>

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved 
   inactivity means that no whole warps are active, which is also very 
   inefficient */
template <class _read_kernel_t, class _op_kernel_t, class T>
__global__ void
reduce0(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    //sdata[tid] = (i < n) ? g_idata[i] : 0;
    sdata[tid] = (i < n) ? read_kernel_inst(i) : init_value;
    
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0) {
	  //sdata[tid] += sdata[tid + s];
	  sdata[tid] = op_kernel_inst(sdata[tid], sdata[tid + s]);
	  
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved 
   addressing results in many shared memory bank conflicts. 
*/
template <class _read_kernel_t, class _op_kernel_t, class T>
__global__ void
reduce1(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    // sdata[tid] = (i < n) ? g_idata[i] : 0;    
    sdata[tid] = (i < n) ? read_kernel_inst(i) : init_value;
    
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) 
    {
        int index = 2 * s * tid;

        if (index < blockDim.x) 
        {
	  //sdata[index] += sdata[index + s];
	  sdata[index] = op_kernel_inst(sdata[index], sdata[index + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class _read_kernel_t, class _op_kernel_t, class T>
__global__ void
reduce2(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    //sdata[tid] = (i < n) ? g_idata[i] : 0;
    sdata[tid] = (i < n) ? read_kernel_inst(i) : init_value;
    
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
	  //sdata[tid] += sdata[tid + s];
	  sdata[tid] = op_kernel_inst(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class _read_kernel_t, class _op_kernel_t, class T>
__global__ void
reduce3(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    //T mySum = (i < n) ? g_idata[i] : 0;
    T mySum = (i < n) ? read_kernel_inst(i) : init_value;
    if (i + blockDim.x < n) {
      //mySum += g_idata[i+blockDim.x];
      //mySum += read_kernel_inst(i+blockDim.x);
      mySum = op_kernel_inst(mySum, read_kernel_inst(i+blockDim.x));
    }

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
	  sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid + s]);
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version unrolls the last warp to avoid synchronization where it 
    isn't needed.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class _read_kernel_t, class _op_kernel_t, class T, unsigned int blockSize>
__global__ void
reduce4(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    //T mySum = (i < n) ? g_idata[i] : 0;
    T mySum = (i < n) ? read_kernel_inst(i) : init_value;
    if (i + blockSize < n)  {
      //mySum += g_idata[i+blockSize];
      //mySum += read_kernel_inst(i+blockSize);
      mySum = op_kernel_inst(mySum, read_kernel_inst(i+blockSize));
    }

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s)
        {
	  sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid + 32]); }
        if (blockSize >=  32) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid + 16]); }
        if (blockSize >=  16) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  8]); }
        if (blockSize >=   8) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  4]); }
        if (blockSize >=   4) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  2]); }
        if (blockSize >=   2) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  1]); }
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version is completely unrolled.  It uses a template parameter to achieve 
    optimal code for any (power of 2) number of threads.  This requires a switch 
    statement in the host code to handle all the different thread block sizes at 
    compile time.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class _read_kernel_t, class _op_kernel_t, class T, unsigned int blockSize>
__global__ void
reduce5(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    //T mySum = (i < n) ? g_idata[i] : 0;
    T mySum = (i < n) ? read_kernel_inst(i) : init_value;
    if (i + blockSize < n) {
      //mySum += read_kernel_inst(i+blockSize);//g_idata[i+blockSize];
      mySum = op_kernel_inst(mySum, read_kernel_inst(i+blockSize));
    }

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid +  64]); } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid + 32]); }
        if (blockSize >=  32) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid + 16]); }
        if (blockSize >=  16) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  8]); }
        if (blockSize >=   8) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  4]); }
        if (blockSize >=   4) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  2]); }
        if (blockSize >=   2) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  1]); }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class _read_kernel_t, class _op_kernel_t, class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(_read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst, T init_value, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = init_value;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
      //mySum += read_kernel_inst(i);//g_idata[i];
      mySum = op_kernel_inst(mySum, read_kernel_inst(i));
      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (nIsPow2 || i + blockSize < n) {
	//mySum += read_kernel_inst(i+blockSize);//g_idata[i+blockSize];  
	mySum = op_kernel_inst(mySum, read_kernel_inst(i+blockSize));
      }
      i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = op_kernel_inst(mySum, sdata[tid +  64]); } __syncthreads(); }
    
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid + 32]); }
        if (blockSize >=  32) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid + 16]); }
        if (blockSize >=  16) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  8]); }
        if (blockSize >=   8) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  4]); }
        if (blockSize >=   4) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  2]); }
        if (blockSize >=   2) { smem[tid] = mySum = op_kernel_inst(mySum, smem[tid +  1]); }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}


extern "C"
bool isPow2(unsigned int x);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class _read_kernel_t, class _op_kernel_t, class T>
void 
reduce(int size, int threads, int blocks, 
       int whichKernel,
       _read_kernel_t read_kernel_inst, _op_kernel_t op_kernel_inst,
       T init_value, 
       T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
    case 0:
      reduce0<_read_kernel_t, _op_kernel_t, T><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size);
        break;
    case 1:
      reduce1<_read_kernel_t, _op_kernel_t, T><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size);
        break;
    case 2:
      reduce2<_read_kernel_t, _op_kernel_t, T><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size);
        break;
    case 3:
      reduce3<_read_kernel_t, _op_kernel_t, T><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size);
        break;
    case 4:
        switch (threads)
        {
        case 512:
	  reduce4<_read_kernel_t, _op_kernel_t, T, 512><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 256:
	  reduce4<_read_kernel_t, _op_kernel_t, T, 256><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 128:
	  reduce4<_read_kernel_t, _op_kernel_t, T, 128><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 64:
	  reduce4<_read_kernel_t, _op_kernel_t, T,  64><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 32:
	  reduce4<_read_kernel_t, _op_kernel_t, T,  32><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 16:
	  reduce4<_read_kernel_t, _op_kernel_t, T,  16><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  8:
	  reduce4<_read_kernel_t, _op_kernel_t, T,   8><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  4:
	  reduce4<_read_kernel_t, _op_kernel_t, T,   4><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  2:
	  reduce4<_read_kernel_t, _op_kernel_t, T,   2><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  1:
	  reduce4<_read_kernel_t, _op_kernel_t, T,   1><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        }
        break; 
    case 5:
        switch (threads)
        {
        case 512:
	  reduce5<_read_kernel_t, _op_kernel_t, T, 512><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 256:
	  reduce5<_read_kernel_t, _op_kernel_t, T, 256><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 128:
	  reduce5<_read_kernel_t, _op_kernel_t, T, 128><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 64:
	  reduce5<_read_kernel_t, _op_kernel_t, T,  64><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 32:
	  reduce5<_read_kernel_t, _op_kernel_t, T,  32><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case 16:
	  reduce5<_read_kernel_t, _op_kernel_t, T,  16><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  8:
	  reduce5<_read_kernel_t, _op_kernel_t, T,   8><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  4:
	  reduce5<_read_kernel_t, _op_kernel_t, T,   4><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  2:
	  reduce5<_read_kernel_t, _op_kernel_t, T,   2><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        case  1:
	  reduce5<_read_kernel_t, _op_kernel_t, T,   1><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
        }
        break;       
    case 6:
    default:
        if (isPow2(size))
        {
            switch (threads)
            {
            case 512:
	      reduce6<_read_kernel_t, _op_kernel_t, T, 512, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 256:
	      reduce6<_read_kernel_t, _op_kernel_t, T, 256, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 128:
	      reduce6<_read_kernel_t, _op_kernel_t, T, 128, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 64:
	      reduce6<_read_kernel_t, _op_kernel_t, T,  64, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 32:
	      reduce6<_read_kernel_t, _op_kernel_t, T,  32, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 16:
	      reduce6<_read_kernel_t, _op_kernel_t, T,  16, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  8:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   8, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  4:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   4, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  2:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   2, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  1:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   1, true><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            }
        }
        else
        {
            switch (threads)
            {
            case 512:
	      reduce6<_read_kernel_t, _op_kernel_t, T, 512, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 256:
	      reduce6<_read_kernel_t, _op_kernel_t, T, 256, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 128:
	      reduce6<_read_kernel_t, _op_kernel_t, T, 128, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 64:
	      reduce6<_read_kernel_t, _op_kernel_t, T,  64, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 32:
	      reduce6<_read_kernel_t, _op_kernel_t, T,  32, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case 16:
	      reduce6<_read_kernel_t, _op_kernel_t, T,  16, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  8:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   8, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  4:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   4, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  2:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   2, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            case  1:
	      reduce6<_read_kernel_t, _op_kernel_t, T,   1, false><<< dimGrid, dimBlock, smemSize >>>(read_kernel_inst, op_kernel_inst, init_value, d_odata, size); break;
            }
        }
        break;       
    }
}

#endif // #ifndef __REDUCTION_KERNEL_TCC__


