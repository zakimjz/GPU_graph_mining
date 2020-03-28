#ifndef __KERNEL_EXECUTION_HPP__
#define __KERNEL_EXECUTION_HPP__

#include <iostream>
#include <stdio.h>
#include <cuda_computation_parameters.hpp>
#include <stdexcept>
#include <cuda_tools.hpp>
#include <logger.hpp>

namespace cudapp {

template<class _kernel_op_>
__global__ void execute_kernel_device(int start_idx_global_range, int end_idx_global_range, int values_per_thread, _kernel_op_ kop)
{
  // local variables
  // blockDim.x
  // gridDim.x
  // blockIdx.x
  // threadIdx.x

  // First, compute the index of the thread. The threads are ordered
  // by threadIdx.x inside one block and the blocks are ordered by
  // blockIdx.x

  int thread_count = blockDim.x * gridDim.x;
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  thread_idx += start_idx_global_range; // shift the thread by the start_idx offset.

  for(int round = 0; round < values_per_thread; round++, thread_idx += thread_count) {
    if(thread_idx >= end_idx_global_range) break;
    kop(thread_idx);
  } // thread_idx
} // execute_kernel_device



template<class _kernel_op_>
void for_each(int start_idx, int end_idx, _kernel_op_ kop, int max_grid_dim, int max_block_dim, int values_per_thread)
{
  if(end_idx == start_idx) return;

  if(end_idx < start_idx) {
    CRITICAL_ERROR(*Logger::get_logger("CPP"), "start_idx: " << start_idx << "; end_idx: " << end_idx);
    throw std::runtime_error("Invalid start_idx and end_idx: end_idx <= start_idx.");
  }
  dim3 block_dim(max_block_dim, 1, 1);
  dim3 grid_dim(max_grid_dim, 1, 1);

  int thread_num = end_idx - start_idx;
  if(max_block_dim * max_grid_dim * values_per_thread < thread_num) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "max_block_dim: " << max_block_dim << "; max_grid_dim: " << max_grid_dim << "; thread_num: " << thread_num);
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "start_idx: " << start_idx << "; end_idx: " << end_idx);
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "max_block_dim * max_grid_dim: " << max_block_dim * max_grid_dim);
    throw std::runtime_error("invalid dimensions of the thread block/grid");
  } // if

  //DEBUG(*Logger::get_logger("CUDA_KE"), "executing kernel with grid dim: " << max_grid_dim << "; block dim: " << max_block_dim);
  TRACE4(*Logger::get_logger("CUDA_PP"), "start_idx: " << start_idx << "; end_idx: " << end_idx);

  execute_kernel_device<<< grid_dim, block_dim >>>(start_idx, end_idx, values_per_thread, kop);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "error while executing kernel ; error: " << err << "; string: " << cudaGetErrorString(err));
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "max_grid_dim: " << max_grid_dim << "; max_block_dim: " << max_block_dim << "; thread_num: " << thread_num);
    abort();
  }


  CUDA_EXEC(cudaDeviceSynchronize(), *Logger::get_logger("CUDA_PP"));

} // execute_kernel


template<class _kernel_op_>
void for_each(int start_idx, int end_idx, _kernel_op_ kop, cuda_computation_parameters params)
{
  if(params.check_for_linear_arrays() == false) {
    throw std::runtime_error("computation params should specify a 1D array of blocks/threads");
  } // if
  for_each(start_idx, end_idx, kop, params.grid_dim.x, params.block_dim.x, params.values_per_thread);
  CUDA_EXEC(cudaDeviceSynchronize(), *Logger::get_logger("CUDA_PP"));
} // for_each


template<class _kernel_op_, class T>
struct transform_and_store_kernel {
  _kernel_op_ op;
  T *array;
  transform_and_store_kernel(_kernel_op_ op, T *array) : op(op), array(array) {
  }
  __device__
  void operator()(int idx) {
    array[idx] = op(idx);
  }
};

template<class _kernel_op_, class T>
void transform(int start_idx, int end_idx, _kernel_op_ kop, T *array, cuda_computation_parameters params)
{
  for_each(start_idx, end_idx, transform_and_store_kernel<_kernel_op_, T>(kop, array), params);
  CUDA_EXEC(cudaDeviceSynchronize(), *Logger::get_logger("CUDA_PP"));
} // transform

} // namespace cuda

#endif

