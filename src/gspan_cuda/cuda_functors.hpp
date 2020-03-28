#ifndef __CUDA_FUNCTORS_HPP__
#define __CUDA_FUNCTORS_HPP__


namespace gspan_cuda {

struct is_one_array
{
  int *d_array;
  is_one_array(int *arr) {
    d_array = arr;
  }

  __device__ __host__
  int operator()(int idx)
  {
    return (d_array[idx] == 1) ? 1 : 0;
  }
};


} // namespace gspan_cuda

#endif

