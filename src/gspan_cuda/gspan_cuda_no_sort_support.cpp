#include <gspan_cuda_no_sort.hpp>

gspan_cuda::gspan_cuda *get_gspan_cuda_instance()
{
  return new gspan_cuda::gspan_cuda_no_sort();
}


