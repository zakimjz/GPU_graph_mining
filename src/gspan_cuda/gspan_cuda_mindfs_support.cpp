#include <gspan_cuda_mindfs.hpp>

gspan_cuda::gspan_cuda *get_gspan_cuda_instance()
{
  INFO(*Logger::get_logger("MAIN"), "creating gspan_cuda::gspan_cuda_mindfs instance");
  return new gspan_cuda::gspan_cuda_mindfs();
}


