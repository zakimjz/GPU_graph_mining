#ifndef __CUDA_COMPUTATION_PARAMETERS_HPP__
#define __CUDA_COMPUTATION_PARAMETERS_HPP__

#include <cuda.h>
#include <string>
#include <sstream>

namespace cudapp {

struct cuda_computation_parameters {
  dim3 grid_dim;
  dim3 block_dim;
  int values_per_thread;

  cuda_computation_parameters() {
    grid_dim = dim3(0,0,0);
    block_dim = dim3(0,0,0);
    values_per_thread = 0;
  }

  cuda_computation_parameters(dim3 grid_d, dim3 block_d, int vpt) : grid_dim(grid_d), block_dim(block_d) {
    values_per_thread = vpt;
  }

  cuda_computation_parameters(int grid_d, int block_d, int vpt) : grid_dim(grid_d, 1, 1), block_dim(block_d, 1, 1) {
    values_per_thread = vpt;
  }


  bool check_for_linear_arrays() {
    if(grid_dim.y != 1 || grid_dim.z != 1 || block_dim.y != 1 || block_dim.z != 1) {
      return false;
    }   // if
    return true;
  }   // check_for_linear_arrays


  std::string to_string() const {
    std::stringstream ss;
    ss << "grid_dim: (" << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z << "); block_dim: (" << block_dim.x << "," << block_dim.y << "," << block_dim.z << ")";
    return ss.str();
  }
};

struct cuda_gspan_configuration {
  cuda_computation_parameters first_embeddings;
};

extern cuda_gspan_configuration global_cuda_execution_configuration;

} // namespace cuda

#endif

