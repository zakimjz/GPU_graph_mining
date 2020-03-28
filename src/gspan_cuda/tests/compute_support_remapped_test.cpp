#include <gtest/gtest.h>
#include <logger.hpp>
#include <cuda_datastructures.hpp>
#include <dfs_code.hpp>
#include <cuda_gspan_ops.hpp>


static Logger *value_logger = Logger::get_logger("VAL");
static Logger *logger = Logger::get_logger("MAIN");

using gspan_cuda::extension_element_t;
using types::DFS;
using gspan_cuda::compute_support_remapped_db;


TEST(compute_support_remapped_db, basic_test)
{

  //throw std::runtime_error("compute_support_remapped_db test produces segmentation fault, needs to be fixed");

  cudaDeviceReset();

  gspan_cuda::extension_element_t ext_array[] = {
    //                   DFS part       | graph/embedding part
    extension_element_t( 2, 3, 0, 1, 0,    1,  2, 0),
    extension_element_t( 2, 3, 0, 2, 2,    2,  3, 0),
    extension_element_t( 2, 3, 0, 1, 2,    1,  3, 0),
    extension_element_t( 2, 0, 0, 1, 0,    0,  3, 0),

    extension_element_t( 2, 3, 0, 1, 0,   10, 11, 0),
    extension_element_t( 2, 3, 0, 1, 2,   10, 12, 0),
    extension_element_t( 2, 0, 0, 1, 0,   10, 13, 0),

    extension_element_t( 2, 3, 0, 1, 2,   20, 21, 0),

    extension_element_t( 2, 3, 0, 2, 2,   50, 51, 0),
    extension_element_t( 2, 3, 0, 1, 0,   50, 52, 0),
    extension_element_t( 2, 3, 0, 2, 1,   52, 51, 0),
    extension_element_t( 2, 0, 0, 1, 0,   52, 53, 0),

    extension_element_t( 2, 3, 0, 2, 1,   61, 63, 0),
    extension_element_t( 2, 0, 0, 1, 0,   60, 62, 0)
  }; // 14
     /*
        DFS(2, 3, 0, 1, 0): 3
        DFS(2, 3, 0, 2, 2): 2
        DFS(2, 3, 0, 1, 2): 3
        DFS(2, 3, 0, 2, 1): 2
        DFS(2, 0, 0, 1, 0): 4
      */

  int ext_array_length = 14;
  gspan_cuda::extension_element_t *d_ext_array = 0;
  CUDAMALLOC(&d_ext_array, sizeof(gspan_cuda::extension_element_t) * ext_array_length, *logger);
  CUDA_EXEC(cudaMemcpy(d_ext_array, ext_array, sizeof(gspan_cuda::extension_element_t) * ext_array_length, cudaMemcpyHostToDevice), *logger);


  int max_graph_vertex_count = 10;





  types::DFS dfs_array[] = {
    DFS(2, 3, 0, 1, 0), // support: 3
    DFS(2, 3, 0, 2, 2), // support: 2
    DFS(2, 3, 0, 1, 2), // support: 3
    DFS(2, 3, 0, 2, 1), // support: 2
    DFS(2, 0, 0, 1, 0) // support: 4
  };
  int expected_supports [] = {3, 2, 3, 2, 4};

  types::DFS *d_dfs_array = 0;
  int dfs_array_length = 5;
  CUDAMALLOC(&d_dfs_array, sizeof(types::DFS) * dfs_array_length, *logger);
  CUDA_EXEC(cudaMemcpy(d_dfs_array, dfs_array, sizeof(types::DFS) * dfs_array_length, cudaMemcpyHostToDevice), *logger);

  int *supports = 0;
  int *d_graph_boundaries_scan = 0;
  int d_graph_boundaries_scan_length = 0;
  int mapped_db_size = 0;
  int *d_graph_flags = 0;
  int d_graph_flags_length = 0;

  INFO(*logger, "running compute_support_remapped_db");

  cuda_segmented_reduction *reduction = new cuda_segmented_reduction();
  cuda_segmented_scan *scanner = new cuda_segmented_scan();

  compute_support_remapped_db_multiple_dfs(d_ext_array,
                                           ext_array_length,
                                           d_dfs_array,
                                           dfs_array_length,
                                           supports,
                                           max_graph_vertex_count,
                                           d_graph_flags,
                                           d_graph_flags_length,
                                           d_graph_boundaries_scan,
                                           d_graph_boundaries_scan_length,
                                           mapped_db_size,
                                           reduction,
                                           scanner);
  for(int i = 0; i < dfs_array_length; i++) {
    DEBUG(*logger, "dfs: " << dfs_array[i].to_string() << "; support: " << supports[i] << "; expected support: " << expected_supports[i]);
    ASSERT_EQ(supports[i], expected_supports[i]);
  } // for i

  CUDAFREE(d_ext_array, *logger);
  CUDAFREE(d_dfs_array, *logger);
  CUDAFREE(d_graph_boundaries_scan, *logger);
  CUDAFREE(d_graph_flags, *logger);
  delete [] supports;
  delete reduction;
  delete scanner;
  //memory_checker::detect_memory_leaks();
  //Logger::free_loggers();
}

