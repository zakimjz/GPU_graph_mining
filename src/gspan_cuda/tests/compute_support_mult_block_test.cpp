#include <gtest/gtest.h>
#include <logger.hpp>
#include <cuda_datastructures.hpp>
#include <dfs_code.hpp>
#include <cuda_gspan_ops.hpp>


static Logger *value_logger = Logger::get_logger("VAL");
static Logger *logger = Logger::get_logger("MAIN");

using gspan_cuda::extension_element_t;
using types::DFS;


TEST(compute_support_mult_block, basic_test)
{


  cudaDeviceReset();

  gspan_cuda::extension_element_t ext_array[] = {
    //                   DFS part       | graph/embedding part
    extension_element_t( 0, 3, 1, 1, 0,    4,  5, 0),
    extension_element_t( 0, 3, 1, 2, 0,    6,  5, 0),
    extension_element_t( 0, 3, 1, 1, 2,    6,  8, 0),

    extension_element_t( 0, 3, 1, 1, 0,   18, 15, 0),
    extension_element_t( 0, 3, 1, 1, 2,   18, 16, 0),

    extension_element_t( 0, 3, 1, 2, 0,   51, 55, 0),

    ////

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
  }; //20
     /*
        DFS(0, 3, 1, 1, 0); 2
        DFS(0, 3, 1, 2, 0); 2
        DFS(0, 3, 1, 1, 2); 2
        DFS(2, 3, 0, 1, 0): 3
        DFS(2, 3, 0, 2, 2): 2
        DFS(2, 3, 0, 1, 2): 3
        DFS(2, 3, 0, 2, 1): 2
        DFS(2, 0, 0, 1, 0): 4
      */

  int ext_array_length = 20;
  gspan_cuda::extension_element_t *d_ext_array = 0;
  CUDAMALLOC(&d_ext_array, sizeof(gspan_cuda::extension_element_t) * ext_array_length, *logger);
  CUDA_EXEC(cudaMemcpy(d_ext_array, ext_array, sizeof(gspan_cuda::extension_element_t) * ext_array_length, cudaMemcpyHostToDevice), *logger);

  int ext_block_offsets[] = {0, -1, 6};
  int ext_block_lengths[] = {6, -1, 14};

  int max_graph_vertex_count = 10;


  types::DFS dfs_array[] = {
    DFS(0, 3, 1, 1, 0), //support 2
    DFS(0, 3, 1, 2, 0), //support 2
    DFS(0, 3, 1, 1, 2), //support 2
    DFS(2, 3, 0, 1, 0), // support: 3
    DFS(2, 3, 0, 2, 2), // support: 2
    DFS(2, 3, 0, 1, 2), // support: 3
    DFS(2, 3, 0, 2, 1), // support: 2
    DFS(2, 0, 0, 1, 0) // support: 4
  };
  int expected_supports [] = {2, 2, 2, 3, 2, 3, 2, 4};

  types::DFS *d_dfs_array = 0;
  int dfs_array_length = 8;
  CUDAMALLOC(&d_dfs_array, sizeof(types::DFS) * dfs_array_length, *logger);
  CUDA_EXEC(cudaMemcpy(d_dfs_array, dfs_array, sizeof(types::DFS) * dfs_array_length, cudaMemcpyHostToDevice), *logger);

  int *supports = 0;
  int *d_graph_boundaries_scan = 0;
  int d_graph_boundaries_scan_length = 0;
  CUDAMALLOC(&d_graph_boundaries_scan, sizeof(int) * ext_array_length, *logger);

  //int *d_ext_block_offsets;
  //CUDAMALLOC(&d_ext_block_offsets, sizeof(int) * 3, *logger);
  //CUDA_EXEC(cudaMemcpy(d_ext_block_offsets, ext_block_offsets, sizeof(int) * 3, cudaMemcpyHostToDevice), *logger);


  int mapped_db_size = 0;
  int *d_graph_flags = 0;
  int d_graph_flags_length = 0;

  int max_mapped_db_size = 0;

  cuda_segmented_scan *scanner = new cuda_segmented_scan();

  for(int i = 0; i<3; i++) {
    if(ext_block_offsets[i] == -1 ) continue;

    remap_database_graph_ids_mult(d_ext_array + ext_block_offsets[i],
                                  ext_block_lengths[i],
                                  max_graph_vertex_count,
                                  d_graph_boundaries_scan + ext_block_offsets[i],
                                  mapped_db_size,
                                  scanner);
    if(max_mapped_db_size < mapped_db_size)
      max_mapped_db_size = mapped_db_size;
  }



  std::cout << "graph boundaries for all block after the scan\n" << print_d_array(d_graph_boundaries_scan, ext_array_length) << std::endl;

  INFO(*logger, "running compute_support_remapped_db_multiple_dfs_blocks");


  cuda_segmented_reduction *reduction = new cuda_segmented_reduction();

  compute_support_remapped_db_multiple_dfs_blocks(d_ext_array,
                                                  ext_array_length,
                                                  d_dfs_array,
                                                  dfs_array_length,
                                                  supports,
                                                  max_graph_vertex_count,
                                                  d_graph_flags,
                                                  d_graph_flags_length,
                                                  d_graph_boundaries_scan,
                                                  d_graph_boundaries_scan_length,
                                                  max_mapped_db_size,
                                                  reduction);
  for(int i = 0; i < dfs_array_length; i++) {
    INFO(*logger, "dfs: " << dfs_array[i].to_string() << "; support: " << supports[i] << "; expected support: " << expected_supports[i]);
    ASSERT_EQ(supports[i], expected_supports[i]);
  } // for i

  CUDAFREE(d_ext_array, *logger);
  CUDAFREE(d_dfs_array, *logger);
  CUDAFREE(d_graph_boundaries_scan, *logger);
  CUDAFREE(d_graph_flags, *logger);
  //CUDAFREE(d_ext_block_offsets, *logger);
  delete [] supports;
  delete reduction;
  delete scanner;

  memory_checker::detect_memory_leaks();
  Logger::free_loggers();
}

