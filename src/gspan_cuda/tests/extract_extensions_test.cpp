#include <string>
#include <iomanip>
#include <iostream>

#include <logger.hpp>
#include <utils.hpp>

#include <test_support.hpp>
#include <sys/time.h>

#include <cuda_utils.hpp>
#include <gtest/gtest.h>

#include <cuda_datastructures.hpp>
#include <cuda_gspan_ops.hpp>

using std::string;
using namespace types;
using namespace gspan_cuda;
using std::cerr;

using std::fixed;

static Logger *value_logger = Logger::get_logger("VAL");
static Logger *logger = Logger::get_logger("MAIN");


TEST(extract_extensions, basic_test)
{
  cudaDeviceReset();

  gspan_cuda::extension_element_t ext_array[] = {
    //                   DFS part       | graph/embedding part
    extension_element_t( 0, 3, 0, 1, 0,    1,  2, 0),
    extension_element_t( 0, 3, 0, 2, 2,    2,  3, 0),
    extension_element_t( 0, 3, 0, 3, 1,   10, 11, 0),
    extension_element_t( 1, 3, 0, 1, 2,    1,  3, 0),
    extension_element_t( 1, 3, 0, 0, 2,   10, 12, 0),
    extension_element_t( 1, 3, 0, 1, 2,   20, 21, 0),
    extension_element_t( 1, 3, 0, 2, 1,   30, 31, 0),
    extension_element_t( 2, 3, 0, 1, 1,    1,  2, 0),
    extension_element_t( 2, 3, 0, 1, 1,    0,  2, 0),
    extension_element_t( 2, 3, 0, 1, 1,    1,  3, 0),
    extension_element_t( 2, 0, 0, 1, 0,    0,  3, 0)
  };
  int max_graph_vertex_count = 10;
  int num_edge_labels = 4;
  int num_vertex_labels = 3;
  int num_columns = 3;
  int ext_array_length = 11;
  int db_size = 4;

  int *block_offsets = 0;
  int num_blocks = -1;
  types::DFS *dfs_array = 0;
  int dfs_array_length = -1;
  int *d_fwd_flags = 0, *d_bwd_flags = 0;
  types::DFS *d_fwd_block_dfs_array = 0, *d_bwd_block_dfs_array = 0;
  types::DFS *d_ext_dfs_array = 0;
  int *d_ext_block_offsets = 0, d_ext_block_offsets_length = 0, d_ext_dfs_array_length = 0;

  cuda_segmented_scan *scanner = new cuda_segmented_scan();
  cuda_copy<types::DFS, is_one_array> *copier = new cuda_copy<types::DFS, is_one_array>();


  gspan_cuda::extension_element_t *d_ext_array = 0;
  CUDAMALLOC(&d_ext_array, sizeof(gspan_cuda::extension_element_t) * ext_array_length, *logger);
  CUDA_EXEC(cudaMemcpy(d_ext_array, ext_array, sizeof(gspan_cuda::extension_element_t) * ext_array_length, cudaMemcpyHostToDevice), *logger);

  CUDAMALLOC(&d_fwd_flags, sizeof(int) * num_edge_labels * num_vertex_labels, *logger);
  CUDA_EXEC( cudaMemset(d_fwd_flags,0, sizeof(int) * num_edge_labels * num_vertex_labels), *logger );
  CUDAMALLOC(&d_bwd_flags, sizeof(int) * num_edge_labels * num_columns, *logger);
  CUDA_EXEC( cudaMemset(d_bwd_flags,0, sizeof(int) * num_edge_labels * num_columns), *logger );
  CUDAMALLOC(&d_fwd_block_dfs_array, sizeof(types::DFS) * num_edge_labels * num_vertex_labels, *logger);
  CUDAMALLOC(&d_bwd_block_dfs_array, sizeof(types::DFS) * num_edge_labels * num_columns, *logger);

  gspan_cuda::extract_extensions(num_edge_labels,
                                 num_vertex_labels,
                                 num_columns,
                                 d_ext_array,
                                 ext_array_length,
                                 d_ext_block_offsets,
                                 d_ext_block_offsets_length,
                                 d_ext_dfs_array,
                                 d_ext_dfs_array_length,
                                 d_fwd_flags,
                                 d_bwd_flags,
                                 d_fwd_block_dfs_array,
                                 d_bwd_block_dfs_array,
                                 block_offsets,
                                 dfs_array,
                                 dfs_array_length,
                                 copier);

  CUDAFREE(d_fwd_flags, *logger);
  CUDAFREE(d_bwd_flags, *logger);
  CUDAFREE(d_fwd_block_dfs_array, *logger);
  CUDAFREE(d_bwd_block_dfs_array, *logger);
  CUDAFREE(d_ext_dfs_array, *logger);
  CUDAFREE(d_ext_block_offsets, *logger);

  int *d_graph_flags;
  CUDAMALLOC(&d_graph_flags, sizeof(int) * db_size, *logger);


  int *supp = new int[dfs_array_length];
  for( int i = 0; i < dfs_array_length; i++) {
    int block_id = dfs_array[i].from;
    int block_size;
    int *d_scan_array = 0;
    int mapped_db_size = 0;
    int scan_array_size = 0;
    bool compute_boundaries = true;

    if(block_id == num_blocks - 1) block_size = ext_array_length - block_offsets[block_id];
    else block_size = block_offsets[block_id + 1] - block_offsets[block_id];

    supp[i] = compute_support_remapped_db(d_ext_array + block_offsets[block_id],
                                          block_size,
                                          dfs_array[i],
                                          max_graph_vertex_count,
                                          d_graph_flags,
                                          d_scan_array,
                                          scan_array_size,
                                          compute_boundaries,
                                          mapped_db_size,
                                          scanner);
    std::cout << dfs_array[i] << "( support = " << supp[i]  << " )" << std::endl;

    CUDAFREE(d_scan_array,*logger);
  }

  //Now test results
  int len_res = 8;
  types::DFS dfs_res[] = { DFS(0, 3, 0, 1, 0),
                           DFS(0, 3, 0, 3, 1),
                           DFS(0, 3, 0, 2, 2),
                           DFS(1, 3, 0, 2, 1),
                           DFS(1, 3, 0, 0, 2),
                           DFS(1, 3, 0, 1, 2),
                           DFS(2, 3, 0, 1, 1),
                           DFS(2, 0, 0, 1, 0)};

  int supp_res[] = { 1, 1, 1, 1, 1, 2, 1, 1};

  EXPECT_EQ(dfs_array_length, len_res);
  for(int i = 0; i<dfs_array_length; i++) {
    EXPECT_EQ(dfs_array[i], dfs_res[i]);
    EXPECT_EQ(supp[i], supp_res[i]);
  }

  CUDAFREE(d_ext_array, *logger);
  CUDAFREE(d_graph_flags, *logger);
  delete scanner;
}


