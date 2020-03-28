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

#include <cuda_segmented_scan.hpp>

using std::string;
using namespace types;
using namespace gspan_cuda;
using std::cerr;

using std::fixed;

static Logger *value_logger = Logger::get_logger("VAL");
static Logger *logger = Logger::get_logger("MAIN");


TEST(filter_backward_embeddings, basic_test)
{
  cudaDeviceReset();

  types::embedding_element emb_col1[] = { embedding_element(2,-1),
                                          embedding_element(3,-1),
                                          embedding_element(17,-1),
                                          embedding_element(19,-1),
                                          embedding_element(22,-1),
                                          embedding_element(23,-1)};

  types::embedding_element emb_col2[] = { embedding_element(5,0),
                                          embedding_element(6,1),
                                          embedding_element(7,1),
                                          embedding_element(16,2),
                                          embedding_element(18,3),
                                          embedding_element(24,4),
                                          embedding_element(25,5)};

  types::embedding_element emb_col3[] = { embedding_element(1,0),
                                          embedding_element(4,1),
                                          embedding_element(8,1),
                                          embedding_element(11,3),
                                          embedding_element(12,3),
                                          embedding_element(27,5),
                                          embedding_element(28,5)};

  types::embedding_element *ptr[] = {emb_col1, emb_col2, emb_col3};
  int col_lens[] = {6, 7, 7};

  types::DFS dfs[] = { DFS(0, 1, 2, 0, 3), DFS(1, 2, 3, 0, 5) };

  types::embedding_list_columns emb_list_in(true);
  emb_list_in.columns_count = 3;
  emb_list_in.columns_lengths = col_lens;
  emb_list_in.columns = ptr;
  emb_list_in.dfscode = dfs;
  emb_list_in.dfscode_length = 2;

  types::embedding_list_columns d_emb_list_in(false), d_emb_list_out(false);
  emb_list_in.copy_to_device(&d_emb_list_in);
  d_emb_list_out = d_emb_list_in.d_get_half_copy();


  gspan_cuda::extension_element_t exts_array[] = {
    //                   DFS part       | graph/embedding part
    extension_element_t( 2, 0, 5, 0, 2,    1,  2, 0), //bwd
    extension_element_t( 2, 3, 5, 0, 1,    1,  9, 0),
    extension_element_t( 2, 3, 5, 0, 5,    4,  8, 1),
    extension_element_t( 2, 3, 5, 0, 5,    8,  4, 2),
    extension_element_t( 2, 3, 5, 0, 3,   11, 13, 3),
    extension_element_t( 2, 0, 5, 0, 2,   12, 17, 4), //bwd
    extension_element_t( 2, 3, 5, 0, 3,   12, 13, 4),
    extension_element_t( 2, 0, 5, 0, 2,   28, 22, 6) //bwd
  };
  int exts_len = 8;
  gspan_cuda::extension_element_t *d_exts_array;
  CUDAMALLOC(&d_exts_array, sizeof(extension_element_t) * exts_len, *logger);
  CUDA_EXEC(cudaMemcpy(d_exts_array, exts_array, sizeof(extension_element_t) * exts_len, cudaMemcpyHostToDevice),*logger);

  types::DFS dfs_elem(2, 0, 5, 0, 2);
  embedding_element *d_result_col = 0;
  int d_result_col_length = 0;

  cuda_segmented_scan *scanner = new cuda_segmented_scan();
  gspan_cuda::filter_backward_embeddings(d_emb_list_in, d_exts_array, exts_len, dfs_elem, d_result_col, d_result_col_length, scanner);
  delete scanner;
  d_emb_list_out.d_replace_last_column(d_result_col, d_result_col_length);


  std::cout <<  "Input Embeddings : " << emb_list_in.to_string() << std::endl;

  std::cout <<  "Input Extension Elements: " << std::endl;
  for(int i = 0; i < exts_len; i++ )
    std::cout << exts_array[i].to_string() << std::endl;
  std::cout << std::endl;

  std::cout <<  "Input DFS code : " << dfs_elem.to_string() << std::endl << std::endl;

  embedding_list_columns h_emb_list_out(true);
  h_emb_list_out.copy_from_device(&d_emb_list_out);
  std::cout <<  "Filtered Embeddings : " << h_emb_list_out.to_string() << std::endl;

  //Now test results
  int result_len = 3;
  types::embedding_element emb_res[] = { embedding_element(1,0),
                                         embedding_element(12,3),
                                         embedding_element(28,5)};

  EXPECT_EQ(result_len, h_emb_list_out.columns_lengths[2]);
  for(int i = 0; i < h_emb_list_out.columns_lengths[2]; i++)
    EXPECT_TRUE(h_emb_list_out.columns[2][i].vertex_id == emb_res[i].vertex_id && h_emb_list_out.columns[2][i].back_link == emb_res[i].back_link);

  //Free resources
  d_emb_list_out.d_half_deallocate();
  d_emb_list_in.delete_from_device();
  CUDAFREE(d_exts_array, *logger);
  CUDAFREE(d_result_col, *logger);

}


