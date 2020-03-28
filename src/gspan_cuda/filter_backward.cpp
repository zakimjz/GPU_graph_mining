#include <embedding_lists.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_computation_parameters.hpp>
#include <kernel_execution.hpp>
#include <cuda_gspan_ops.hpp>

#include <embedding_lists.hpp>

#include <cuda_tools.hpp>
#include <cuda.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include <host_operations.hpp>

using namespace types;

namespace gspan_cuda {

static Logger *logger = Logger::get_logger("BWD_FILTER");

struct store_col_indice {
  extension_element_t *exts;
  int *indices;

  store_col_indice(extension_element_t *exts, int *indices) {
    this->exts = exts;
    this->indices = indices;
  } // store_col_indice

  __host__ __device__
  void operator()(int thread_idx) {
    int from_row = exts[thread_idx].row;
    indices[from_row] = 1;
  } // operator()
};

struct copy_embedding_rows {
  embedding_element *dst;
  embedding_element *src;
  int *index;
  int *indices;
  copy_embedding_rows(embedding_element *dst, embedding_element *src, int *index, int *indices)
  {
    this->dst = dst;
    this->src = src;
    this->index = index;
    this->indices = indices;
  }

  __host__ __device__
  void operator()(int thread_idx) {
    if(indices[thread_idx] == 1) {
      dst[index[thread_idx]] = src[thread_idx];
    }
  } // operator()
};





struct copy_embedding_rows_no_indices {
  embedding_element *dst;
  embedding_element *src;
  int *index;
  copy_embedding_rows_no_indices(embedding_element * dst, embedding_element * src, int *index)
  {
    this->dst = dst;
    this->src = src;
    this->index = index;
  }

  __host__ __device__
  void operator()(int thread_idx) {
    dst[thread_idx] = src[index[thread_idx]];
  } // operator()

};


void filter_backward_embeddings(types::embedding_element *&input_embeddings,
                                int input_embeddings_length,
                                types::embedding_element *&filtered_embeddings,
                                int &filtered_embeddings_length,
                                int *d_input_offsets,
                                int d_input_offsets_length,
                                cudapp::cuda_configurator *exec_conf)
{
  cudapp::cuda_computation_parameters store_col_indices_params = exec_conf->get_exec_config("filter_backward_embeddings.store_col_indices",
                                                                                            d_input_offsets_length);

  TRACE(*logger, "filtered_embeddings_length: " << filtered_embeddings_length
        << "; input_embeddings_length: " << input_embeddings_length
        << "; d_input_offsets_length: " << d_input_offsets_length);
  // copy the rows
  cudapp::cuda_computation_parameters copy_rows_params = exec_conf->get_exec_config("filter_backward_embeddings.copy_rows",
                                                                                    input_embeddings_length);
  CUDAMALLOC(&filtered_embeddings, sizeof(embedding_element) * d_input_offsets_length, *logger);

  copy_embedding_rows_no_indices copy_rows(filtered_embeddings, input_embeddings, d_input_offsets);
  cudapp::for_each(0, d_input_offsets_length, copy_rows, copy_rows_params);
  filtered_embeddings_length = d_input_offsets_length;
}




void filter_backward_embeddings(types::embedding_list_columns &embeds_in,
                                embedding_extension_t &backward_extension,
                                cudapp::cuda_configurator *exec_conf)
{
  int last_col_length = 0;
  CUDA_EXEC(cudaMemcpy(&last_col_length, embeds_in.columns_lengths + embeds_in.columns_count - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);

  embedding_element *last_col = 0;
  CUDA_EXEC(cudaMemcpy(&last_col, embeds_in.columns + embeds_in.columns_count - 1, sizeof(embedding_element*), cudaMemcpyDeviceToHost), *logger);

  filter_backward_embeddings(last_col, last_col_length,
                             backward_extension.embedding_column, backward_extension.col_length, //filtered_last_col, filtered_last_col_len,
                             backward_extension.filtered_emb_offsets, backward_extension.filtered_emb_offsets_length,
                             exec_conf);
}



void filter_backward_embeddings(types::embedding_element *&input_embeddings,
                                int input_embeddings_length,
                                types::embedding_element *&filtered_embeddings,
                                int &filtered_embeddings_length,
                                int ext_index,
                                extension_element_t *exts,
                                int exts_length,
                                int *h_dfs_offsets,
                                types::DFS *h_dfs_elem,
                                int dfs_array_length,
                                cuda_segmented_scan *scanner,
                                cudapp::cuda_configurator *exec_conf)
{
  int embedding_count = 0;
  if(ext_index == dfs_array_length - 1) {
    embedding_count = exts_length - h_dfs_offsets[ext_index];
  } else embedding_count = h_dfs_offsets[ext_index + 1] - h_dfs_offsets[ext_index];


  int *input_embed_filter_indices = 0;
  int *input_embed_indexes = 0;
  CUDAMALLOC(&input_embed_filter_indices, sizeof(int) * input_embeddings_length, *logger);
  CUDAMALLOC(&input_embed_indexes, sizeof(int) * input_embeddings_length, *logger);
  CUDA_EXEC(cudaMemset(input_embed_filter_indices, 0, sizeof(int) * input_embeddings_length), *logger);
  CUDA_EXEC(cudaMemset(input_embed_indexes, 0, sizeof(int) * input_embeddings_length), *logger);

  cudapp::cuda_computation_parameters store_col_indices_params = exec_conf->get_exec_config("filter_backward_embeddings.store_col_indices",
                                                                                            embedding_count);
  // find which row should be copied (by storing 0/1 indices)
  TRACE(*logger, "storing valid row indices");
  cudapp::for_each(0, embedding_count, store_col_indice(exts + h_dfs_offsets[ext_index], input_embed_filter_indices), store_col_indices_params);
  TRACE(*logger, "input_embed_filter_indices: " << print_d_array(input_embed_filter_indices, input_embeddings_length));
  TRACE(*logger, "calling exclusive scan on the row indices");
  // compute indexes using exclusive_scan
  //thrust::exclusive_scan(thrust::device_pointer_cast<int>(input_embed_filter_indices),
  //thrust::device_pointer_cast<int>(input_embed_filter_indices + input_embeddings_length),
  //thrust::device_pointer_cast<int>(input_embed_indexes),
  //0,
  //thrust::plus<int>());
  scanner->scan((uint*)input_embed_filter_indices, (uint*)input_embed_indexes, input_embeddings_length, EXCLUSIVE);

  // copy information from the previous step to host
  int last_used = 0;
  CUDA_EXEC(cudaMemcpy(&filtered_embeddings_length, input_embed_indexes + input_embeddings_length - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(&last_used, input_embed_filter_indices + input_embeddings_length - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  filtered_embeddings_length += last_used;

  TRACE(*logger, "filtered_embeddings_length: " << filtered_embeddings_length);

  // copy the rows
  cudapp::cuda_computation_parameters copy_rows_params = exec_conf->get_exec_config("filter_backward_embeddings.copy_rows",
                                                                                    input_embeddings_length);
  CUDAMALLOC(&filtered_embeddings, sizeof(embedding_element) * filtered_embeddings_length, *logger);

  copy_embedding_rows copy_rows(filtered_embeddings, input_embeddings, input_embed_indexes, input_embed_filter_indices);
  cudapp::for_each(0, input_embeddings_length, copy_rows, copy_rows_params);


  CUDAFREE(input_embed_filter_indices, *logger);
  CUDAFREE(input_embed_indexes, *logger);
}


void filter_backward_embeddings(types::embedding_list_columns &embeds_in,
                                int ext_index,
                                extension_element_t *exts,
                                int exts_length,
                                int *h_dfs_offsets,
                                types::DFS *h_dfs_elem,
                                int dfs_array_length,
                                types::embedding_element *&filtered_embeddings,
                                int &filtered_embeddings_length,
                                cuda_segmented_scan *scanner,
                                cudapp::cuda_configurator *exec_conf)
{
  int embeds_in_last_col_length = 0;
  CUDA_EXEC(cudaMemcpy(&embeds_in_last_col_length, embeds_in.columns_lengths + embeds_in.columns_count - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);

  embedding_element *last_col = 0;
  CUDA_EXEC(cudaMemcpy(&last_col, embeds_in.columns + embeds_in.columns_count - 1, sizeof(embedding_element*), cudaMemcpyDeviceToHost), *logger);

  embedding_element *dst = 0;
  int dst_length = 0;

  filter_backward_embeddings(last_col,
                             embeds_in_last_col_length,
                             dst,
                             dst_length,
                             ext_index,
                             exts,
                             exts_length,
                             h_dfs_offsets,
                             h_dfs_elem,
                             dfs_array_length,
                             scanner,
                             exec_conf);


  //embedding_element **d_columns = embeds_out.columns;
  //CUDA_EXEC(cudaMemcpy(d_columns + embeds_out.columns_count - 1, &dst, sizeof(embedding_element*), cudaMemcpyHostToDevice), *logger);
  //CUDA_EXEC(cudaMemcpy(embeds_out.columns_lengths + embeds_out.columns_count - 1, &dst_length, sizeof(int), cudaMemcpyHostToDevice), *logger);

  //TRACE5(*logger, "columns count: " << embeds_out.columns_count);
  filtered_embeddings = dst;
  filtered_embeddings_length = dst_length;
} // filter_backward_embeddings










struct store_bwd_extension_indices {
  int *indices;
  extension_element_t *d_exts;
  types::DFS dfs_elem;

  store_bwd_extension_indices(int *indices, extension_element_t *d_exts, types::DFS dfs_elem) {
    this->indices = indices;
    this->d_exts = d_exts;
    this->dfs_elem = dfs_elem;
  }

  __host__ __device__
  void operator()(int thread_idx) {
    if(d_exts[thread_idx].from == dfs_elem.from &&
       d_exts[thread_idx].to == dfs_elem.to &&
       d_exts[thread_idx].fromlabel == dfs_elem.fromlabel &&
       d_exts[thread_idx].elabel == dfs_elem.elabel &&
       d_exts[thread_idx].tolabel == dfs_elem.tolabel)
    {
      indices[d_exts[thread_idx].row] = 1;
    }   // if
  } // operator()
};


/**
 *
 * Takes the exts that contains dfs_elem and stores 0/1 indices in
 * input_embed_filter_indices on position 'extension_element_t::row'.
 * Then it scans the input_embed_filter_indices and stores the scanned
 * array in input_embed_indexes. Then it copies elements with 1 in the
 * input_embed_filter_indices in last column of embeds_in into new
 * embedding_element column that replace the last column in
 * embeds_out.
 *
 * @param embeds_in   input embeddings
 * @param embeds_out  output embeddings
 * @param exts        block of extensions having the same from pattern id
 * @param exts_length size of exts
 * @param dfs_elem    the dfs element that is filtered
 * @param new_column  the resulting (filtered) column. This is for further deallocation.
 *
 */
void filter_backward_embeddings(types::embedding_list_columns &embeds_in,
                                extension_element_t *d_exts,
                                int exts_length,
                                types::DFS dfs_elem,
                                embedding_element *&new_column,
                                int &new_column_length,
                                cuda_segmented_scan *scanner)
{
  if(dfs_elem.is_forward()) {
    abort();
  }

  TRACE(*logger, "block: " << print_d_array(d_exts, exts_length));


  int embeds_in_last_col_size = 0;
  CUDA_EXEC(cudaMemcpy(&embeds_in_last_col_size, embeds_in.columns_lengths + embeds_in.columns_count - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  TRACE(*logger, "embeds_in_last_col_size: " << embeds_in_last_col_size);

  int *input_embed_filter_indices = 0;
  int *input_embed_indexes = 0;
  CUDAMALLOC(&input_embed_filter_indices, sizeof(int) * embeds_in_last_col_size, *logger);
  CUDAMALLOC(&input_embed_indexes, sizeof(int) * embeds_in_last_col_size, *logger);
  CUDA_EXEC(cudaMemset(input_embed_filter_indices, 0, sizeof(int) * embeds_in_last_col_size), *logger);
  CUDA_EXEC(cudaMemset(input_embed_indexes, 0, sizeof(int) * embeds_in_last_col_size), *logger);

  thrust::for_each(thrust::counting_iterator<int>(0),
                   thrust::counting_iterator<int>(exts_length),
                   store_bwd_extension_indices(input_embed_filter_indices, d_exts, dfs_elem));

  TRACE(*logger, "input_embed_filter_indices: " << print_d_array(input_embed_filter_indices, embeds_in_last_col_size));

  //thrust::exclusive_scan(thrust::device_pointer_cast<int>(input_embed_filter_indices),
  //thrust::device_pointer_cast<int>(input_embed_filter_indices + embeds_in_last_col_size),
  //thrust::device_pointer_cast<int>(input_embed_indexes),
  //0,
  //thrust::plus<int>());
  scanner->scan((uint*)input_embed_filter_indices, (uint*)input_embed_indexes, embeds_in_last_col_size, EXCLUSIVE);

  int filtered_col_length = 0;
  int last_filter_indice = 0;
  CUDA_EXEC(cudaMemcpy(&filtered_col_length, input_embed_indexes + embeds_in_last_col_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(&last_filter_indice, input_embed_filter_indices + embeds_in_last_col_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  filtered_col_length += last_filter_indice;

  embedding_element *filtered_col = 0;
  embedding_element *last_col = 0;
  //int last_col_length;


  CUDAMALLOC(&filtered_col, sizeof(embedding_element) * filtered_col_length, *logger);
  CUDA_EXEC(cudaMemcpy(&last_col, embeds_in.columns + embeds_in.columns_count - 1, sizeof(embedding_element*), cudaMemcpyDeviceToHost), *logger);
  //CUDA_EXEC(cudaMemcpy(&last_col_length, embeds_in.columns_lengths + embeds_in.columns_count - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  //copy_embedding_rows(embedding_element *dst, embedding_element *src, int *index, int *indices)

  TRACE(*logger, "last_col: " << print_d_array(last_col, embeds_in_last_col_size));

  copy_embedding_rows copy_rows(filtered_col, last_col, input_embed_indexes, input_embed_filter_indices);
  thrust::for_each(thrust::counting_iterator<int>(0),
                   thrust::counting_iterator<int>(embeds_in_last_col_size),
                   copy_rows);

  new_column = filtered_col;
  new_column_length = filtered_col_length;

  CUDAFREE(input_embed_filter_indices, *logger);
  CUDAFREE(input_embed_indexes, *logger);
} // filter_backward_embeddings



} // namespace gspan_cuda

