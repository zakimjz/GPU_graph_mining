#include <embedding_lists.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_computation_parameters.hpp>
#include <kernel_execution.hpp>
#include <cuda_gspan_ops.hpp>
#include <cuda_datastructures.hpp>

#include <cuda_tools.hpp>
#include <cuda.h>

#include <dfs_code.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>

#include <host_operations.hpp>

//using namespace types;

namespace gspan_cuda {

static Logger *logger = Logger::get_logger("MAINLOOP_SUPPORT");


struct store_dfs_validity {
  extension_element_t *d_exts;
  int *d_indices;
  types::DFS dfs_elem;
  store_dfs_validity(extension_element_t *d_exts, int *d_indices, types::DFS dfs_elem) {
    this->d_exts = d_exts;
    this->d_indices = d_indices;
    this->dfs_elem = dfs_elem;
  }

  __host__ __device__
  void operator()(int tid) {
    if(d_exts[tid].from == dfs_elem.from &&
       d_exts[tid].to == dfs_elem.to &&
       d_exts[tid].fromlabel == dfs_elem.fromlabel &&
       d_exts[tid].elabel == dfs_elem.elabel &&
       d_exts[tid].tolabel == dfs_elem.tolabel) {
      d_indices[tid] = 1;
    } else {
      d_indices[tid] = 0;
    }
  } // operator()
};

struct copy_valid_dfs {
  extension_element_t *d_exts;
  types::embedding_element *emb_col;
  int *d_indices;
  int *d_offsets;

  copy_valid_dfs(extension_element_t *d_exts, types::embedding_element *emb_col, int *d_indices, int *d_offsets) {
    this->d_exts = d_exts;
    this->emb_col = emb_col;
    this->d_indices = d_indices;
    this->d_offsets = d_offsets;
  } // copy_valid_dfs

  __host__ __device__
  void operator()(int tid) {
    if(d_indices[tid] == 1) {
      emb_col[d_offsets[tid]].vertex_id = d_exts[tid].to_grph;
      emb_col[d_offsets[tid]].back_link = d_exts[tid].row;
    } // if
  } // operator()
};


void extract_embedding_column(gspan_cuda::extension_element_t *d_exts,
                              int d_exts_size,
                              types::DFS dfs_elem,
                              types::embedding_element *&emb_col,
                              int &emb_col_size,
                              cuda_segmented_scan *scanner)
{
  int *d_indices = 0;
  int *d_offsets = 0;
  CUDAMALLOC(&d_indices, d_exts_size * sizeof(int), *logger);
  CUDAMALLOC(&d_offsets, d_exts_size * sizeof(int), *logger);
  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e(d_exts_size);
  thrust::for_each(b, e, store_dfs_validity(d_exts, d_indices, dfs_elem));

  //thrust::exclusive_scan(thrust::device_pointer_cast<int>(d_indices),
  //thrust::device_pointer_cast<int>(d_indices + d_exts_size),
  //thrust::device_pointer_cast<int>(d_offsets));
  scanner->scan((uint*)d_indices, (uint*) d_offsets, d_exts_size, EXCLUSIVE);

  int last_valid = 0;
  emb_col_size = 0;
  CUDA_EXEC(cudaMemcpy(&last_valid, d_indices + d_exts_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(&emb_col_size, d_offsets + d_exts_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  emb_col_size += last_valid;

  CUDAMALLOC(&emb_col, sizeof(types::embedding_element) * emb_col_size, *logger);
  thrust::for_each(thrust::counting_iterator<int>(0),
                   thrust::counting_iterator<int>(d_exts_size),
                   copy_valid_dfs(d_exts, emb_col, d_indices, d_offsets));

  CUDAFREE(d_indices, *logger);
  CUDAFREE(d_offsets, *logger);
}


void compact_labels(types::graph_database_cuda &gdb, std::set<int> &vertex_label_set, std::set<int> &edge_label_set)
{
  std::map<int, int> edges_labels;
  std::map<int, int> vertex_labels;
  int last_edge_label = 0;
  int last_vertex_label = 0;

  vertex_label_set.clear();
  edge_label_set.clear();

  // collect all vertex labels and remap them into a range 0..|vertex labels|
  for(int i = 0; i < gdb.db_size * gdb.max_graph_vertex_count; i++) {
    if(gdb.vertex_labels[i] == -1) continue;
    if(vertex_labels.find(gdb.vertex_labels[i]) != vertex_labels.end()) continue;

    vertex_labels[gdb.vertex_labels[i]] = last_vertex_label;
    vertex_label_set.insert(last_vertex_label);
    last_vertex_label++;
  } // for i

  // collect all edge labels and remap them into a range 0..|edge labels|
  for(int i = 0; i < gdb.edges_sizes; i++) {
    if(gdb.edges_labels[i] == -1) continue;
    if(edges_labels.find(gdb.edges_labels[i]) != edges_labels.end()) continue;

    edges_labels[gdb.edges_labels[i]] = last_edge_label;
    edge_label_set.insert(last_edge_label);
    last_edge_label++;
  } // for i


  // remap the vertex labels
  for(int i = 0; i < gdb.db_size * gdb.max_graph_vertex_count; i++) {
    if(gdb.vertex_labels[i] == -1) continue;

    gdb.vertex_labels[i] = vertex_labels[gdb.vertex_labels[i]];
  } // for i


  // remap the edge labels
  for(int i = 0; i < gdb.edges_sizes; i++) {
    if(gdb.edges_labels[i] == -1) continue;

    gdb.edges_labels[i] = edges_labels[gdb.edges_labels[i]];
  } // for i

}


} // namespace gspan_cuda


