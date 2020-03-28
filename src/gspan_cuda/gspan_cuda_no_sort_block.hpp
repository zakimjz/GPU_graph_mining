#ifndef __GSPAN_CUDA_NO_SORT_BLOCK_HPP__
#define __GSPAN_CUDA_NO_SORT_BLOCK_HPP__

#include <gspan_cuda_no_sort.hpp>
#include <cuda_segmented_reduction.hpp>


namespace gspan_cuda {


/**
 * Computes one block at a time, i.e., multiple extensions in one block at once.
 *
 */
class gspan_cuda_no_sort_block : public gspan_cuda_no_sort {
protected:
  Logger *logger;

  int compute_support_max_memory_size;
  //int dfs_count_per_compute_support_call;

  int *d_graph_boundaries_scan;
  int d_graph_boundaries_scan_length;

  void fill_supports(extension_element_t *d_exts_block,
                     int exts_block_length,
                     types::DFS *h_block_dfs_elems,
                     types::DFS *d_block_dfs_elems,
                     int dfs_elem_length,
                     int max_graph_vertex_count,
                     int mapped_db_size,
                     std::vector<types::DFS> &freq_dfs_elems,
                     std::vector<int> &freq_dfs_elems_supports);


  virtual void filter_extensions(extension_element_t *d_exts_result,
                                 int exts_result_length,types::DFS *h_dfs_elem,
                                 int dfs_array_length,
                                 int *block_offsets,
                                 int num_blocks,
                                 int col_count,
                                 types::DFSCode code,
                                 extension_set_t &not_frequent_extensions,
                                 types::DFS *&h_frequent_dfs_elem,
                                 int *&h_frequent_dfs_supports,
                                 int &frequent_candidate_count);

  virtual bool should_filter_non_min(const types::DFS &dfs_elem, const types::DFSCode &code) const;
public:
  gspan_cuda_no_sort_block();
  virtual ~gspan_cuda_no_sort_block();
};

} // namespace gspan_cuda

#endif

