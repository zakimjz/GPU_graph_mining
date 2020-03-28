#ifndef __GSPAN_CUDA_MULT_BLOCK_HPP__
#define __GSPAN_CUDA_MULT_BLOCK_HPP__

#include <gspan_cuda_no_sort.hpp>
#include <cuda_segmented_scan.hpp>


namespace gspan_cuda {

/**
 * This variant computes the support for extensions in multiple blocks at once.
 *
 *
 */
class gspan_cuda_mult_block : public gspan_cuda_no_sort {
protected:
  Logger *logger;

  int compute_support_max_memory_size;
  //int dfs_count_per_compute_support_call;

  //Used in remap_db and compute_support functions. resized if necessary in remap_db function
  int *d_graph_boundaries_scan;
  int d_graph_boundaries_scan_length;

  //Used in filter_extensions function and resized if necessary inside the function
  types::DFS *d_prefiltered_dfs_elems;
  int d_prefiltered_dfs_elems_array_size;



  virtual void fill_supports(extension_element_t *d_exts,
                             int exts_array_length,
                             types::DFS *h_dfs_elems,
                             types::DFS *d_dfs_elems,
                             int dfs_elem_length,
                             int max_graph_vertex_count,
                             int mapped_db_size,
                             std::vector<types::DFS> &freq_dfs_elems,
                             std::vector<int> &freq_dfs_elems_supports);

  virtual void filter_extensions(extension_element_t *d_exts_result,
                                 int exts_result_length,
                                 types::DFS *h_dfs_elem,
                                 int dfs_array_length,
                                 int *ext_block_offsets,
                                 int ext_num_blocks,
                                 int col_count,
                                 types::DFSCode code,
                                 extension_set_t &not_frequent_extensions,
                                 types::DFS *&h_frequent_dfs_elem,
                                 int *&h_frequent_dfs_supports,
                                 int &frequent_candidate_count);


  virtual bool should_filter_non_min(const types::DFS &dfs_elem, const types::DFSCode &code) const;
public:
  gspan_cuda_mult_block();
  virtual ~gspan_cuda_mult_block();
};

} // namespace gspan_cuda

#endif

