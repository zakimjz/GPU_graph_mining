#ifndef __GSPAN_CUDA_FREQ_MINDFS_HPP__
#define __GSPAN_CUDA_FREQ_MINDFS_HPP__

#include <gspan_cuda_no_sort_block.hpp>
#include <logger.hpp>
#include <cuda_utils.hpp>

#define FREQ     0
#define NOT_FREQ 1
#define UNKNOWN  2

namespace gspan_cuda {


/**
 * Implements the three states (frequent, not frequent, unknown) for
 * each extensions. See the defines. THIS VERSION DOES NOT WORKS.
 *
 */
class gspan_cuda_freq_mindfs : public gspan_cuda_no_sort_block {
protected:
  Logger *logger;
  typedef std::map<types::DFS, int, embedding_extension_compare_less_then_t> extension_map_t;


/*
   virtual void filter_rmpath(const types::RMPath &gspan_rmpath_in,
                             types::RMPath &gspan_rmpath_out,
                             types::DFS *h_dfs_elem,
                             int *supports,
                             int h_dfs_elem_count);

   virtual void filter_rmpath(const types::RMPath &gspan_rmpath_in,
                             types::RMPath &gspan_rmpath_out,
                             types::RMPath &gspan_rmpath_has_extension);
 */

  virtual bool is_min_dfs_ext(const types::DFS &dfs_elem, const types::DFSCode &code) const;

  virtual void run_intern2();

  virtual void fill_supports(extension_element_t *d_exts_block,
                             int exts_block_length,
                             types::DFS *h_block_dfs_elems,
                             types::DFS *d_block_dfs_elems,
                             int dfs_elem_length,
                             int max_graph_vertex_count,
                             int mapped_db_size,
                             int col_count,
                             std::vector<types::DFS> &freq_dfs_elems,
                             std::vector<int> &freq_dfs_elems_supports,
                             extension_map_t &dfs_support_info);


  virtual void filter_extensions(extension_element_t *d_exts_result,
                                 int exts_result_length,
                                 types::DFS *h_dfs_elem,
                                 int dfs_array_length,
                                 int *ext_block_offsets,
                                 int ext_num_blocks,
                                 int col_count,
                                 types::DFSCode code,
                                 extension_map_t &dfs_support_info_in,
                                 extension_map_t &dfs_support_info_out,
                                 types::DFS *&h_frequent_dfs_elem,
                                 int *&h_frequent_dfs_supports,
                                 int &frequent_candidate_count);

  virtual void mainloop(types::embedding_list_columns &embeddings,
                        types::DFSCode &code,
                        int support,
                        dfs_extension_element_set_t backward_edges,
                        types::RMPath scheduled_rmpath,
                        extension_map_t &dfs_support_info);


public:
  gspan_cuda_freq_mindfs();
  virtual ~gspan_cuda_freq_mindfs();
};

} // namespace gspan_cuda

#endif

