#ifndef __GSPAN_CUDA_NO_SORT_HPP__
#define __GSPAN_CUDA_NO_SORT_HPP__

#include <gspan_cuda.hpp>

namespace gspan_cuda {

/**
 * Replace the sorting with computation in one block using reductions/scans. One extensions at a
 * time.
 *
 *
 */
class gspan_cuda_no_sort : public gspan_cuda {
protected:

  // The followings are used in extract_extensions function
  int *d_fwd_flags; // There would be at most |E|*|V| different fwd extensions
  int *d_bwd_flags;  // There would be at most |E|*|emb col| different bwd extensions, since we don't know the size in advance we allocate |E|*|V| at the beginning
  types::DFS *d_fwd_block_dfs_array; //These are for storing the different DFS extensions, the size is the same as above
  types::DFS *d_bwd_block_dfs_array;

  //Used in extract extensions function and resized if necessary inside the function
  int *d_ext_block_offsets;
  int d_ext_block_offsets_length;
  types::DFS *d_ext_dfs_array;
  int d_ext_dfs_array_length;


  virtual void prepare_run(types::edge_gid_list3_t &root);
  void fill_labels();

  //int compute_support(int from_label, int elabel, int to_label);

  struct embedding_extension_compare_less_then_t {
    types::DFS_less_then comp;
    bool operator()(const types::DFS &ee1, const types::DFS &ee2) const {
      bool less = comp(ee1, ee2);
      return less;
    } // operator()
  };

  typedef std::set<types::DFS, embedding_extension_compare_less_then_t> extension_set_t;

  int get_block_size(int block_id, const int *block_offsets, int num_blocks);

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


  void mainloop(types::embedding_list_columns &embeddings,
                types::DFSCode &code,
                int support,
                dfs_extension_element_set_t backward_edges,
                types::RMPath scheduled_rmpath,
                extension_set_t not_frequent_extensions);

  virtual void run_intern2();


  void test_embeddings(types::embedding_list_columns &embeddings,
                       types::DFSCode code,
                       types::RMPath scheduled_rmpath,
                       types::DFS dfs_elem,
                       types::embedding_element *d_embed_column_test,
                       int d_embed_column_test_length);

  void test_extensions(types::embedding_list_columns &embeddings,
                       types::DFSCode code,
                       types::RMPath scheduled_rmpath,
                       types::DFS *h_frequent_dfs_elem,
                       int *h_frequent_dfs_supports,
                       int frequent_candidate_count);

  int compute_support2(const types::edge_gid_list3_t &root, int from_label, int elabel, int to_label) const;

  struct embedding_element_less_then_t {
    bool operator()(const types::embedding_element &ee1, const types::embedding_element &ee2) {
      if(ee1.back_link > ee2.back_link) return false;
      if(ee1.back_link < ee2.back_link) return true;
      if(ee1.vertex_id < ee2.vertex_id) return true;
      return false;
    }
  };


public:
  gspan_cuda_no_sort();

  void set_edge_label_set(std::set<int> edge_label_set) {
    this->edge_label_set = edge_label_set;
  }

  void set_vertex_label_set(std::set<int> vertex_label_set) {
    this->vertex_label_set = vertex_label_set;
  }

  virtual void run();

};

} // namespace gspan_cuda

#endif

