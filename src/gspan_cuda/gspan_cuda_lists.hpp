#ifndef __GSPAN_CUDA_LISTS_HPP__
#define __GSPAN_CUDA_LISTS_HPP__

#include <cuda_graph_types.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>
#include <graph_output.hpp>
#include <embedding_lists.hpp>
#include <cuda_gspan_ops.hpp>
#include <gspan_cuda.hpp>

namespace gspan_cuda {

/**
 * This variant of the gspan_cuda algorithm uses the lists of extensions and its intersections for
 * computing extended embeddings.
 *
 *
 */
class gspan_cuda_lists : public gspan_cuda {
protected:
  typedef std::set<embedding_extension_t> list_extensions_set_t;
  typedef std::vector<embedding_extension_t> list_extensions_vec_t;

  struct new_extensions_t {
    types::DFS *h_dfs_elem;
    int *h_dfs_offsets;
    int *h_support;
    int dfs_array_length;
    extension_element_t *d_exts_result;
    int exts_result_length;

    new_extensions_t() {
      h_dfs_elem = 0;
      h_dfs_offsets = 0;
      h_support = 0;
      dfs_array_length = 0;
      d_exts_result = 0;
      exts_result_length = 0;
    }

    void delete_from_device() {
      CUDAFREE(d_exts_result, *Logger::get_logger("GSPAN_CUDA_LISTS"));
      d_exts_result = 0;
      delete [] h_dfs_elem;
      delete [] h_dfs_offsets;
      h_dfs_elem = 0;
      h_dfs_offsets = 0;

      dfs_array_length = 0;
      exts_result_length = 0;
    }
  };

  struct embedding_extension_compare_less_then_t_neel {
    dfs_comparator<types::DFS> comp;
    bool operator()(const embedding_extension_t &ee1, const embedding_extension_t &ee2) const {
      bool less = comp(ee1.dfs_elem, ee2.dfs_elem) < 0 ? true : false;
      return less;
    } // operator()
  };

  struct embedding_extension_compare_less_then_t_kessl {
    types::DFS_less_then comp;
    bool operator()(const embedding_extension_t &ee1, const embedding_extension_t &ee2) const {
      bool less = comp(ee1.dfs_elem, ee2.dfs_elem);
      return less;
    } // operator()
  };

  typedef embedding_extension_compare_less_then_t_kessl embedding_extension_compare_less_then_t;

  void filter_extensions(types::DFSCode code, list_extensions_vec_t &extensions, list_extensions_vec_t &result, int ext_start, types::DFS last_edge);
  bool possible_extension(types::DFSCode code, types::RMPath rmpath, types::DFS dfs_to_test);
  void test_extensions(list_extensions_vec_t extensions_to_test, types::embedding_list_columns &embeddings, types::DFSCode code);

  void mainloop(types::embedding_list_columns &embeddings,
                types::DFSCode &code,
                dfs_extension_element_set_t backward_edges,
                list_extensions_vec_t extensions);




  /**
   * This function takes the embeddings, the cuda_rmpath, and the
   * scheduled_rmpath and calls the get_all_extensions and then
   * get_support_for_extensions and fills the new_extensions_t (which
   * is just a container for a lot of data.
   *
   */
  void get_rmpath_extensions(types::embedding_list_columns &embeddings,
                             types::DFSCode code,
                             types::RMPath cuda_rmpath,
                             types::RMPath scheduled_rmpath,
                             new_extensions_t &ne);


  /**
   * Takes Its arguments, calls the get_rmpath_extensions(emb, code,
   * cuda_rmpath, scheduled_rmpath, new_extensions_t) and converts its
   * result to list_extensions_vec_t. I.e., it computes the extensions
   * for the given scheduled_rmpath vertices.
   *
   */
  void get_rmpath_extensions(types::embedding_list_columns &embeddings,
                             types::DFSCode code,
                             types::RMPath cuda_rmpath,
                             types::RMPath scheduled_rmpath,
                             list_extensions_vec_t &exts);

  /**
   * Computes the extension starting from the right-most vertex only !
   * It computes the extension using the get_rmpath_extensions
   * functions.
   *
   */
  void get_rmpath_vertex_extensions(types::embedding_list_columns &embeddings,
                                    types::DFSCode code,
                                    list_extensions_vec_t &exts);

  /**
   * Computes the extensions from the whole right-most path in the
   * given dfscode code using the other get_rmpath_vertex_extensions.
   *
   */
  void get_rmpath_extensions(types::embedding_list_columns &embeddings,
                             types::DFSCode code,
                             list_extensions_vec_t &exts);

  void run_intern(void);
  void run_intern2();

  void create_new_extensions(list_extensions_vec_t &extensions,
                             int start_idx,
                             list_extensions_vec_t &new_extensions,
                             embedding_extension_t &d_filtered_last_col,
                             const types::embedding_list_columns &embeddings,
                             const types::DFSCode &code,
                             const types::RMPath &rmpath);

  void filter_non_minimal_extensions(types::DFSCode code, list_extensions_vec_t &extensions);

  Logger *logger;
public:
  gspan_cuda_lists();
  virtual ~gspan_cuda_lists();
  virtual void run();


  struct copy_backlink_info {
    int *dest;
    extension_element_t *src;
    int src_offset;

    copy_backlink_info(int *dest, extension_element_t * src, int src_offset) {
      this->dest = dest;
      this->src = src;
      this->src_offset = src_offset;
    } // copy_embedding_info

    __host__ __device__
    void operator()(int idx) {
      dest[idx] = src[src_offset + idx].row;
    } // operator()
  };
};

} // namespace gspan_cuda



#endif
