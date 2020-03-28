#ifndef __GSPAN_CUDA_HPP__
#define __GSPAN_CUDA_HPP__

#include <cuda_graph_types.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>
#include <graph_output.hpp>
#include <embedding_lists.hpp>
#include <cuda_gspan_ops.hpp>
#include <cuda_functors.hpp>

bool check_one_embedding(types::graph_database_t &gdb,
                         types::graph_database_cuda &cuda_gdb,
                         types::RMPath rmpath,
                         types::embedding_list_columns h_embeddings,
                         int row,
                         gspan_cuda::extension_element_t *extensions,
                         int extensions_length);


void check_host_db_agains_embedding(const types::graph_database_t &gdb, types::DFS dfs, const types::embedding_list_columns &h_embed);
void check_embedding_against_host_db(const types::graph_database_t &gdb, types::DFS dfs, const types::embedding_list_columns &h_embed);

namespace gspan_cuda {

struct copy_embedding_info {
  types::embedding_element *dest;
  extension_element_t *src;
  int src_offset;

  copy_embedding_info(types::embedding_element *dest, extension_element_t *src, int src_offset) {
    this->dest = dest;
    this->src = src;
    this->src_offset = src_offset;
  } // copy_embedding_info


  __host__ __device__
  void operator()(int idx) {
    dest[idx].vertex_id = src[src_offset + idx].to_grph;
    dest[idx].back_link = src[src_offset + idx].row;
  } // operator()
};



/**
 * The vanilla flavor of the gspan_cuda algorithm. It uses get_all_extensions in combination with
 * get_support_for_extensions that uses thrust::sort for extraction of extensions and computation of
 * support. This is a slow variant.
 *
 */
class gspan_cuda {
protected:

  typedef std::set<types::DFS, types::DFS_less_then_fast> dfs_extension_element_set_t;
  typedef std::map<types::DFS, int, types::DFS_less_then_fast> dfs_extension_element_map_t;


  int minimal_support;
  types::graph_database_t graph_database;
  graph_output * output;
  cudapp::cuda_configurator *exec_config;
  types::graph_database_cuda h_cuda_graph_database;
  types::graph_database_cuda d_cuda_graph_database;
  cuda_allocs_for_get_all_extensions cuda_allocs_get_all_ext;

  cuda_segmented_scan *scanner;
  cuda_segmented_reduction *reduction;
  cuda_copy<types::DFS, is_one_array> *copier;

  Logger *logger;

  bool execute_tests;

  void test_embeddings(types::embedding_list_columns &d_embeddings, types::RMPath rmpath, extension_element_t *d_extensions, int extensions_length);
  void test_database();

  void report_single(types::Graph &g, std::map<unsigned int, unsigned int>& ncount);

  void fill_root(types::edge_gid_list3_t &root);

  virtual void run_intern2();
  void mainloop(types::embedding_list_columns &embeddings,
                types::DFSCode &code,
                types::Projected &projected,
                int support,
                dfs_extension_element_set_t backward_edges,
                types::RMPath scheduled_rmpath);

  types::RMPath convert_rmpath(const types::RMPath &gspan_rmpath, types::DFSCode &code);

  struct create_extension_column {
    extension_element_t *extensions;
    int offset;
    types::embedding_element *result;
    create_extension_column(extension_element_t *exts, int off) {
      extensions = exts;
      offset = off;
    } // create_extension_column

    __device__ __host__
    void operator() (int idx) {
      result[idx].vertex_id = extensions[offset + idx].to_grph;
      result[idx].back_link = extensions[offset + idx].row;
    } // operator()
  };


  unsigned int support(types::Projected &projected);

  bool test_supports2(types::Projected &projected,
                      types::DFSCode code,
                      types::DFS *h_dfs_elem,
                      int *h_support,
                      int size,
                      extension_element_t *d_exts_result,
                      int exts_result_length);


  void test_supports(types::DFSCode code, types::DFS *h_dfs_elem, int size, extension_element_t *d_exts_result, int exts_result_length);
  void get_all_extensions_orig(types::Projected &projected, types::DFSCode DFS_CODE, dfs_extension_element_map_t &all_dfs_elements);
  void get_new_projected(types::Projected &old_projected, types::DFSCode DFS_CODE, types::DFS dfs_elem, types::Projected &new_projected);


  virtual void filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::DFS *h_dfs_elem, int *supports, int h_dfs_elem_count);
  virtual void filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::RMPath &gspan_rmpath_has_extension);


  int *d_graph_flags;  // Used in compute_support function, reduction of this array gives the support an extension
  int d_graph_flags_length;

  std::set<int> edge_label_set;
  std::set<int> vertex_label_set;

  int max_elabel;
  int max_vlabel;

  virtual void prepare_run(types::edge_gid_list3_t &root);

  void fill_root_cuda(types::edge_gid_list3_t &root);
  void fill_labels_cuda();
  int compute_support(int from_label, int elabel, int to_label);

  void fill_host_projected(types::Projected_map3 *root);
public:
  struct store_edge_label {
    int *d_edge_labels_set;
    int *db_edge_labels;

    store_edge_label(int *d_edge_labels_set, int *db_edge_labels) {
      this->d_edge_labels_set = d_edge_labels_set;
      this->db_edge_labels = db_edge_labels;
    } // store_edge_info

    __host__ __device__
    void operator()(int idx) {
      int edge_label = db_edge_labels[idx];
      if(edge_label != -1) {
        d_edge_labels_set[edge_label] = 1;
      } // if
    } // operator()
  };



  struct store_vertex_label {
    int *d_vertex_labels_set;
    int *db_vertex_labels;

    store_vertex_label(int *d_vertex_labels_set, int *db_vertex_labels) {
      this->d_vertex_labels_set = d_vertex_labels_set;
      this->db_vertex_labels = db_vertex_labels;
    } // store_edge_info

    __host__ __device__
    void operator()(int idx) {
      int vertex_label = db_vertex_labels[idx];
      if(vertex_label != -1) {
        db_vertex_labels[vertex_label] = 1;
      }
    } // operator()
  };


  struct store_edge_flag {
    types::graph_database_cuda gdb;
    int *edge_flags;
    int max_vlabel;
    int max_elabel;
    store_edge_flag(types::graph_database_cuda db, int *edge_flags, int mvl, int mel) : gdb(db) {
      this->edge_flags = edge_flags;
      //gdb = db;
      max_vlabel = mvl;
      max_elabel = mel;
    }

    __host__ __device__
    void operator()(int idx) {
      if(gdb.vertex_is_valid(idx)) {
        int degree = gdb.get_vertex_degree(idx);
        int from_vlabel = gdb.get_vertex_label(idx);
        int from_neigh_off = gdb.get_neigh_offsets(idx);
        for(int i = 0; i < degree; i++) {
          int to_idx = gdb.edges[from_neigh_off + i];
          int to_vlabel = gdb.get_vertex_label(to_idx);
          int elabel = gdb.edges_labels[from_neigh_off + i];
          int flag_idx = elabel * (max_vlabel + 1) * (max_vlabel + 1) + to_vlabel * (max_vlabel + 1) + from_vlabel;
          edge_flags[flag_idx] = 1;
        } // for i
      } // if
    } // operator()
  };


  struct store_extension {
    int max_vlabel;
    int max_elabel;
    int *all_edges_flags;
    int *all_edges_flags_scanned;
    extension_element_t *result;

    store_extension(int mvl, int mel, int *eflags, int *eflags_sanned, extension_element_t *result) {
      this->max_vlabel = mvl + 1;
      this->max_elabel = mel + 1;
      this->all_edges_flags = eflags;
      this->all_edges_flags_scanned = eflags_sanned;
      this->result = result;
    }

    __host__ __device__
    void operator()(int idx) {
      if(all_edges_flags[idx] == 1) {
        int tmp_idx = idx;
        int elabel = idx / ((max_vlabel) * (max_vlabel));
        idx = idx - elabel * ((max_vlabel) * (max_vlabel));
        int to_vlabel = idx / (max_vlabel);
        int from_vlabel = idx - to_vlabel * (max_vlabel);
        int dest_idx = all_edges_flags_scanned[tmp_idx];
        result[dest_idx].fromlabel = from_vlabel;
        result[dest_idx].elabel = elabel;
        result[dest_idx].tolabel = to_vlabel;
      } // if
    } // operator()
  };


  struct store_graph_id {
    int *graph_flags;
    extension_element_t edge_labels;
    types::graph_database_cuda gdb;

    store_graph_id(int *graph_flags, extension_element_t edge_labels, types::graph_database_cuda gdb) : gdb(gdb){
      this->graph_flags = graph_flags;
      this->edge_labels = edge_labels;
    }

    __host__ __device__
    void operator()(int idx) {
      if(gdb.vertex_is_valid(idx)) {
        int degree = gdb.get_vertex_degree(idx);
        int from_vlabel = gdb.get_vertex_label(idx);
        int from_neigh_off = gdb.get_neigh_offsets(idx);
        if(from_vlabel != edge_labels.fromlabel) return;
        for(int i = 0; i < degree; i++) {
          int to_idx = gdb.edges[from_neigh_off + i];
          int to_vlabel = gdb.get_vertex_label(to_idx);

          if(to_vlabel != edge_labels.tolabel) continue;
          int elabel = gdb.edges_labels[from_neigh_off + i];
          if(elabel == edge_labels.elabel) {
            graph_flags[gdb.get_graph_id(idx)] = 1;
          }
        } // for i
      } // if
    } // operator()
  };



public:
  gspan_cuda();
  virtual ~gspan_cuda();
  void set_database(types::graph_database_t &graph_database, bool convert_to_cuda = true);
  void set_database(types::graph_database_cuda &cuda_graph_database);
  void delete_database_from_device() {
    d_cuda_graph_database.delete_from_device();
  }
  void set_min_support(int minsup);
  void set_graph_output(graph_output * gout);
  void set_exec_configurator(cudapp::cuda_configurator *cc);
  virtual void run();
};

} // namespace gspan_cuda


#endif
