#include <embedding_lists.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_computation_parameters.hpp>
#include <kernel_execution.hpp>
#include <cuda_gspan_ops.hpp>

#include <cuda_tools.hpp>
#include <cuda.h>

#include <dfs_code.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/device_ptr.h>

#include <host_operations.hpp>

using namespace types;

// bit 1: backward edge; bit 0: forward edge
#define FORWARD_EDGE_BIT    1
#define BACKWARD_EDGE_BIT   2
#define AT_LEAST_ONE    3
#define IMPOSSIBLE_FORWARD_EDGE_BIT  4
#define IMPOSSIBLE_BACKWARD_EDGE_BIT 8

namespace gspan_cuda {

static Logger *logger = Logger::get_logger("GET_EXTS");


struct compute_extensions_base {
  graph_database_cuda gdb;
  embedding_list_columns emb_lists;
  int *rmpath;
  int rmpath_length;

  int *scheduled_columns;
  int scheduled_columns_length;


  compute_extensions_base(graph_database_cuda gdb,
                          embedding_list_columns emb_lists,
                          int *rmpath,
                          int rmpath_length,
                          int *scheduled_columns,
                          int scheduled_columns_length)
    : gdb(gdb), emb_lists(emb_lists) {
    this->emb_lists = emb_lists;
    this->rmpath = rmpath;
    this->rmpath_length = rmpath_length;
    this->scheduled_columns = scheduled_columns;
    this->scheduled_columns_length = scheduled_columns_length;
  }

  __device__ __host__
  int get_column(int thread_idx) {
    int vertices_to_extend = emb_lists.columns_lengths[emb_lists.columns_count - 1];
    int col_idx = thread_idx / vertices_to_extend;
    return scheduled_columns[col_idx];
  } // get_column

  __device__ __host__
  int get_row(int thread_idx) {
    int vertices_to_extend = emb_lists.columns_lengths[emb_lists.columns_count - 1];
    return thread_idx % vertices_to_extend;
  } // get_row

};


struct get_vertex_degree : public compute_extensions_base {
  get_vertex_degree(graph_database_cuda gdb,
                    embedding_list_columns emb_lists,
                    int *rmpath,
                    int rmpath_length,
                    int *scheduled_columns,
                    int scheduled_columns_length)
    : compute_extensions_base(gdb, emb_lists, rmpath, rmpath_length, scheduled_columns, scheduled_columns_length) {
  }

  __device__ __host__
  int operator()(int thread_idx) {
    int row = get_row(thread_idx);
    int col = get_column(thread_idx);

    for(int i = emb_lists.columns_count - 1; i > col; i--) {
      row = emb_lists.columns[i][row].back_link;
    } // for i
    int vertex_id = emb_lists.columns[col][row].vertex_id;
    int vertex_degree = gdb.get_vertex_degree(vertex_id);

    return vertex_degree;
  }
};



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// operators used for storing valid edges and finding out the validity of an edge
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// the index of the embedding, i.e., the row in the last column must
// be taken into count !!!!!
template<class __operator__>
struct execute_extension_operator : public compute_extensions_base {
  int *valid_vertex_indices;
  int valid_vertex_indices_length;
  int max_vertex_degree;
  //int minlabel;
  __operator__ perform_op;

  execute_extension_operator(const execute_extension_operator &eeo) : compute_extensions_base(eeo), perform_op(eeo.perform_op) {
    valid_vertex_indices = eeo.valid_vertex_indices;
    valid_vertex_indices_length = eeo.valid_vertex_indices_length;
    max_vertex_degree = eeo.max_vertex_degree;
  }

  execute_extension_operator(graph_database_cuda gdb, embedding_list_columns emb_lists,
                             int *rmpath, int rmpath_length,
                             int *vvd, int vvd_length, int mvd,
                             __operator__ perform_op,
                             int *scheduled_columns,
                             int scheduled_columns_length)
    : compute_extensions_base(gdb, emb_lists, rmpath, rmpath_length, scheduled_columns, scheduled_columns_length), perform_op(perform_op)
  {
    valid_vertex_indices = vvd;
    valid_vertex_indices_length = vvd_length;
    max_vertex_degree = mvd;
  }


  // WE HAVE TO BE VERY CAREFULL ABOUT THE ALLOCATION OF THREADS
  // => EACH WARP SHOULD PROCESS NEIGHBORHOOD VERTICES

  // ADDITIONALLY THE NEIGHBRHOOD VERTICES SHOULD BE VERTICES THAT
  // BELONGS TO SIMILAR EMBEDDINGS. THIS PRECONDITION WILL BE PROBABLY
  // VALID IF WE STORE THE EXTENSIONS IN BLOCKS WHERE EACH BLOCK ARE
  // NEIGHBORHOOD VERTICES OF ONE VERTEX
  __device__ __host__
  int operator()(int thread_idx) {
    // follow the back links from the last column to the proper column
    // and count the number of extensions


    int start_row = get_row(thread_idx);
    int assigned_row = start_row;
    int assigned_col = get_column(thread_idx);
    for(int i = emb_lists.columns_count - 1; i > assigned_col; i--) {
      assigned_row = emb_lists.columns[i][assigned_row].back_link;
    } // for i

    int from_id_grph = emb_lists.columns[assigned_col][assigned_row].vertex_id;
    int vertex_degree = gdb.get_vertex_degree(from_id_grph);
    int neighborhood_start_offset = gdb.vertex_offsets[from_id_grph];
    int rmpath_idx = rmpath_length - 1;

    // is the order of these two for-loops good for memory access
    // pattern ?
    int curr_row = start_row;
    for(int j = emb_lists.columns_count - 1; j >= 0; j--) {
      // curr_embed_vertex_id is the vertex in the embedding processed in column j
      int curr_embed_vertex_id = emb_lists.columns[j][curr_row].vertex_id;
      bool rmpath_vertex = false;

      if(rmpath[rmpath_idx] == j) {
        rmpath_vertex = true;
        rmpath_idx--;
      }

      if(j == assigned_col) {
        curr_row = emb_lists.columns[j][curr_row].back_link;
        continue;
      }

      // for each vertex in the neighborhood of the vertex at position
      // [col][curr_row] count the valid neighborhood vertices.
      for(int i = 0; i < vertex_degree; i++) {
        if(perform_op.storing_validity() == false && ((valid_vertex_indices[thread_idx * max_vertex_degree + i] & AT_LEAST_ONE) == 0)) {
          continue;
        }

        // neighborhood vertex id of the vertex processed by this thread
        int to_id_grph = gdb.edges[neighborhood_start_offset + i];
        int to_label = gdb.get_vertex_label(to_id_grph);
        int edge_label = gdb.edges_labels[neighborhood_start_offset + i];

        int from_pat = assigned_col;
        int to_pat = -1;
        // processing forward edge
        if(assigned_col < emb_lists.columns_count - 1 || (assigned_col == emb_lists.columns_count - 1 && curr_embed_vertex_id != to_id_grph)) {
          to_pat = emb_lists.columns_count;
        }



        // processing backward edge
        if(assigned_col == emb_lists.columns_count - 1 && rmpath_vertex && curr_embed_vertex_id == to_id_grph) {
          if(rmpath_idx == rmpath_length - 3) {

            if(perform_op.storing_validity()) {
              valid_vertex_indices[thread_idx * max_vertex_degree + i] |= IMPOSSIBLE_BACKWARD_EDGE_BIT;
              valid_vertex_indices[thread_idx * max_vertex_degree + i] |= IMPOSSIBLE_FORWARD_EDGE_BIT;
              valid_vertex_indices[thread_idx * max_vertex_degree + i] &= ~BACKWARD_EDGE_BIT;
              valid_vertex_indices[thread_idx * max_vertex_degree + i] &= ~FORWARD_EDGE_BIT;
            }
            continue;
          } //
          to_pat = j;
        }


        if( perform_op.storing_validity() == true && !rmpath_vertex && curr_embed_vertex_id == to_id_grph) {
          valid_vertex_indices[thread_idx * max_vertex_degree + i] |= IMPOSSIBLE_BACKWARD_EDGE_BIT;
          valid_vertex_indices[thread_idx * max_vertex_degree + i] |= IMPOSSIBLE_FORWARD_EDGE_BIT;
          valid_vertex_indices[thread_idx * max_vertex_degree + i] &= ~BACKWARD_EDGE_BIT;
          valid_vertex_indices[thread_idx * max_vertex_degree + i] &= ~FORWARD_EDGE_BIT;
          continue;
        }


        perform_op(rmpath_vertex, valid_vertex_indices, thread_idx * max_vertex_degree + i, curr_embed_vertex_id,
                   from_id_grph, to_id_grph, from_pat, to_pat, edge_label, start_row);
      } // for i: 0 ... vertex_degree
      curr_row = emb_lists.columns[j][curr_row].back_link;
    } // for j: emb_lists.columns_count - 1 ... 0

    int valid_edge_count = 0;
    if(perform_op.storing_validity()) {
      for(int i = 0; i < vertex_degree; i++) {
        valid_edge_count += ((valid_vertex_indices[thread_idx * max_vertex_degree + i] & FORWARD_EDGE_BIT) != 0);
        valid_edge_count += ((valid_vertex_indices[thread_idx * max_vertex_degree + i] & BACKWARD_EDGE_BIT) != 0);
      } // for i
    } // if
    return valid_edge_count;
  } // operator ()
};

struct store_validity {
  embedding_list_columns emb_lists;
  graph_database_cuda gdb;
  int minlabel;
  int *frwrd_edge_labels;
  int *frwrd_vertex_labels;
  int col_count;

  __host__ __device__
  bool storing_validity() {
    return true;
  }

  __host__ __device__
  store_validity(graph_database_cuda gdb, embedding_list_columns emb_lists, int ml, int *frwrd_edge_labels, int *frwrd_vertex_labels) : gdb(gdb), emb_lists(emb_lists) {
    minlabel = ml;
    this->frwrd_edge_labels = frwrd_edge_labels;
    this->frwrd_vertex_labels = frwrd_vertex_labels;
  }

  __device__ __host__
  void operator()(bool rmpath_vertex, int *valid_edges_to, int ve_idx, int current_embed_vertex,
                  int from_id_grph, int to_id_grph, int from_pat, int to_pat, int edge_label, int row) {
    if(from_pat > to_pat && rmpath_vertex && (valid_edges_to[ve_idx] & IMPOSSIBLE_BACKWARD_EDGE_BIT) == 0) {
      // backward edge
      // e1: edge let say with (e1_from_gid, e1_to_gid) then the backward edge goes from the right-most vertex to the vertex e1_from_gid
      // e2: edge let say with (e2_from_gid, e2_to_gid) e2_from_gid is the right-most_vertex
      int e1_elabel = frwrd_edge_labels[to_pat];
      int e1_to_label = frwrd_vertex_labels[to_pat];
      int e2_to_label = gdb.get_vertex_label(from_id_grph);

      //int from_label = gdb.get_vertex_label(from_id_grph);
      //int to_label = gdb.get_vertex_label(to_id_grph);

      if(e1_elabel < edge_label || (e1_elabel == edge_label && e1_to_label <= e2_to_label)) {
        valid_edges_to[ve_idx] |= BACKWARD_EDGE_BIT;
        valid_edges_to[ve_idx] &= ~IMPOSSIBLE_BACKWARD_EDGE_BIT;
      }
      valid_edges_to[ve_idx] |= IMPOSSIBLE_FORWARD_EDGE_BIT;
      valid_edges_to[ve_idx] &= ~FORWARD_EDGE_BIT;
      return;
    }

    // forward edge
    if(from_pat < to_pat && rmpath_vertex && (valid_edges_to[ve_idx] & IMPOSSIBLE_FORWARD_EDGE_BIT) == 0) {
      //valid_vertex_indices[ve_idx] |= IMPOSSIBLE_BACKWARD_EDGE_BIT;
      //valid_vertex_indices[ve_idx] |= IMPOSSIBLE_FORWARD_EDGE_BIT;
      //valid_vertex_indices[ve_idx] &= ~BACKWARD_EDGE_BIT;
      //valid_vertex_indices[ve_idx] &= ~FORWARD_EDGE_BIT;

      if(to_id_grph == current_embed_vertex) {
        valid_edges_to[ve_idx] |= IMPOSSIBLE_FORWARD_EDGE_BIT;
        valid_edges_to[ve_idx] &= ~FORWARD_EDGE_BIT;
        return;
      }  // if

      int to_grph_vlabel = gdb.get_vertex_label(to_id_grph);
      if(from_pat == emb_lists.columns_count - 1 && to_grph_vlabel < minlabel) return;
      if(from_pat < emb_lists.columns_count - 1 && to_grph_vlabel < minlabel) return;
      int pat_elabel = frwrd_edge_labels[from_pat];
      int to_pat_vlabel = frwrd_vertex_labels[from_pat];

      if(from_pat == emb_lists.columns_count - 1) {
        valid_edges_to[ve_idx] |= FORWARD_EDGE_BIT;
        //valid_edges_to[ve_idx] |= IMPOSSIBLE_BACKWARD_EDGE_BIT;
        valid_edges_to[ve_idx] &= ~BACKWARD_EDGE_BIT;
        return;
      }


      if(from_pat < emb_lists.columns_count - 1 &&
         (pat_elabel < edge_label || (pat_elabel == edge_label && to_pat_vlabel <= to_grph_vlabel)) ) {
        valid_edges_to[ve_idx] |= FORWARD_EDGE_BIT;
        valid_edges_to[ve_idx] &= ~BACKWARD_EDGE_BIT;
        return;
      }

      if(current_embed_vertex == to_id_grph && !rmpath_vertex) {
        valid_edges_to[ve_idx] &= ~BACKWARD_EDGE_BIT;
        valid_edges_to[ve_idx] &= ~FORWARD_EDGE_BIT;
        valid_edges_to[ve_idx] |= IMPOSSIBLE_BACKWARD_EDGE_BIT;
        valid_edges_to[ve_idx] |= IMPOSSIBLE_FORWARD_EDGE_BIT;
      }
    } // if-else

  } // operator()
};


struct store_embedding_extension {
  extension_element_t *extensions;
  int extensions_length;

  embedding_list_columns emb_lists;
  graph_database_cuda gdb;

  int *extension_offsets;
  int minlabel;


  __host__ __device__
  bool storing_validity() {
    return false;
  }

  __host__ __device__
  store_embedding_extension(graph_database_cuda gdb, embedding_list_columns emb_lists,
                            extension_element_t *extensions, int *ext_offsets, int extensions_length, int ml)
    : gdb(gdb), emb_lists(emb_lists) {
    this->extensions = extensions;
    this->extensions_length = extensions_length;
    this->extension_offsets = ext_offsets;
    minlabel = ml;
  }


  __device__ __host__
  void operator()(bool rmpath_vertex, int *valid_edges_to, int ve_idx, int current_embed_vertex,
                  int from_id_grph, int to_id_grph, int from_pat, int to_pat, int edge_label, int row) {

    //if(valid_edges_to[ve_idx] == 0) return;
    int extension_idx = extension_offsets[ve_idx];

    // backward edge
    if(from_pat > to_pat && (valid_edges_to[ve_idx] & BACKWARD_EDGE_BIT)) {
      //valid_edges_to[ve_idx] = 0;
      extensions[extension_idx].from_grph = from_id_grph;
      extensions[extension_idx].to_grph = to_id_grph;
      extensions[extension_idx].from = from_pat;
      extensions[extension_idx].to = to_pat;
      extensions[extension_idx].fromlabel = gdb.get_vertex_label(from_id_grph);
      extensions[extension_idx].elabel = edge_label;
      extensions[extension_idx].tolabel = gdb.get_vertex_label(to_id_grph);
      extensions[extension_idx].row = row;
    }

    //forward edge
    if(from_pat < to_pat && (valid_edges_to[ve_idx] & FORWARD_EDGE_BIT)) {
      extensions[extension_idx].from_grph = from_id_grph;
      extensions[extension_idx].to_grph = to_id_grph;
      extensions[extension_idx].from = from_pat;
      extensions[extension_idx].to = to_pat;
      extensions[extension_idx].fromlabel = gdb.get_vertex_label(from_id_grph);
      extensions[extension_idx].elabel = edge_label;
      extensions[extension_idx].tolabel = gdb.get_vertex_label(to_id_grph);
      extensions[extension_idx].row = row;
    }  // if
  }
};


struct not_zero_unary_op {
  __host__ __device__
  int operator()(const int &val) const {
    return int((val & AT_LEAST_ONE) != 0 ? 1 : 0);
  }
};


void get_frwrd_edge_label_array(const types::RMPath  &cuda_rmpath,
                                types::DFSCode &code,
                                int *&frwrd_edge_labels,
                                cuda_allocs_for_get_all_extensions *allocs)
{
  types::RMPath code_rmpath = code.buildRMPath();
  delete [] frwrd_edge_labels;
  int max = *std::max_element(cuda_rmpath.begin(), cuda_rmpath.end()) + 1;
  int *h_frwrd_edge_labels = new int[max];
  memset(h_frwrd_edge_labels, 0, sizeof(int) * max);
  for(int cuda_rmpath_idx = 0; cuda_rmpath_idx < cuda_rmpath.size(); cuda_rmpath_idx++) {
    bool found = false;
    int col = cuda_rmpath[cuda_rmpath_idx];
    for(int i = 0; i < code_rmpath.size(); i++) {
      if(code[code_rmpath[i]].from == col) {
        found = true;
        h_frwrd_edge_labels[col] = code[code_rmpath[i]].elabel;
        break;
      } // if
    } // for
    if(!found) {
      h_frwrd_edge_labels[col] = 0;
    }
  }

#ifdef LOG_TRACE
  std::stringstream ss;
  for(int i = 0; i < max; i++) {
    ss << "col: " << i << "; frwrd edge label: " << h_frwrd_edge_labels[i] << "  |  ";
  }
  std::stringstream ss_rmpath;
  for(int i = 0; i < code_rmpath.size(); i++) {
    ss_rmpath << code_rmpath[i] << " ";
  }
  TRACE4(*logger, "code: " << code.to_string() << "; right-most path: " << ss_rmpath.str() << "; forward edge labels: " << ss.str());
#endif
  //CUDAMALLOC(&frwrd_edge_labels, sizeof(int) * max, *logger);
  frwrd_edge_labels = allocs->get_d_frwd_edge_labels(max);
  CUDA_EXEC(cudaMemcpy(frwrd_edge_labels, h_frwrd_edge_labels, sizeof(int) * max, cudaMemcpyHostToDevice), *logger);
  delete [] h_frwrd_edge_labels;
} // get_host_edge_label_array


void get_frwrd_edge_vertex_array(const types::RMPath  &cuda_rmpath,
                                 types::DFSCode &code,
                                 int *&frwrd_vertex_labels,
                                 cuda_allocs_for_get_all_extensions *allocs)
{
  types::RMPath code_rmpath = code.buildRMPath();
  //int cuda_rmpath_idx = 0;
  delete [] frwrd_vertex_labels;
  int max = *std::max_element(cuda_rmpath.begin(), cuda_rmpath.end()) + 1;
  int * h_frwrd_vertex_labels = new int[max];
  memset(h_frwrd_vertex_labels, 0, sizeof(int) * max);
  for(int cuda_rmpath_idx = 0; cuda_rmpath_idx < cuda_rmpath.size(); cuda_rmpath_idx++) {
    bool found = false;
    int col = cuda_rmpath[cuda_rmpath_idx];
    for(int i = 0; i < code_rmpath.size(); i++) {
      if(code[code_rmpath[i]].from == col) {
        found = true;
        h_frwrd_vertex_labels[col] = code[code_rmpath[i]].tolabel;
        break;
      } // if
    } // for
    if(!found) {
      h_frwrd_vertex_labels[col] = 0;
    }
  } // for cuda_rmpath_idx

#ifdef LOG_TRACE
  std::stringstream ss;
  for(int i = 0; i < max; i++) {
    ss << "col: " << i << "; frwrd vertex labels: " << h_frwrd_vertex_labels[i] << "  |  ";
  }
  std::stringstream ss_rmpath;
  for(int i = 0; i < code_rmpath.size(); i++) {
    ss_rmpath << code_rmpath[i] << " ";
  }
  TRACE4(*logger, "code: " << code.to_string() << "; right-most path: " << ss_rmpath.str() << "; forward vertex labels: " << ss.str());
#endif
  //CUDAMALLOC(&frwrd_vertex_labels, sizeof(int) * max, *logger);
  frwrd_vertex_labels = allocs->get_d_frwd_vertex_labels(max);
  CUDA_EXEC(cudaMemcpy(frwrd_vertex_labels, h_frwrd_vertex_labels, sizeof(int) * max, cudaMemcpyHostToDevice), *logger);
  delete [] h_frwrd_vertex_labels;
} // get_host_edge_label_array



/**
 * TODO: fix the execution configuration, it should be given as the argument.
 *       CUDAMALLOC is still done for extension_array, need to fix it. Need to do it carefully
 *       as it is assigned to exts_result and exts_result is CUDAFREEd in the caller function
 */
void get_all_extensions(graph_database_cuda *gdb,
                        cuda_allocs_for_get_all_extensions *allocs,
                        embedding_list_columns *embeddings,
                        types::RMPath cuda_rmpath,
                        types::RMPath host_rmpath,
                        types::DFSCode code,
                        extension_element_t *&exts_result,
                        int &exts_result_length,
                        int minlabel,
                        cudapp::cuda_configurator *exec_conf,
                        types::RMPath scheduled_rmpath_cols)
{
  // 1) copy from device the lengths of all columns and the right-most path
  int *column_lengths = new int[embeddings->columns_count];
  CUDA_EXEC(cudaMemcpy(column_lengths, embeddings->columns_lengths, sizeof(int) * embeddings->columns_count, cudaMemcpyDeviceToHost), *logger);

  TRACE2(*logger, "get_all_extensions, minlabel: " << minlabel);

  TRACE3(*logger, "copying rmpath to cuda");
  int *d_rmpath = 0;
  int rmpath_length = cuda_rmpath.size();
  //CUDAMALLOC(&d_rmpath, cuda_rmpath.size() * sizeof(int), *logger);
  d_rmpath = allocs->get_d_rmpath(rmpath_length);
  CUDA_EXEC(cudaMemcpy(d_rmpath, cuda_rmpath.data(), cuda_rmpath.size() * sizeof(int), cudaMemcpyHostToDevice), *logger);


  TRACE3(*logger, "copying scheduled_rmpath_cols to cuda: " << utils::print_vector(scheduled_rmpath_cols));
  int *d_scheduled_rmpath_cols = 0;
  int scheduled_rmpath_cols_length = scheduled_rmpath_cols.size();
  //CUDAMALLOC(&d_scheduled_rmpath_cols, scheduled_rmpath_cols_length * sizeof(int), *logger);
  d_scheduled_rmpath_cols = allocs->get_d_scheduled_rmpath_cols(scheduled_rmpath_cols_length);
  CUDA_EXEC(cudaMemcpy(d_scheduled_rmpath_cols, scheduled_rmpath_cols.data(), scheduled_rmpath_cols_length * sizeof(int), cudaMemcpyHostToDevice), *logger);




  TRACE3(*logger, "finding valid vertices, scheduled_rmpath_cols.size(): " << scheduled_rmpath_cols.size());
  // 2) allocate array of temporary edge-valid indices.
  int last_column_size = column_lengths[embeddings->columns_count - 1];
  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e(last_column_size * scheduled_rmpath_cols.size());
  // 2a) find maximum vertex degree by transform_reduce
  get_vertex_degree gvd(*gdb, *embeddings, d_rmpath, rmpath_length, d_scheduled_rmpath_cols, scheduled_rmpath_cols_length);
  TRACE3(*logger, "computing the size of valid_vertex_indices array (by reducing the vertex degrees); last_column_size: " << last_column_size
         << "; cuda_rmpath.size(): " << cuda_rmpath.size());
  int max_rmpath_vertex_degree = thrust::transform_reduce(b, e, gvd, 0, thrust::maximum<int>());
  int *valid_vertex_indices = 0;
  int valid_vertex_indices_length = max_rmpath_vertex_degree * last_column_size * rmpath_length;


  int *extension_offsets = 0;
  // 2b) allocate the array
  TRACE3(*logger, "allocating array valid_vertex_indices of size: " << valid_vertex_indices_length);
  //CUDAMALLOC(&valid_vertex_indices, sizeof(int) * valid_vertex_indices_length, *logger);
  //CUDAMALLOC(&extension_offsets, sizeof(int) * valid_vertex_indices_length, *logger);
  valid_vertex_indices = allocs->get_valid_vertex_indices(valid_vertex_indices_length);
  extension_offsets = allocs->get_extension_offsets(valid_vertex_indices_length);
  CUDA_EXEC(cudaMemset(valid_vertex_indices, 0, sizeof(int) * valid_vertex_indices_length), *logger);
  CUDA_EXEC(cudaMemset(extension_offsets, 0, sizeof(int) * valid_vertex_indices_length), *logger);


  // 3) find out which edges are valid
  int *d_frwrd_edge_labels = 0;
  int *d_frwrd_vertex_labels = 0;
  get_frwrd_edge_label_array(cuda_rmpath, code, d_frwrd_edge_labels, allocs);
  //print_d_array(d_frwrd_edge_labels, cuda_rmpath.back(), "d_frwrd_edge_labels");
  get_frwrd_edge_vertex_array(cuda_rmpath, code, d_frwrd_vertex_labels, allocs);
  //print_d_array(d_frwrd_vertex_labels, cuda_rmpath.back(), "d_frwrd_edge_labels");

  store_validity store_val(*gdb, *embeddings, minlabel, d_frwrd_edge_labels, d_frwrd_vertex_labels);
  execute_extension_operator<store_validity>
  gvec(*gdb,
       *embeddings,
       d_rmpath,
       rmpath_length,
       valid_vertex_indices,
       valid_vertex_indices_length,
       max_rmpath_vertex_degree,
       store_val,
       d_scheduled_rmpath_cols,
       scheduled_rmpath_cols_length);

  TRACE3(*logger, "executing execute_extension_operator<store_validity> kernel");
  TRACE3(*logger, "last_column_size: " << last_column_size << "; scheduled_rmpath_cols.size(): " << scheduled_rmpath_cols.size());
  cudapp::cuda_computation_parameters gvec_params = exec_conf->get_exec_config("compute_extensions.store_validity", last_column_size * scheduled_rmpath_cols.size());
  TRACE3(*logger, "gvec_params: " << gvec_params.to_string() << "; last_column_size: " << last_column_size);
  cudapp::for_each(0, last_column_size * scheduled_rmpath_cols.size(), gvec, gvec_params);
  //CUDAFREE(d_frwrd_vertex_labels, *logger);
  //CUDAFREE(d_frwrd_edge_labels, *logger);
  //print_d_array(valid_vertex_indices, valid_vertex_indices_length, "valid_vertex_indices");

  // 4) compute the offsets in the resulting array from the valid_vertex_indices
  TRACE3(*logger, "executing exclusive_scan on valid_edges indices to get offsets");
  TRACE3(*logger, "valid_vertex_indices: " << valid_vertex_indices);
  TRACE3(*logger, "extension_offsets: " << extension_offsets << "; length: " << valid_vertex_indices_length);
  thrust::transform_exclusive_scan(thrust::device_pointer_cast(valid_vertex_indices),
                                   thrust::device_pointer_cast(valid_vertex_indices + valid_vertex_indices_length),
                                   thrust::device_pointer_cast(extension_offsets),
                                   not_zero_unary_op(),
                                   0,
                                   thrust::plus<int>());


  int extension_array_length = 0;
  int last_valid = 0;
  CUDA_EXEC(cudaMemcpy(&extension_array_length, extension_offsets + (valid_vertex_indices_length - 1), sizeof(int), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(&last_valid, valid_vertex_indices + (valid_vertex_indices_length - 1), sizeof(int), cudaMemcpyDeviceToHost), *logger);

  // we have performed exclusive scan, so we have to add the last 0 or 1 to the extension_array_length
  extension_array_length += ((last_valid & AT_LEAST_ONE) != 0 ? 1 : 0);
  extension_element_t *extension_array = 0;
  CUDAMALLOC(&extension_array, sizeof(extension_element_t) * (extension_array_length + 1), *logger);
  //extension_array = allocs->get_extension_array(extension_array_length + 1);
  CUDA_EXEC(cudaMemset(extension_array, 0, sizeof(extension_element_t) * (extension_array_length + 1)), *logger);

  TRACE3(*logger, "extension array length: " << extension_array_length);


  // 5) store the extensions in the new array at the position computed in step 4
  store_embedding_extension store_emb_ext(*gdb, *embeddings,
                                          extension_array, extension_offsets, extension_array_length, minlabel);
  execute_extension_operator<store_embedding_extension>
  exec_extension_op(*gdb,
                    *embeddings,
                    d_rmpath,
                    rmpath_length,
                    valid_vertex_indices,
                    valid_vertex_indices_length,
                    max_rmpath_vertex_degree,
                    store_emb_ext,
                    d_scheduled_rmpath_cols,
                    scheduled_rmpath_cols_length);
  TRACE3(*logger, "executing extension operator: store_embedding_extension");
  TRACE3(*logger, "extension_array: " << extension_array << "; end: " << (extension_array + extension_array_length) << "; length: " << extension_array_length);

  cudapp::cuda_computation_parameters
    exec_extension_op_params = exec_conf->get_exec_config("compute_extensions.exec_extension_op",
                                                          last_column_size * scheduled_rmpath_cols.size());
  for_each(0, last_column_size * scheduled_rmpath_cols.size(), exec_extension_op, exec_extension_op_params);
  TRACE3(*logger, "get_all_extensions execution finished");

  // 6) free temporary buffers
  //CUDAFREE(d_rmpath, *logger);
  //CUDAFREE(valid_vertex_indices, *logger);
  //CUDAFREE(extension_offsets, *logger);
  //CUDAFREE(d_scheduled_rmpath_cols, *logger);

  delete [] column_lengths;

  // 7) store result
  exts_result = extension_array;
  exts_result_length = extension_array_length;
} // compute_supports

} // namespace gspan_cuda

