#include <gspan_cuda.hpp>
#include <cuda_gspan_ops.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_tools.hpp>
#include <kernel_execution.hpp>
#include <algorithm>
#include <cassert>
#include <graph_types.hpp>
#include <dfs_code.hpp>

#include <set>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


using types::graph_database_t;
using types::embedding_element;
using types::embedding_list_columns;

using types::Graph;

using types::Projected_map3;
using types::Projected_map2;
using types::Projected_map1;
using types::Projected_iterator3;
using types::Projected_iterator2;
using types::Projected_iterator1;

using types::Projected_riterator3;

namespace gspan_cuda {

gspan_cuda::gspan_cuda() : h_cuda_graph_database(true), d_cuda_graph_database(false)
{
  output = 0;
  minimal_support = -1;
  logger = Logger::get_logger("GSPAN_CUDA");
  execute_tests = false;
  exec_config = 0;
  cuda_allocs_get_all_ext.init();

  d_graph_flags = 0;

  reduction = new cuda_segmented_reduction();
  scanner = new cuda_segmented_scan();
  copier = new cuda_copy<types::DFS, is_one_array>();
}


gspan_cuda::~gspan_cuda()
{
  d_cuda_graph_database.delete_from_device();
  h_cuda_graph_database.delete_from_host();
  cuda_allocs_get_all_ext.device_free();

  CUDAFREE(d_graph_flags, *logger);
  d_graph_flags = 0;

  delete reduction;
  delete scanner;
  delete copier;
}


void gspan_cuda::set_exec_configurator(cudapp::cuda_configurator *cc)
{
  exec_config = cc;
}


void gspan_cuda::set_database(graph_database_t &graph_database, bool convert_to_cuda)
{
  this->graph_database = graph_database;

  if(convert_to_cuda == true) {
    DEBUG(*logger, "converting host database to cuda database");
    h_cuda_graph_database = types::graph_database_cuda::create_from_host_representation(graph_database);
    h_cuda_graph_database.copy_to_device(&d_cuda_graph_database);
    TRACE5(*logger, "cuda database: " << h_cuda_graph_database.to_string());
  }

  if(execute_tests) {
    bool test_ko = h_cuda_graph_database.test_database();
    if(test_ko == true) throw std::runtime_error("error while translating database to device representation.");
  }
}


void gspan_cuda::set_database(types::graph_database_cuda &cuda_graph_database)
{
  if(cuda_graph_database.is_on_host()) {
    h_cuda_graph_database = cuda_graph_database;
    h_cuda_graph_database.copy_to_device(&d_cuda_graph_database);
  } else {
    d_cuda_graph_database = cuda_graph_database;
    d_cuda_graph_database.copy_from_device(&h_cuda_graph_database);
  }

  if(execute_tests) {
    h_cuda_graph_database.convert_to_host_representation(graph_database);
  }
}


void gspan_cuda::set_min_support(int minsup)
{
  minimal_support = minsup;
}


void gspan_cuda::set_graph_output(graph_output * gout)
{
  output = gout;
}


void gspan_cuda::run()
{
  DEBUG(*logger, "gspan_cuda::run");

  if(execute_tests) {
    DEBUG(*logger, "testing database");
    test_database();
  }
  if(exec_config == 0) throw std::runtime_error("gspan_cuda does not have its execution configurator.");
  DEBUG(*logger, "calling gspan_cuda::run_intern2");
  run_intern2();
}




void gspan_cuda::report_single(Graph &g, std::map<unsigned int, unsigned int>& ncount)
{
  unsigned int sup = 0;
  for(std::map<unsigned int, unsigned int>::iterator it = ncount.begin(); it != ncount.end(); ++it) {
    sup += (*it).second;
  }
}


void gspan_cuda::fill_host_projected(types::Projected_map3 *root)
{
  types::EdgeList edges;
  //Projected_map3 root;


  for(unsigned int id = 0; id < graph_database.size(); ++id) {
    Graph &g = graph_database[id];
    for(unsigned int from = 0; from < g.size(); ++from) {
      if(get_forward_root(g, g[from], edges)) {
        for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
          (*root)[g[from].label][(*it)->elabel][g[(*it)->to].label].push(id, *it, 0);
      } // if
    } // for from
  } // for id
}


void gspan_cuda::run_intern2()
{
  DEBUG(*logger, "gspan_cuda::run_intern2");
  types::edge_gid_list3_t root;
  prepare_run(root);
  Projected_map3 host_projected_root;
  if(execute_tests) {
    fill_host_projected(&host_projected_root);
  }

  for(types::edge_gid_list3_t::iterator fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
    for(types::edge_gid_list2_t::iterator elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
      for(types::edge_gid_list1_t::iterator tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
        unsigned int supp = 0; //tolabel->second.size();
        int last_id = -1;
        for(int i = 0; i < tolabel->second.size(); i++) {
          if(tolabel->second[i] != last_id) {
            last_id = tolabel->second[i];
            supp++;
          }
        }
        if(supp < minimal_support) continue;

        // Build the initial two-node graph.  It will be grownrecursively within project.
        types::DFSCode code;
        code.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
        DEBUG(*logger, "*************** creating first embeddings for: " << code.to_string());
        output->output_graph(code, supp);

        types::embedding_list_columns first_embeddings(false);
        create_first_embeddings(code[0], &d_cuda_graph_database, &first_embeddings, exec_config);
        types::Projected *projected = 0;
        types::Projected projected_empty;

        if(execute_tests) {
          embedding_list_columns h_first_embeddings(true);
          h_first_embeddings.copy_from_device(&first_embeddings);
          check_host_db_agains_embedding(graph_database, code[0], h_first_embeddings);
          check_embedding_against_host_db(graph_database, code[0], h_first_embeddings);
          DEBUG(*logger, "first_embeddings.size(): " << h_first_embeddings.columns_lengths[1]);
          projected = &(host_projected_root[fromlabel->first][elabel->first][tolabel->first]);
        } else {
          projected = &projected_empty;
        }
        TRACE(*logger, "original projection size: " << tolabel->second.size());
        DEBUG(*logger, memory_checker::print_memory_usage());
        dfs_extension_element_set_t new_backward_edges;
        types::RMPath scheduled_rmpath;
        scheduled_rmpath.push_back(0);
        scheduled_rmpath.push_back(1);
        mainloop(first_embeddings, code, *projected, supp, new_backward_edges, scheduled_rmpath);
        first_embeddings.delete_from_device();
      } // for tolabel
    } // for elabel
  } // for fromlabel
} // gspan_cuda::run_intern2


types::RMPath gspan_cuda::convert_rmpath(const types::RMPath &gspan_rmpath, types::DFSCode &code)
{
  std::set<int> columns;
  for(int i = 0; i < gspan_rmpath.size(); i++) {
    columns.insert(code[gspan_rmpath[i]].from);
    columns.insert(code[gspan_rmpath[i]].to);
  } // for i

  types::RMPath cuda_rmpath;
  for(std::set<int>::iterator it = columns.begin(); it != columns.end(); it++) {
    cuda_rmpath.push_back(*it);
  } // for it

  return cuda_rmpath;
} // DFSCode::build_cuda_rmpath



void gspan_cuda::fill_root(types::edge_gid_list3_t &root)
{
  int max_vertex_count = h_cuda_graph_database.db_size * h_cuda_graph_database.max_graph_vertex_count;

  for(int vid = 0; vid < max_vertex_count; vid++) {
    if(!h_cuda_graph_database.vertex_is_valid(vid)) continue;
    int vid_label = h_cuda_graph_database.get_vertex_label(vid);
    int vid_degree = h_cuda_graph_database.get_vertex_degree(vid);
    int vid_neigh_offset = h_cuda_graph_database.get_neigh_offsets(vid);
    for(int eid = 0; eid < vid_degree; eid++) {
      int elabel = h_cuda_graph_database.edges_labels[vid_neigh_offset + eid];
      int to_vid = h_cuda_graph_database.edges[vid_neigh_offset + eid];
      int to_label = h_cuda_graph_database.get_vertex_label(to_vid);
      int gid = h_cuda_graph_database.get_graph_id(vid);
      if(vid_label <= to_label)
        root[vid_label][elabel][to_label].push_back(gid);
    } // for eid_
  } // for vid
}


//
// TODO: the notion of rmpath is invalid. The reason is that the
// rmpath in the kernel are the columns (i.e., embeddings vertices)
// not the edges.
//
void gspan_cuda::mainloop(types::embedding_list_columns &embeddings,
                          types::DFSCode &code,
                          types::Projected &projected,
                          int support,
                          dfs_extension_element_set_t backward_edges,
                          types::RMPath scheduled_rmpath)
{
  //if(code.size() >= 6) return;
  DEBUG(*logger, "=====================================================================================================");
  DEBUG(*logger, "=====================================================================================================");
  DEBUG(*logger, "===============  mainloop for: " << code.to_string() << "; support: " << support);

  const types::RMPath &rmpath = code.buildRMPath();
  types::RMPath cuda_rmpath = convert_rmpath(rmpath, code);
  // TODO: the rmpath filtering feature was switched off.
  types::RMPath scheduled_rmpath_local = cuda_rmpath;
  //filter_rmpath(cuda_rmpath, scheduled_rmpath_local, scheduled_rmpath);

  int minlabel = code[0].fromlabel;

  if(execute_tests) {
    types::embedding_list_columns h_embeddings(true);
    h_embeddings.copy_from_device(&embeddings);
    TRACE(*logger, "embeddings: " << h_embeddings.to_string());
    DEBUG(*logger, "cuda embeddings size: " << h_embeddings.columns_lengths[h_embeddings.columns_count - 1]);
    DEBUG(*logger, "cuda embeddings: " << endl << h_embeddings.to_string_with_labels(h_cuda_graph_database, code));
    DEBUG(*logger, "========================================");
    DEBUG(*logger, "host embeddings size: " << projected.size());
    DEBUG(*logger, "host embeddings: ");
    for(int i = 0; i < projected.size(); i++) {
      DEBUG(*logger, i << ", gid " << projected[i].id << ": " << projected[i].to_string_projection(graph_database, h_cuda_graph_database));
    }

    if(h_embeddings.columns_lengths[h_embeddings.columns_count - 1] != projected.size()) {
      CRITICAL_ERROR(*logger, "scheduled_rmpath_local: " << utils::print_vector(scheduled_rmpath_local) << "; rmpath: " << utils::print_vector(cuda_rmpath));
      CRITICAL_ERROR(*logger, "conditin failed: " << h_embeddings.columns_lengths[h_embeddings.columns_count - 1] << " == " <<  projected.size());
      throw std::runtime_error("error XXX");
    }
    assert(h_embeddings.columns_lengths[h_embeddings.columns_count - 1] == projected.size());
    h_embeddings.delete_from_host();
  } // if(execute_tests)

#ifdef LOG_TRACE
  TRACE(*logger, "cuda rmpath: " << utils::print_vector(cuda_rmpath)
        << "; rmpath: " << utils::print_vector(rmpath)
        << "; scheduled rmpath: " << utils::print_vector(scheduled_rmpath_local));
  TRACE(*logger, "embeddings columns: " << embeddings.columns_count);
#endif

  extension_element_t *d_exts_result = 0;
  int exts_result_length = 0;
  TRACE(*logger, "computing extensions");
  //get_all_extensions(&d_cuda_graph_database, &embeddings, cuda_rmpath, d_exts_result, exts_result_length, minlabel, exec_config);
  TRACE4(*logger, "scheduled_rmpath_local: " << utils::print_vector(scheduled_rmpath_local));

  get_all_extensions(&d_cuda_graph_database,
                     &cuda_allocs_get_all_ext,
                     &embeddings,
                     cuda_rmpath,
                     rmpath,
                     code,
                     d_exts_result,
                     exts_result_length,
                     minlabel,
                     exec_config,
                     //cuda_rmpath);
                     scheduled_rmpath_local);


  if(exts_result_length == 0) {
    CUDAFREE(d_exts_result, *logger);
    return;
  }


  if(execute_tests) {
    TRACE(*logger, "testing embeddings");
    test_embeddings(embeddings, cuda_rmpath, d_exts_result, exts_result_length);
  }

  TRACE(*logger, "computing supports for extensions");
  types::DFS *h_dfs_elem = 0;
  int *h_dfs_offsets = 0;
  int *h_support = 0;
  int dfs_array_length = 0;
  get_support_for_extensions(d_cuda_graph_database.max_graph_vertex_count,
                             d_exts_result,
                             exts_result_length,
                             dfs_array_length,
                             h_dfs_elem,
                             h_dfs_offsets,
                             h_support,
                             exec_config);

  scheduled_rmpath_local.clear();
  filter_rmpath(cuda_rmpath, scheduled_rmpath_local, h_dfs_elem, h_support, dfs_array_length);

#ifdef LOG_TRACE
  TRACE(*logger, "device extensions: (" << dfs_array_length << ")");
  for(int i = 0; i < dfs_array_length; i++) {
    int count = 0;
    if(i == dfs_array_length - 1) {
      count = exts_result_length - h_dfs_offsets[i];
    } else count = h_dfs_offsets[i + 1] - h_dfs_offsets[i];
    TRACE(*logger, h_dfs_elem[i].to_string() << "; support: " << h_support[i] << "; embedding count: " << count << "; offset: " << h_dfs_offsets[i]);
  }   // for i
  TRACE(*logger, "existing backward edges, count: " << backward_edges.size());
  for(dfs_extension_element_set_t::iterator it = backward_edges.begin(); it != backward_edges.end(); it++) {
    TRACE(*logger, "backward edge: " << it->to_string());
  }
#endif

  if(execute_tests) {
    extension_element_t *h_exts_result = new extension_element_t[exts_result_length];
    CUDA_EXEC(cudaMemcpy(h_exts_result, d_exts_result, sizeof(extension_element_t) * exts_result_length, cudaMemcpyDeviceToHost), *logger);
    for(int i = 0; i < exts_result_length; i++) {
      DEBUG(*logger, "i: " << i  << "; gid: " << h_cuda_graph_database.get_graph_id(h_exts_result[i].from_grph) << "; ext: " << h_exts_result[i].to_string());
    }

    TRACE(*logger, "testing embeddings");
    test_embeddings(embeddings, cuda_rmpath, d_exts_result, exts_result_length);
    test_supports(code, h_dfs_elem, dfs_array_length, d_exts_result, exts_result_length);
    DEBUG(*logger, "running extension check against the projections made using the taku kudo procedure");
    DEBUG(*logger, "h_dfs_elem array length: " << dfs_array_length);
    bool error_found = test_supports2(projected, code, h_dfs_elem, h_support, dfs_array_length, d_exts_result, exts_result_length);
    if(error_found) {
      types::embedding_list_columns h_embeddings(true);
      h_embeddings.copy_from_device(&embeddings);
      DEBUG(*logger, "cuda embeddings, size " << h_embeddings.get_embedding_count()  << ": " << endl << h_embeddings.to_string_with_labels(h_cuda_graph_database, code));
      DEBUG(*logger, "========================================");
      DEBUG(*logger, "host embeddings, size  " << projected.size() << ": ");
      for(int i = 0; i < projected.size(); i++) {
        DEBUG(*logger, "gid: " << projected[i].id << "; projected: " << projected[i].to_string_projection(graph_database, h_cuda_graph_database));
      }
    } else {
      DEBUG(*logger, "no error found");
    } // if
  } // if(execute_tests)



  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // looping over extensions
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  TRACE(*logger, "running the gspan for all frequent and minimal extensions");
  for(int i = 0; i < dfs_array_length; i++) {
    // check whether the new extension is frequent
    if(h_support[i] < minimal_support) {
      TRACE(*logger, "NOT FREQUENT extension: " << h_dfs_elem[i].to_string() << "; support: " << h_support[i]);
      continue;
    }


    if(backward_edges.find(h_dfs_elem[i]) != backward_edges.end()) {
      DEBUG(*logger, "skipping already found backward edge");
      continue;
    }
    // check whether the new extension is minimal
    types::DFSCode new_code = code;
    new_code.push_back(h_dfs_elem[i]);
    if(new_code.dfs_code_is_min() == false) {
      TRACE(*logger, "NOT MINIMAL extension: " << h_dfs_elem[i].to_string() << "; support: " << h_support[i]);
      continue;
    }

    output->output_graph(new_code, h_support[i]);

    TRACE(*logger, "PROCESSING extension #" << i << "; h_dfs_elem[i]: " << h_dfs_elem[i].to_string());


    // do "half copy" of the embeddings, i.e., move only the pointers into the
    // d_new_embeddings without doing actual copying of the content. This saves
    // memory. HOWEVER: special care must be taken when deallocating. Instead of
    // calling delete_from_device the d_half_deallocate must be called
    TRACE(*logger, "half-copying embeddings");
    embedding_list_columns d_new_embeddings(false);
    int new_embeddings_length;
    embedding_element *d_new_embed_vertices = 0;
    dfs_extension_element_set_t new_backward_edges(backward_edges);

    d_new_embeddings = embeddings.d_get_half_copy();
    if(h_dfs_elem[i].is_forward()) {
      //compute the number of the embeddings
      if(i == dfs_array_length - 1) {
        new_embeddings_length = exts_result_length - h_dfs_offsets[i];
      } else {
        new_embeddings_length = h_dfs_offsets[i + 1] - h_dfs_offsets[i];
      }

      // copy the embeddings from the extension_element_t on device into the
      // embedding_element, creating new column in the embedding
      CUDAMALLOC(&d_new_embed_vertices, sizeof(embedding_element) * new_embeddings_length, *logger);
      cudapp::cuda_computation_parameters copy_params = cudapp::cuda_configurator::get_computation_parameters(new_embeddings_length, 512);
      TRACE(*logger, "copying extensions, from: " << 0 << "; to: " << new_embeddings_length);
      cudapp::for_each(0, new_embeddings_length, copy_embedding_info(d_new_embed_vertices, d_exts_result, h_dfs_offsets[i]), copy_params);


      // extend the current embedding
      TRACE(*logger, "adding new column to d_new_embeddings");
      d_new_embeddings.d_extend_by_one_column(h_dfs_elem[i], d_new_embed_vertices, new_embeddings_length);
    } else {
      TRACE(*logger, "processing backward edge, filtering embeddings, embedding count: " << new_embeddings_length);
      filter_backward_embeddings(embeddings,
                                 i,
                                 d_exts_result,
                                 exts_result_length,
                                 h_dfs_offsets,
                                 h_dfs_elem,
                                 dfs_array_length,
                                 d_new_embed_vertices,
                                 new_embeddings_length,
                                 scanner,
                                 exec_config);

      d_new_embeddings.d_replace_last_column(d_new_embed_vertices, new_embeddings_length);
      new_backward_edges.insert(h_dfs_elem[i]);
      backward_edges.insert(h_dfs_elem[i]);
    }

    types::Projected new_projected;
    if(execute_tests) {
      DEBUG(*logger, "executing tests");
      DEBUG(*logger, "getting new host projection for: " << h_dfs_elem[i].to_string());
      get_new_projected(projected, code, h_dfs_elem[i], new_projected);
      unsigned int test_supp = this->support(new_projected);
      if(test_supp != h_support[i]) {
        CRITICAL_ERROR(*logger, "****************************************************************************************");
        CRITICAL_ERROR(*logger, "supports do not match ! host support: " << test_supp << "; cuda support: " << h_support[i]);
        CRITICAL_ERROR(*logger, "****************************************************************************************");
        //assert(h_support[i] == test_supp);
      }
      TRACE(*logger, "new projected.size(): " << new_projected.size());
    } // if(execute_tests)

    // execute the mainloop recursivelly
    mainloop(d_new_embeddings, new_code, new_projected, h_support[i], new_backward_edges, scheduled_rmpath_local);
    //scheduled_rmpath_local);
    d_new_embeddings.d_half_deallocate();


    CUDAFREE(d_new_embed_vertices, *logger);
  } // for i

  CUDAFREE(d_exts_result, *logger);
  DEBUG(*logger, "=============== exiting mainloop");
} // gspan_cuda::mainloop


unsigned int gspan_cuda::support(types::Projected &projected)
{
  unsigned int oid = 0xffffffff;
  unsigned int size = 0;

  for(types::Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
    if(oid != cur->id) {
      ++size;
    }
    oid = cur->id;
  }

  return size;
}







void gspan_cuda::prepare_run(types::edge_gid_list3_t &root)
{
  CUDAMALLOC(&d_graph_flags, sizeof(int) * h_cuda_graph_database.db_size, *logger);
  d_graph_flags_length = h_cuda_graph_database.db_size;

  //fill_labels();
  //fill_root_cuda(root);
  fill_root(root);
}


void gspan_cuda::fill_labels_cuda()
{
  // TODO: these reductions should be replaced. However, it is not
  // that easy to replace them using the code we already have. The
  // reason is that our reduction uses + operator, not max
  // operator. Therefore, we have copy-paste the global scan code and
  // modify it. Or we have to create a template that takes the
  // operator as a parameter.

  // The question is, whether it makes sense to do: we call the thrust
  // reduction at the beginning and only ONCE ! That is: all the slow
  // memory allocations are done only once.
  max_elabel = thrust::reduce(thrust::device_ptr<int>(d_cuda_graph_database.edges_labels),
                              thrust::device_ptr<int>(d_cuda_graph_database.edges_labels + d_cuda_graph_database.edges_sizes),
                              0,
                              thrust::maximum<int>());


  int vertex_count = d_cuda_graph_database.db_size * d_cuda_graph_database.max_graph_vertex_count;
  max_vlabel = thrust::reduce(thrust::device_ptr<int>(d_cuda_graph_database.vertex_labels),
                              thrust::device_ptr<int>(d_cuda_graph_database.vertex_labels + vertex_count),
                              0,
                              thrust::maximum<int>());
  DEBUG(*logger, "max_vlabel: " << max_vlabel << "; max_elabel: " << max_elabel << "; vertex_count: " << vertex_count);
  DEBUG(*logger, "max_graph_vertex_count: " << d_cuda_graph_database.max_graph_vertex_count << "; db size: " << d_cuda_graph_database.db_size);

  int *d_edge_labels = 0;
  int *d_vertex_labels = 0;
  CUDAMALLOC(&d_edge_labels, sizeof(int) * (max_elabel + 1), *logger);
  CUDA_EXEC(cudaMemset(d_edge_labels, 0, sizeof(int) * (max_elabel + 1)), *logger);

  CUDAMALLOC(&d_vertex_labels, sizeof(int) * (max_vlabel + 1), *logger);
  CUDA_EXEC(cudaMemset(d_vertex_labels, 0, sizeof(int) * (max_vlabel + 1)), *logger);

  int *h_edge_labels = new int[max_elabel + 1];
  int *h_vertex_labels = new int[max_vlabel + 1];

  store_edge_label store_elabel(d_edge_labels, d_cuda_graph_database.edges_labels);
  cudapp::cuda_computation_parameters store_elabel_config =
    cudapp::cuda_configurator::get_computation_parameters(h_cuda_graph_database.edges_sizes, 128);
  cudapp::for_each(0, h_cuda_graph_database.edges_sizes, store_elabel, store_elabel_config);

  store_edge_label store_vlabel(d_vertex_labels, d_cuda_graph_database.vertex_labels);
  cudapp::cuda_computation_parameters store_vlabel_config =
    cudapp::cuda_configurator::get_computation_parameters(h_cuda_graph_database.max_graph_vertex_count * h_cuda_graph_database.db_size, 128);
  cudapp::for_each(0, h_cuda_graph_database.max_graph_vertex_count * h_cuda_graph_database.db_size, store_vlabel, store_vlabel_config);

  CUDA_EXEC(cudaMemcpy(h_edge_labels, d_edge_labels, sizeof(int) * (max_elabel + 1), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(h_vertex_labels, d_vertex_labels, sizeof(int) * (max_vlabel + 1), cudaMemcpyDeviceToHost), *logger);


  for(int i = 0; i < max_elabel + 1; i++) {
    if(h_edge_labels[i] == 1) {
      edge_label_set.insert(i);
    } // if
  } // for i

  for(int i = 0; i < max_vlabel + 1; i++) {
    if(h_vertex_labels[i] == 1) {
      vertex_label_set.insert(i);
    } // if
  } // for i

  CUDAFREE(d_edge_labels, *logger);
  CUDAFREE(d_vertex_labels, *logger);
  delete [] h_edge_labels;
  delete [] h_vertex_labels;
}


void gspan_cuda::fill_root_cuda(types::edge_gid_list3_t &root)
{
  int *all_edges_flags = 0;
  int *all_edges_flags_scanned = 0;
  int all_edges_flags_size = (max_vlabel + 1) * (max_elabel + 1) * (max_vlabel + 1);
  DEBUG(*logger, "max_vlabel: " << max_vlabel << "; max_elabel: " << max_elabel);
  DEBUG(*logger, "all_edges_flags_size: " << all_edges_flags_size);
  CUDAMALLOC(&all_edges_flags, all_edges_flags_size * sizeof(int), *logger);
  CUDAMALLOC(&all_edges_flags_scanned, all_edges_flags_size * sizeof(int), *logger);
  DEBUG(*logger, "all_edges_flags: " << all_edges_flags);
  CUDA_EXEC(cudaMemset(all_edges_flags, 0, sizeof(int) * all_edges_flags_size), *logger);
  CUDA_EXEC(cudaMemset(all_edges_flags_scanned, 0, sizeof(int) * all_edges_flags_size), *logger); // TODO: is this necessary ?

  store_edge_flag sef(d_cuda_graph_database, all_edges_flags, max_vlabel, max_elabel);
  cudapp::cuda_computation_parameters sef_config =
    cudapp::cuda_configurator::get_computation_parameters(h_cuda_graph_database.max_graph_vertex_count * h_cuda_graph_database.db_size, 128);
  cudapp::for_each(0, h_cuda_graph_database.max_graph_vertex_count * h_cuda_graph_database.db_size, sef, sef_config);

  //int size = (max_vlabel + 1) * (max_elabel + 1) * (max_vlabel + 1);
  //INFO(*logger, "all_edges_flags: " << all_edges_flags << "; all_edges_flags_scanned: " << all_edges_flags_scanned << "; all_edges_flags_size: " << all_edges_flags_size);
  //scanner->global_scan((uint*)all_edges_flags, (uint*)all_edges_flags_scanned, all_edges_flags_size, EXCLUSIVE);
  scanner->scan((uint*)all_edges_flags, (uint*)all_edges_flags_scanned, all_edges_flags_size, EXCLUSIVE);

  TRACE5(*logger, "all_edges_flags_scanned: " << print_d_array(all_edges_flags_scanned, all_edges_flags_size));
  int total_extensions = 0;
  int last_flag = 0;
  CUDA_EXEC(cudaMemcpy(&total_extensions, all_edges_flags_scanned + all_edges_flags_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(&last_flag, all_edges_flags + all_edges_flags_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  total_extensions += last_flag;
  DEBUG(*logger, "total_extensions: " << total_extensions);


  extension_element_t *d_extensions = 0;
  CUDAMALLOC(&d_extensions, sizeof(extension_element_t) * total_extensions, *logger);
  store_extension se(max_vlabel, max_elabel, all_edges_flags, all_edges_flags_scanned, d_extensions);
  cudapp::cuda_computation_parameters se_config =
    cudapp::cuda_configurator::get_computation_parameters(all_edges_flags_size, 128);
  cudapp::for_each(0, all_edges_flags_size, se, se_config);


  extension_element_t *h_extensions = new extension_element_t[total_extensions];
  CUDA_EXEC(cudaMemcpy(h_extensions, d_extensions, sizeof(extension_element_t) * total_extensions, cudaMemcpyDeviceToHost), *logger);

  for(int i = 0; i < total_extensions; i++) {
    int from_label = h_extensions[i].fromlabel;
    int elabel = h_extensions[i].elabel;
    int to_label = h_extensions[i].tolabel;
    if(from_label <= to_label) {
      root[from_label][elabel][to_label] = types::graph_id_list_t();
      DEBUG(*logger, "edge: (" << from_label << ", " << elabel << ", " << to_label << ")");
    }
  } // for i
  delete [] h_extensions;

  CUDAFREE(all_edges_flags, *logger);
  CUDAFREE(all_edges_flags_scanned, *logger);
  CUDAFREE(d_extensions, *logger);
}



int gspan_cuda::compute_support(int from_label, int elabel, int to_label)
{
  int vertex_count = h_cuda_graph_database.max_graph_vertex_count * h_cuda_graph_database.db_size;
  //d_graph_flags;
  CUDA_EXEC(cudaMemset(d_graph_flags, 0, sizeof(int) * d_graph_flags_length), *logger);

  TRACE3(*logger, "d_graph_flags_length: " << d_graph_flags_length << "; " << (vertex_count));

  extension_element_t edge_labels;
  edge_labels.fromlabel = from_label;
  edge_labels.elabel = elabel;
  edge_labels.tolabel = to_label;

  store_graph_id store_gid(d_graph_flags, edge_labels, d_cuda_graph_database);

  cudapp::cuda_computation_parameters store_gid_config =
    cudapp::cuda_configurator::get_computation_parameters(vertex_count, 128);

  cudapp::for_each(0, vertex_count, store_gid, store_gid_config);
  //INFO(*logger, "d_graph_flags: " << print_d_array(d_graph_flags, d_graph_flags_length));
  std::vector<uint> segment_count;
  std::vector<uint> segment_sizes;
  std::vector<uint> results;
  segment_count.push_back(1);
  segment_sizes.push_back(d_graph_flags_length);
  reduction->reduce_inclusive((uint*)d_graph_flags,
                              d_graph_flags_length,
                              segment_count,
                              segment_sizes,
                              results);

  TRACE3(*logger, "results.size(): " << results.size() << "; results.back(): " << results.back());

  return results.back();
}


} // namespace gspan_cuda;


