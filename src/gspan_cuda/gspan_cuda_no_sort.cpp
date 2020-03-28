#include <gspan_cuda_no_sort.hpp>
#include <embedding_lists.hpp>
#include <cuda_utils.hpp>
#include <kernel_execution.hpp>
#include <algorithm>

using types::embedding_list_columns;
using types::embedding_element;

namespace gspan_cuda {

gspan_cuda_no_sort::gspan_cuda_no_sort()
{
  execute_tests = false;

  max_elabel = 0;
  max_vlabel = 0;
}

void gspan_cuda_no_sort::run()
{
  run_intern2();
}


void gspan_cuda_no_sort::fill_labels()
{
  std::stringstream ss;
  for(int i = 0; i < h_cuda_graph_database.edges_sizes; i++) {
    if(h_cuda_graph_database.edges_labels[i] == -1) continue;
    edge_label_set.insert(h_cuda_graph_database.edges_labels[i]);
  }

  for(std::set<int>::iterator it = edge_label_set.begin(); it != edge_label_set.end(); it++) {
    ss << *it << "; ";
  }
  TRACE(*logger, "edge labels: " << ss.str());

  int vertices = h_cuda_graph_database.db_size * h_cuda_graph_database.max_graph_vertex_count;
  for(int i = 0; i < vertices; i++) {
    if(h_cuda_graph_database.vertex_labels[i] == -1) continue;
    vertex_label_set.insert(h_cuda_graph_database.vertex_labels[i]);
  } // for i


  ss.str("");
  for(std::set<int>::iterator it = vertex_label_set.begin(); it != vertex_label_set.end(); it++) {
    ss << *it << "; ";
  }
  TRACE(*logger, "vertex labels: " << ss.str());


  TRACE(*logger, "edge label count: " << edge_label_set.size());
  TRACE(*logger, "vertex label count: " << vertex_label_set.size());
}


void gspan_cuda_no_sort::prepare_run(types::edge_gid_list3_t &root)
{
  CUDAMALLOC(&d_graph_flags, sizeof(int) * h_cuda_graph_database.db_size, *logger);
  d_graph_flags_length = h_cuda_graph_database.db_size;

  fill_labels_cuda();
  fill_root_cuda(root);
}


void gspan_cuda_no_sort::run_intern2()
{
  types::edge_gid_list3_t root;
  prepare_run(root);

  types::edge_gid_list3_t test_root;
  if(execute_tests) {
    fill_root(test_root);
  }

  int num_EV_labels = edge_label_set.size() * vertex_label_set.size();
  CUDAMALLOC(&d_fwd_flags, sizeof(int) * num_EV_labels, *logger);
  CUDAMALLOC(&d_bwd_flags, sizeof(int) * num_EV_labels, *logger);
  CUDAMALLOC(&d_fwd_block_dfs_array, sizeof(types::DFS) * num_EV_labels, *logger);
  CUDAMALLOC(&d_bwd_block_dfs_array, sizeof(types::DFS) * num_EV_labels, *logger);

  //these will be allocated in the extract extensions function
  d_ext_block_offsets = 0;
  d_ext_block_offsets_length = 0;
  d_ext_dfs_array = 0;
  d_ext_dfs_array_length = 0;

  for(types::edge_gid_list3_t::iterator fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
    for(types::edge_gid_list2_t::iterator elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
      for(types::edge_gid_list1_t::iterator tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
        unsigned int supp = compute_support(fromlabel->first, elabel->first, tolabel->first);
        if(execute_tests) {
          unsigned int supp_test = compute_support2(test_root, fromlabel->first, elabel->first, tolabel->first);
          if(supp != supp_test) {
            CRITICAL_ERROR(*logger, "cuda supp: " << supp << "; host supp: " << supp_test << "; ARE NOT EQUAL");
            throw std::runtime_error("Supports do not match.");
          }
        }
        if(supp < minimal_support) continue;

        // Build the initial two-node graph.  It will be grownrecursively within mainloop.
        types::DFSCode code;
        code.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
        DEBUG(*logger, "*************** creating first embeddings for: " << code.to_string());
        output->output_graph(code, supp);

        types::embedding_list_columns first_embeddings(false);
        create_first_embeddings(code[0], &d_cuda_graph_database, &first_embeddings, exec_config);

        TRACE(*logger, "original projection size: " << tolabel->second.size());
        dfs_extension_element_set_t new_backward_edges;
        extension_set_t not_frequent_extensions;
        types::RMPath scheduled_rmpath;
        scheduled_rmpath.push_back(0);
        scheduled_rmpath.push_back(1);

        mainloop(first_embeddings, code, supp, new_backward_edges, scheduled_rmpath, not_frequent_extensions);

        first_embeddings.delete_from_device();
      } // for tolabel
    } // for elabel
  } // for fromlabel

  //CUDAFREE(d_graph_flags, *logger);
  CUDAFREE(d_fwd_flags, *logger);
  CUDAFREE(d_bwd_flags, *logger);
  CUDAFREE(d_fwd_block_dfs_array, *logger);
  CUDAFREE(d_bwd_block_dfs_array, *logger);

  CUDAFREE(d_ext_block_offsets, *logger);
  CUDAFREE(d_ext_dfs_array, *logger);

  INFO(*logger, memory_checker::print_memory_usage());
} // gspan_cuda::run_intern2


int gspan_cuda_no_sort::get_block_size(int block_id, const int *block_offsets, int num_blocks)
{
  if(block_offsets[block_id] == -1) {
    abort();
  } // if


  for(int i = block_id + 1; i <= num_blocks; i++) {
    if(block_offsets[i] != -1) {
      return block_offsets[i] - block_offsets[block_id];
    } // if
  } // for i

  abort();
} // gspan_cuda_no_sort::get_block_size

void gspan_cuda_no_sort::filter_extensions(extension_element_t *d_exts_result,
                                           int exts_result_length,
                                           types::DFS *h_dfs_elem,
                                           int dfs_array_length,
                                           int *block_offsets,
                                           int num_blocks,
                                           int col_count,
                                           types::DFSCode code,
                                           extension_set_t &not_frequent_extensions,
                                           types::DFS *&h_frequent_dfs_elem,
                                           int *&h_frequent_dfs_supports,
                                           int &frequent_candidate_count)
{
  // remove extensions that are infrequent
  h_frequent_dfs_supports = new int[dfs_array_length];
  h_frequent_dfs_elem = new types::DFS[dfs_array_length];


  for(int i = 0; i <= col_count; i++) {
    TRACE(*logger, "block_offsets[" << i << "]: " << block_offsets[i]);
  }

  int *d_graph_boundaries_scan = 0;
  int graph_boundaries_scan_length = 0;
  bool compute_graph_boundaries = true;
  int mapped_db_size = 0;
  int last_block_id = -1;

  for(int i = 0; i < dfs_array_length; i++) {
    if(not_frequent_extensions.find(h_dfs_elem[i]) == not_frequent_extensions.end()) {
      if(last_block_id != h_dfs_elem[i].from) {
        /*
           if(d_graph_boundaries_scan != 0) {
           CUDAFREE(d_graph_boundaries_scan, *logger);
           }
           d_graph_boundaries_scan = 0;
           mapped_db_size = 0; */
        compute_graph_boundaries = true;
        last_block_id = h_dfs_elem[i].from;
      }

      int bo_start = block_offsets[h_dfs_elem[i].from];
      int bo_length = get_block_size(h_dfs_elem[i].from, block_offsets, num_blocks);

      TRACE(*logger, "checking: " << h_dfs_elem[i].to_string() << "; bo_start: " << bo_start << "; bo_length: " << bo_length);

      int tmp_supp = compute_support_remapped_db(d_exts_result + bo_start,
                                                 bo_length,
                                                 h_dfs_elem[i],
                                                 h_cuda_graph_database.max_graph_vertex_count,
                                                 d_graph_flags,
                                                 d_graph_boundaries_scan,
                                                 graph_boundaries_scan_length,
                                                 compute_graph_boundaries,
                                                 mapped_db_size,
                                                 scanner);

      /*
         int tmp_supp = compute_support(d_exts_result + bo_start,
                                     bo_length,
                                     h_dfs_elem[i],
                                     h_cuda_graph_database.db_size,
                                     h_cuda_graph_database.max_graph_vertex_count);
       */
      TRACE(*logger, "tmp_supp: " << tmp_supp);
      types::DFSCode new_dfscode = code;

      //new_dfscode.push_back(h_dfs_elem[i]);
      //if(new_dfscode.dfs_code_is_min() == false) {
      //TRACE(*logger, "non-minimal: " << h_dfs_elem[i].to_string());
      //continue;
      //}

      if(tmp_supp < minimal_support) {
        types::DFS tmp_dfs = h_dfs_elem[i];
        tmp_dfs.to = col_count + 1;
        not_frequent_extensions.insert(tmp_dfs);
        TRACE(*logger, "non-frequent: " << h_dfs_elem[i].to_string());
      } else {
        h_frequent_dfs_supports[frequent_candidate_count] = tmp_supp;
        h_frequent_dfs_elem[frequent_candidate_count] = h_dfs_elem[i];
        frequent_candidate_count++;
        TRACE(*logger, "frequent: " << h_dfs_elem[i].to_string());
      }
    } else {
      types::DFS tmp_dfs = h_dfs_elem[i];
      tmp_dfs.to = col_count + 1;
      not_frequent_extensions.insert(tmp_dfs);
    } // if-else
  } // for i

  if(d_graph_boundaries_scan != 0) {
    CUDAFREE(d_graph_boundaries_scan, *logger);
  }

}


int gspan_cuda_no_sort::compute_support2(const types::edge_gid_list3_t &root, int from_label, int elabel, int to_label) const
{
  types::edge_gid_list3_t::const_iterator fromlabel_it = root.find(from_label);
  types::edge_gid_list2_t::const_iterator elabel_it = fromlabel_it->second.find(elabel);
  types::edge_gid_list1_t::const_iterator tolabel_it = elabel_it->second.find(to_label);

  const types::graph_id_list_t &grph_list = tolabel_it->second;

  int last_id = -1;
  int supp = 0;
  for(int i = 0; i < grph_list.size(); i++) {
    if(grph_list[i] != last_id) {
      last_id = grph_list[i];
      supp++;
    } // if
  } // for i

  return supp;
} // gspan_cuda_no_sort::compute_support



void gspan_cuda_no_sort::mainloop(types::embedding_list_columns &embeddings,
                                  types::DFSCode &code,
                                  int support,
                                  dfs_extension_element_set_t backward_edges,
                                  types::RMPath scheduled_rmpath,
                                  extension_set_t not_frequent_extensions)
{
  TRACE(*logger, "=====================================================================================================");
  TRACE(*logger, "=====================================================================================================");
  DEBUG(*logger, "===============  mainloop for: " << code.to_string() << "; support: " << support);

  TRACE(*logger, "scheduled_rmpath: " << utils::print_vector(scheduled_rmpath));

  const types::RMPath &rmpath = code.buildRMPath();
  types::RMPath cuda_rmpath = convert_rmpath(rmpath, code);
  types::RMPath scheduled_rmpath_local;
  filter_rmpath(cuda_rmpath, scheduled_rmpath_local, scheduled_rmpath);
  TRACE(*logger, "scheduled_rmpath_local: " << utils::print_vector(scheduled_rmpath_local));

  int minlabel = code[0].fromlabel;


  extension_element_t *d_exts_result = 0;
  int exts_result_length = 0;
  TRACE(*logger, "computing extensions");
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
                     scheduled_rmpath_local);


  if(exts_result_length == 0) {
    CUDAFREE(d_exts_result, *logger);
    return;
  }


  TRACE(*logger, "computing supports for extensions");

  int *block_offsets = 0;
  int num_blocks = embeddings.columns_count;


  // h_dfs_elem are all possible (frequent or infrequent) extensions stored in d_exts_result
  types::DFS *h_dfs_elem = 0;
  int dfs_array_length = 0;

  extract_extensions(edge_label_set.size(),
                     vertex_label_set.size(),
                     embeddings.columns_count,
                     d_exts_result,
                     exts_result_length,
                     d_ext_block_offsets,
                     d_ext_block_offsets_length,
                     d_ext_dfs_array,
                     d_ext_dfs_array_length,
                     d_fwd_flags,
                     d_bwd_flags,
                     d_fwd_block_dfs_array,
                     d_bwd_block_dfs_array,
                     block_offsets,
                     h_dfs_elem,
                     dfs_array_length,
                     copier);


  TRACE(*logger, "dfs extensions: " << dfs_array_length);
  for(int i = 0; i < dfs_array_length; i++) {
    TRACE(*logger, "dfs: " << h_dfs_elem[i].to_string());
  }

  scheduled_rmpath_local.clear();


  extension_set_t not_frequent_extensions_local;

  // h_frequent_dfs_elem stores only the FREQUENT extensions
  types::DFS *h_frequent_dfs_elem = 0;
  int *h_frequent_dfs_supports = 0;
  int frequent_candidate_count = 0;

  // filter-out infrequent DFS from h_dfs_elem
  filter_extensions(d_exts_result, exts_result_length, // the extensions
                    h_dfs_elem, dfs_array_length, block_offsets, num_blocks, embeddings.columns_count, // input DFS elements
                    code,
                    not_frequent_extensions_local, // output not frequent extensions
                    h_frequent_dfs_elem, h_frequent_dfs_supports, frequent_candidate_count); // output of the filtering, the frequent extensions

  //INFO(*logger, "h_dfs_elem: " << h_dfs_elem << "; dfs_array_length: " << dfs_array_length);
  //INFO(*logger, "h_dfs_elem: " << h_dfs_elem);

  if(execute_tests) {
    TRACE(*logger, "TEST EXTENSIONS");
    test_extensions(embeddings, code, scheduled_rmpath, h_frequent_dfs_elem, h_frequent_dfs_supports, frequent_candidate_count);
  }


  TRACE(*logger, "frequent dfs extensions: " << frequent_candidate_count);
  for(int i = 0; i < frequent_candidate_count; i++) {
    TRACE(*logger, "dfs: " << h_frequent_dfs_elem[i].to_string());
  }


  // filter-out from cuda right-most path the columns that do not have any frequent extensions
  filter_rmpath(cuda_rmpath, scheduled_rmpath_local, h_frequent_dfs_elem, h_frequent_dfs_supports, frequent_candidate_count);


  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // looping over extensions
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // the extenions are taken from h_frequent_dfs_elem
  TRACE(*logger, "running the gspan_cuda_no_sort for all frequent and minimal extensions");
  for(int i = 0; i < frequent_candidate_count; i++) {
    // check whether the new extension is frequent
    if(h_frequent_dfs_supports[i] < minimal_support) {
      TRACE(*logger, "NOT FREQUENT extension: " << h_frequent_dfs_elem[i].to_string() << "; support: " << h_frequent_dfs_supports[i]);
      continue;
    }


    if(backward_edges.find(h_frequent_dfs_elem[i]) != backward_edges.end()) {
      DEBUG(*logger, "skipping already found backward edge");
      continue;
    }
    // check whether the new extension is minimal
    types::DFSCode new_code = code;
    new_code.push_back(h_frequent_dfs_elem[i]);
    if(new_code.dfs_code_is_min() == false) {
      TRACE(*logger, "NOT MINIMAL extension: " << h_frequent_dfs_elem[i].to_string() << "; support: " << h_frequent_dfs_supports[i]);
      continue;
    }

    output->output_graph(new_code, h_frequent_dfs_supports[i]);

    TRACE(*logger, "PROCESSING extension #" << i << "; h_frequent_dfs_elem[i]: " << h_frequent_dfs_elem[i].to_string());


    // do "half copy" of the embeddings, i.e., move only the pointers into the
    // d_new_embeddings without doing actual copying of the content. This saves
    // memory. HOWEVER: special care must be taken when deallocating. Instead of
    // calling delete_from_device the d_half_deallocate must be called
    TRACE(*logger, "half-copying embeddings");
    embedding_list_columns d_new_embeddings(false);
    int new_embed_column_length;
    embedding_element *d_new_embed_column = 0;
    dfs_extension_element_set_t new_backward_edges(backward_edges);

    d_new_embeddings = embeddings.d_get_half_copy();
    if(h_frequent_dfs_elem[i].is_forward()) {
      // extend the current embedding
      TRACE(*logger, "forward dfs element: " << h_frequent_dfs_elem[i].to_string());
      TRACE(*logger, "adding new column to d_new_embeddings");
      extension_element_t *block_start = d_exts_result + block_offsets[h_frequent_dfs_elem[i].from];
      int block_length = get_block_size(h_frequent_dfs_elem[i].from, block_offsets, embeddings.columns_count);

      extract_embedding_column(block_start, block_length, h_frequent_dfs_elem[i], d_new_embed_column, new_embed_column_length, scanner);
      d_new_embeddings.d_extend_by_one_column(h_frequent_dfs_elem[i], d_new_embed_column, new_embed_column_length);
    } else {
      TRACE(*logger, "backward dfs element: " << h_frequent_dfs_elem[i].to_string());
      TRACE(*logger, "offset: " << block_offsets[h_frequent_dfs_elem[i].from]);
      TRACE(*logger, "length: " << get_block_size(h_frequent_dfs_elem[i].from, block_offsets, embeddings.columns_count));
      filter_backward_embeddings(embeddings,
                                 d_exts_result + block_offsets[h_frequent_dfs_elem[i].from],
                                 get_block_size(h_frequent_dfs_elem[i].from, block_offsets, embeddings.columns_count),
                                 h_frequent_dfs_elem[i],
                                 d_new_embed_column,
                                 new_embed_column_length,
                                 scanner);

      d_new_embeddings.d_replace_last_column(d_new_embed_column, new_embed_column_length);

      new_backward_edges.insert(h_frequent_dfs_elem[i]);
      backward_edges.insert(h_frequent_dfs_elem[i]);
    }

    if(execute_tests) {
      TRACE(*logger, "TESTING EMBEDDINGS");
      test_embeddings(embeddings, code, scheduled_rmpath, h_frequent_dfs_elem[i], d_new_embed_column, new_embed_column_length);
    }


    // execute the mainloop recursivelly
    mainloop(d_new_embeddings, new_code, h_frequent_dfs_supports[i], new_backward_edges, scheduled_rmpath_local, not_frequent_extensions_local);
    d_new_embeddings.d_half_deallocate();


    CUDAFREE(d_new_embed_column, *logger);
  } // for i

  CUDAFREE(d_exts_result, *logger);

  //INFO(*logger, "deleting h_dfs_elem: " << h_dfs_elem << "; this: " << (void*) (this));
  delete [] h_dfs_elem;
  h_dfs_elem = 0;
  delete [] block_offsets;
  block_offsets = 0;

  delete [] h_frequent_dfs_elem;
  delete [] h_frequent_dfs_supports;
  h_frequent_dfs_elem = 0;
  h_frequent_dfs_supports = 0;



  DEBUG(*logger, "=============== exiting mainloop");
} // gspan_cuda_no_sort::mainloop


void gspan_cuda_no_sort::test_extensions(types::embedding_list_columns &embeddings,
                                         types::DFSCode code,
                                         types::RMPath scheduled_rmpath,
                                         types::DFS *h_frequent_dfs_elem,
                                         int *h_frequent_dfs_supports,
                                         int frequent_candidate_count)
{
  TRACE(*logger, " test_extensions =====================================================");
  const types::RMPath &rmpath = code.buildRMPath();
  types::RMPath cuda_rmpath = convert_rmpath(rmpath, code);
  types::RMPath scheduled_rmpath_local;
  extension_element_t *d_exts_result = 0;
  int exts_result_length = 0;
  int minlabel = code[0].fromlabel;

  filter_rmpath(cuda_rmpath, scheduled_rmpath_local, scheduled_rmpath);

  TRACE(*logger, "scheduled_rmpath_local: " << utils::print_vector(scheduled_rmpath_local));
  TRACE(*logger, "scheduled_rmpath: " << utils::print_vector(scheduled_rmpath));


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
                     scheduled_rmpath_local);

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

  int minimal_dfs_extensions_count = 0;
  for(int i = 0; i < dfs_array_length; i++) {
    types::DFSCode newcode;
    newcode.push_back(h_dfs_elem[i]);
    if(newcode.dfs_code_is_min() == true) {
      minimal_dfs_extensions_count++;
    } else {
      TRACE(*logger, "non-minimal extension: " << h_dfs_elem[i].to_string());
    }
  }

  std::map<types::DFS, int, embedding_extension_compare_less_then_t> correct_extensions;
  std::map<types::DFS, int, embedding_extension_compare_less_then_t> test_extensions;
  for(int i = 0; i < dfs_array_length; i++) {
    if(h_support[i] < minimal_support) continue;
    types::DFSCode newcode = code;
    newcode.push_back(h_dfs_elem[i]);
    if(newcode.dfs_code_is_min() == true) {
      correct_extensions.insert(std::make_pair(h_dfs_elem[i], h_support[i]));
    }
  } // for dfs_index


  for(int i = 0; i < frequent_candidate_count; i++) {
    types::DFSCode newcode = code;
    newcode.push_back(h_frequent_dfs_elem[i]);
    if(newcode.dfs_code_is_min() == true) {
      test_extensions.insert(std::make_pair(h_frequent_dfs_elem[i], h_frequent_dfs_supports[i]));
    }
  } // for i

  if(correct_extensions.size() != test_extensions.size()) {
    std::stringstream correct_extensions_ss;
    std::stringstream test_extensions_ss;
    std::map<types::DFS, int, embedding_extension_compare_less_then_t>::iterator it;
    for(it = correct_extensions.begin(); it != correct_extensions.end(); it++) {
      correct_extensions_ss << it->first.to_string() << "; ";
    }

    for(it = test_extensions.begin(); it != test_extensions.end(); it++) {
      test_extensions_ss << it->first.to_string() << "; ";
    }
    TRACE(*logger, "correct extensions: " << correct_extensions_ss.str());
    TRACE(*logger, "test extensions: " << test_extensions_ss.str());
    throw std::runtime_error("Incorrect frequent candidate count");
  }

  std::map<types::DFS, int, embedding_extension_compare_less_then_t>::iterator itM;
  for(itM = test_extensions.begin(); itM != test_extensions.end(); itM++) {
    std::map<types::DFS, int, embedding_extension_compare_less_then_t>::iterator itM2;
    itM2 = correct_extensions.find(itM->first);
    if(itM2 == correct_extensions.end()) {
      std::stringstream ss;
      ss << itM->first.to_string() << " not found in correct_extensions";
      throw std::runtime_error(ss.str());
    } // if

    if(itM->second != itM2->second) {
      std::stringstream ss;
      ss << "support of " << itM->first.to_string() << " do not match the correct support: " << itM->second << " != " << itM2->second;
      throw std::runtime_error(ss.str());
    } // if
  } // for itM

  CUDAFREE(d_exts_result, *logger);

  delete [] h_dfs_elem;
  delete [] h_dfs_offsets;
  delete [] h_support;
}



void gspan_cuda_no_sort::test_embeddings(types::embedding_list_columns &embeddings,
                                         types::DFSCode code,
                                         types::RMPath scheduled_rmpath,
                                         types::DFS dfs_elem,
                                         embedding_element *d_embed_column_test,
                                         int d_embed_column_test_length)
{
  DEBUG(*logger, "testing embeddings");
  TRACE(*logger, " test_embeddings =====================================================");
  TRACE(*logger, "dfs code: " << code.to_string());
  TRACE(*logger, "dfs element: " << dfs_elem.to_string());
  const types::RMPath &rmpath = code.buildRMPath();
  types::RMPath cuda_rmpath = convert_rmpath(rmpath, code);
  types::RMPath scheduled_rmpath_local;
  extension_element_t *d_exts_result = 0;
  int exts_result_length = 0;
  int minlabel = code[0].fromlabel;


  TRACE(*logger, "printing device embeddings");
  types::embedding_list_columns h_embeddings(true);
  h_embeddings.copy_from_device(&embeddings);
  //int embedding_count = h_embeddings.columns_lengths[h_embeddings.columns_count - 1];
  //TRACE(*logger, "device embeddings: " << endl << h_embeddings.to_string_with_labels(h_cuda_graph_database, code));
  //TRACE(*logger, "printing embeddings, size: " << embedding_count);
  //for(int i = 0; i < embedding_count; i++) {
  //TRACE(*logger, i << ": " << h_embeddings.embedding_to_string(i));
  //}


  filter_rmpath(cuda_rmpath, scheduled_rmpath_local, scheduled_rmpath);

  TRACE(*logger, "scheduled_rmpath_local: " << utils::print_vector(scheduled_rmpath_local));
  TRACE(*logger, "scheduled_rmpath: " << utils::print_vector(scheduled_rmpath));


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
                     scheduled_rmpath_local);

  TRACE5(*logger, "d_exts_result: " << print_d_array(d_exts_result, exts_result_length));

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

  int dfs_index = 0;
  int new_embed_correct_length = 0;
  embedding_element *d_new_embed_correct = 0;


  for(; dfs_index < dfs_array_length; dfs_index++) {
    if(dfs_elem == h_dfs_elem[dfs_index]) break;
  } // for dfs_index

  //compute the number of the embeddings
  if(dfs_index == dfs_array_length - 1) {
    new_embed_correct_length = exts_result_length - h_dfs_offsets[dfs_index];
  } else {
    new_embed_correct_length = h_dfs_offsets[dfs_index + 1] - h_dfs_offsets[dfs_index];
  }

  if(h_dfs_elem[dfs_index].is_forward()) {
    TRACE(*logger, "processing forward edge");

    // copy the embeddings from the extension_element_t on device into the
    // embedding_element, creating new column in the embedding
    CUDAMALLOC(&d_new_embed_correct, sizeof(embedding_element) * new_embed_correct_length, *logger);
    cudapp::cuda_computation_parameters copy_params = cudapp::cuda_configurator::get_computation_parameters(new_embed_correct_length, 512);
    TRACE(*logger, "copying extensions, from: " << 0 << "; to: " << new_embed_correct_length);
    cudapp::for_each(0, new_embed_correct_length, copy_embedding_info(d_new_embed_correct, d_exts_result, h_dfs_offsets[dfs_index]), copy_params);

    // extend the current embedding
    TRACE(*logger, "adding new column to d_new_embed_correct");
  } else {
    TRACE(*logger, "processing backward edge, filtering embeddings, embedding count: " << new_embed_correct_length);
    TRACE(*logger, "filtering for: " << h_dfs_elem[dfs_index].to_string());
    TRACE(*logger, "input array: " << print_d_array(d_exts_result + h_dfs_offsets[dfs_index], new_embed_correct_length));
    filter_backward_embeddings(embeddings,
                               dfs_index,
                               d_exts_result,
                               exts_result_length,
                               h_dfs_offsets,
                               h_dfs_elem,
                               dfs_array_length,
                               d_new_embed_correct,
                               new_embed_correct_length,
                               scanner,
                               exec_config);
  }


  types::embedding_element *correct_column = 0;
  types::embedding_element *test_column = 0;


  copy_d_array_to_h(d_embed_column_test, d_embed_column_test_length, test_column);
  copy_d_array_to_h(d_new_embed_correct, new_embed_correct_length, correct_column);

  std::sort(correct_column, correct_column + new_embed_correct_length, embedding_element_less_then_t());
  std::sort(test_column, test_column + new_embed_correct_length, embedding_element_less_then_t());

  bool same = std::equal(correct_column, correct_column + new_embed_correct_length, test_column);
  if(same == false) {
    std::pair<types::embedding_element*, types::embedding_element*> diff;
    diff = std::mismatch(correct_column, correct_column + new_embed_correct_length, test_column);

    CRITICAL_ERROR(*logger, "diference (position) correct: " << (diff.first - correct_column) << "; test: " << diff.second - test_column);
    CRITICAL_ERROR(*logger, "correct array: " << print_h_array(correct_column, new_embed_correct_length));
    CRITICAL_ERROR(*logger, "test array: " << print_h_array(test_column, new_embed_correct_length));
    throw std::runtime_error("embeddings are not the same !");
  }
  delete [] correct_column;
  delete [] test_column;

  delete [] h_dfs_elem;
  delete [] h_dfs_offsets;
  delete [] h_support;

  h_embeddings.delete_from_host();

  CUDAFREE(d_new_embed_correct, *logger);
  CUDAFREE(d_exts_result, *logger);
} // gspan_cuda_no_sort::test_embeddings



} // namespace gspan_cuda


