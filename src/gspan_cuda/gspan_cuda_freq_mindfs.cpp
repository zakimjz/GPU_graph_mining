#include <gspan_cuda_freq_mindfs.hpp>


using types::embedding_list_columns;
using types::embedding_element;

namespace gspan_cuda {

bool gspan_cuda_freq_mindfs::is_min_dfs_ext(const types::DFS &dfs_elem, const types::DFSCode &code) const
{
  INFO(*logger, "gspan_cuda_mindfs::should_filter_non_min");
  types::DFSCode new_code = code;
  new_code.push_back(dfs_elem);
  bool ismin = new_code.dfs_code_is_min();

  //INFO(*logger, "gspan_cuda_mindfs::should_filter_non_min; ismin: " << ismin);

  return ismin;
}

/*
   void gspan_cuda_freq_mindfs::filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::DFS *h_dfs_elem, int *supports, int h_dfs_elem_count)
   {
   gspan_rmpath_out = gspan_rmpath_in;
   }


   void gspan_cuda_freq_mindfs::filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::RMPath &gspan_rmpath_has_extension)
   {
   gspan_rmpath_out = gspan_rmpath_in;
   }

 */

void gspan_cuda_freq_mindfs::run_intern2()
{
  types::edge_gid_list3_t root;
  fill_root(root);
  fill_labels();
  CUDAMALLOC(&d_graph_flags, sizeof(int) * h_cuda_graph_database.db_size, *logger);
  d_graph_flags_length = h_cuda_graph_database.db_size;

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
        unsigned int supp = 0; //tolabel->second.size();
        int last_id = -1;
        for(int i = 0; i < tolabel->second.size(); i++) {
          if(tolabel->second[i] != last_id) {
            last_id = tolabel->second[i];
            supp++;
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
        extension_map_t dfs_support_info;
        types::RMPath scheduled_rmpath;
        scheduled_rmpath.push_back(0);
        scheduled_rmpath.push_back(1);

        mainloop(first_embeddings, code, supp, new_backward_edges, scheduled_rmpath, dfs_support_info);

        first_embeddings.delete_from_device();
      } // for tolabel
    } // for elabel
  } // for fromlabel

  CUDAFREE(d_graph_flags, *logger);
  CUDAFREE(d_fwd_flags, *logger);
  CUDAFREE(d_bwd_flags, *logger);
  CUDAFREE(d_fwd_block_dfs_array, *logger);
  CUDAFREE(d_bwd_block_dfs_array, *logger);

  CUDAFREE(d_ext_block_offsets, *logger);
  CUDAFREE(d_ext_dfs_array, *logger);

  INFO(*logger, memory_checker::print_memory_usage());
} // gspan_cuda::run_intern2




void gspan_cuda_freq_mindfs::fill_supports(extension_element_t *d_exts_block,
                                           int exts_block_length,
                                           types::DFS *h_block_dfs_elems,
                                           types::DFS *d_block_dfs_elems,
                                           int dfs_elem_length,
                                           int max_graph_vertex_count,
                                           int mapped_db_size,
                                           int col_count,
                                           std::vector<types::DFS> &freq_dfs_elems,
                                           std::vector<int> &freq_dfs_elems_supports,
                                           extension_map_t &dfs_support_info)
{
  TRACE2(*logger, "--------------------------------------------------------------------");
  for(int i = 0; i < dfs_elem_length; i++) {
    TRACE3(*logger, "should compute support for dfs: " << h_block_dfs_elems[i].to_string());
  }
  int dfs_per_cs_call = compute_support_max_memory_size / mapped_db_size;
  //int dfs_per_cs_call = 5;//compute_support_max_memory_size / mapped_db_size;


  if(dfs_per_cs_call > dfs_elem_length) dfs_per_cs_call = dfs_elem_length;
  int last_block_size = dfs_elem_length % dfs_per_cs_call;
  int compute_support_calls = dfs_elem_length / dfs_per_cs_call + (last_block_size > 0 ? 1 : 0);
  TRACE2(*logger, "compute_support_calls: " << compute_support_calls);
  TRACE2(*logger, "dfs_per_cs_call: " << dfs_per_cs_call);
  TRACE2(*logger, "dfs_elem_length: " << dfs_elem_length);

  for(int i = 0; i < compute_support_calls; i++) {
    TRACE2(*logger, "------------ computing support for dfs block " << i);
    int dfs_in_this_block = dfs_per_cs_call;
    if(i == compute_support_calls - 1 && dfs_elem_length % dfs_per_cs_call != 0) {
      dfs_in_this_block = dfs_elem_length % dfs_per_cs_call;
    }

    int *supports = 0;
    int dfs_start_idx = dfs_per_cs_call * i;
    TRACE2(*logger, "dfs_start_idx: " << dfs_start_idx);
    TRACE2(*logger, "dfs_in_this_block: " << dfs_in_this_block);

    compute_support_remapped_db_multiple_dfs(d_exts_block,
                                             exts_block_length,
                                             d_block_dfs_elems + dfs_start_idx,
                                             dfs_in_this_block,
                                             supports,
                                             max_graph_vertex_count,
                                             d_graph_flags,
                                             d_graph_flags_length,
                                             d_graph_boundaries_scan,
                                             d_graph_boundaries_scan_length,
                                             mapped_db_size);
    for(int dfs_id = 0; dfs_id < dfs_in_this_block; dfs_id++) {

      types::DFS tmp_dfs = h_block_dfs_elems[dfs_start_idx + dfs_id];
      tmp_dfs.to++;

      if(supports[dfs_id] >= minimal_support) {
        freq_dfs_elems_supports.push_back(supports[dfs_id]);
        freq_dfs_elems.push_back(h_block_dfs_elems[dfs_start_idx + dfs_id]);
        TRACE2(*logger, "dfs: " << h_block_dfs_elems[dfs_start_idx + dfs_id].to_string() << " is frequent");

        if (tmp_dfs.is_forward() )
          dfs_support_info[tmp_dfs] = FREQ;
        //backward edge will not be valid in the next level

      } else {
        TRACE2(*logger, "dfs: " << h_block_dfs_elems[dfs_start_idx + dfs_id].to_string() << " is NOT frequent");

        dfs_support_info[tmp_dfs] = NOT_FREQ;
      }
    } // for dfs_id

    delete [] supports;
  } // for i
}





void gspan_cuda_freq_mindfs::filter_extensions(extension_element_t *d_exts_result,
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
                                               int &frequent_candidate_count)
{
  TRACE4(*logger, "#############################################################################################");
  TRACE4(*logger, "#############################################################################################");
  TRACE4(*logger, "filtering extensions, dfs_array_length: " << dfs_array_length);


  // Create list of extensions per extension_element_t block;
  std::vector<int> prefiltered_dfs_elems_offsets(ext_num_blocks + 1, -1);
  std::vector<int> prefiltered_dfs_elems_sizes(ext_num_blocks, 0);

  types::DFS *prefiltered_dfs_elems = new types::DFS[dfs_array_length];
  int prefiltered_dfs_elems_size = 0;

  // prefilter the h_dfs_elem array against the not_frequent and non-min extensions
  for(int i = 0; i < dfs_array_length; i++) {
    TRACE4(*logger, "h_dfs_elem[" << i << "]: " << h_dfs_elem[i].to_string());

    if( dfs_support_info_in.count(h_dfs_elem[i]) == 0 ) { //seen for the first time

      if( is_min_dfs_ext(h_dfs_elem[i], code) ) {
        prefiltered_dfs_elems[prefiltered_dfs_elems_size] = h_dfs_elem[i];
        prefiltered_dfs_elems_size++;
      }else{
        if( h_dfs_elem[i].is_forward() ) {
          types::DFS tmp_dfs = h_dfs_elem[i];
          tmp_dfs.to++;
          dfs_support_info_out[tmp_dfs] = UNKNOWN;
        }
        //backward edge is not needed to be stored, because it won't be a valid edge in the next level

      } //else

    }else { //already in the map

      if(dfs_support_info_in[h_dfs_elem[i]] == UNKNOWN && is_min_dfs_ext(h_dfs_elem[i], code)) {
        prefiltered_dfs_elems[prefiltered_dfs_elems_size] = h_dfs_elem[i];
        prefiltered_dfs_elems_size++;
      }

      if(dfs_support_info_in[h_dfs_elem[i]] == FREQ) {
        prefiltered_dfs_elems[prefiltered_dfs_elems_size] = h_dfs_elem[i];
        prefiltered_dfs_elems_size++;
      }

      //NOT_FREQ's are ommited from the prefiltered list

      //Now copy to the output map, the FREQ and NON_FREQ extensions will also be updated after support computation
      if( h_dfs_elem[i].is_forward() ) {
        types::DFS tmp_dfs = h_dfs_elem[i];
        tmp_dfs.to++;
        dfs_support_info_out[tmp_dfs] = dfs_support_info_in[h_dfs_elem[i]];
      }

      //backward edge is not needed to be stored, because it won't be a valid edge in the next level

    } //else

  } // for i

  if(prefiltered_dfs_elems_size == 0) return;

  // create blocks of dfs elements: offsets and sizes. The dfs
  // elements are stored in prefiltered_dfs_elems. The created
  // offsets/sizes are related to prefiltered_dfs_elems.
  int last_from = prefiltered_dfs_elems[0].from;
  prefiltered_dfs_elems_offsets[prefiltered_dfs_elems[0].from] = 0;
  for(int i = 0; i < prefiltered_dfs_elems_size; i++) {
    if(prefiltered_dfs_elems[i].from != last_from) {
      prefiltered_dfs_elems_offsets[prefiltered_dfs_elems[i].from] = i;
      TRACE4(*logger, "new dfs block: " << prefiltered_dfs_elems[i].from << " starts at: " << i);
      last_from = prefiltered_dfs_elems[i].from;
    } // if
  } // for i
  prefiltered_dfs_elems_offsets.back() = prefiltered_dfs_elems_size;

  for(int i = 0; i < prefiltered_dfs_elems_offsets.size() - 1; i++) {
    if(prefiltered_dfs_elems_offsets[i] == -1) {
      prefiltered_dfs_elems_sizes[i] = 0;
      continue;
    }
    prefiltered_dfs_elems_sizes[i] = get_block_size(i, prefiltered_dfs_elems_offsets.data(), prefiltered_dfs_elems_offsets.size() - 1);
    TRACE4(*logger, "block size: " << prefiltered_dfs_elems_sizes[i]);
  } // for i



  // copy the prefiltered dfs elements to device
  types::DFS *d_prefiltered_dfs_elems = 0;
  CUDAMALLOC(&d_prefiltered_dfs_elems, sizeof(types::DFS) * prefiltered_dfs_elems_size, *logger);
  CUDA_EXEC(cudaMemcpy(d_prefiltered_dfs_elems, prefiltered_dfs_elems, sizeof(types::DFS) * prefiltered_dfs_elems_size, cudaMemcpyHostToDevice), *logger);

  std::vector<types::DFS> freq_dfs_elems;
  std::vector<int> freq_dfs_elems_supports;

  // decide the size of the memory used for storing flags for various
  // dbs allocate the memory and start calling the
  // compute_support_remapped_db_multiple_dfs this step involves call
  // to remap_database_graph_ids.
  TRACE4(*logger, "prefiltered_dfs_elems_offsets.size(): " << prefiltered_dfs_elems_offsets.size());
  for(int block_id = 0; block_id < prefiltered_dfs_elems_sizes.size(); block_id++) {
    if(ext_block_offsets[block_id] == -1) continue;

    TRACE2(*logger, "processing ext block: " << block_id << " ==================================================");

    int dfs_in_this_block = prefiltered_dfs_elems_sizes[block_id];
    int dfs_block_start = prefiltered_dfs_elems_offsets[block_id];
    if(dfs_in_this_block == 0) continue;

    TRACE2(*logger, "dfs_block_start: " << dfs_block_start);

    int mapped_db_size = 0;
    int ext_bo_start = ext_block_offsets[block_id];
    int ext_bo_length = get_block_size(block_id, ext_block_offsets, ext_num_blocks);
    TRACE4(*logger, "ext_bo_start: " << ext_bo_start << "; ext_bo_length: " << ext_bo_length);
    remap_database_graph_ids(d_exts_result + ext_bo_start,
                             ext_bo_length,
                             h_cuda_graph_database.max_graph_vertex_count,
                             d_graph_boundaries_scan,
                             d_graph_boundaries_scan_length,
                             mapped_db_size);


    fill_supports(d_exts_result + ext_bo_start,
                  ext_bo_length,
                  prefiltered_dfs_elems + dfs_block_start,
                  d_prefiltered_dfs_elems + dfs_block_start,
                  dfs_in_this_block,
                  h_cuda_graph_database.max_graph_vertex_count,
                  mapped_db_size,
                  col_count,
                  freq_dfs_elems,
                  freq_dfs_elems_supports,
                  dfs_support_info_out);
  } // for(block_id)


  // copy the content of freq_dfs_elems to the output array
  h_frequent_dfs_elem = new types::DFS[freq_dfs_elems_supports.size()];
  h_frequent_dfs_supports = new int[freq_dfs_elems_supports.size()];
  frequent_candidate_count = freq_dfs_elems_supports.size();
  TRACE4(*logger, "filtering dfs extensions using support: " << minimal_support);
  for(int dfs_id = 0; dfs_id < freq_dfs_elems_supports.size(); dfs_id++) {
    TRACE4(*logger, "frequent: " << freq_dfs_elems[dfs_id].to_string());
    h_frequent_dfs_elem[dfs_id] = freq_dfs_elems[dfs_id];
    TRACE4(*logger, "copied: " << h_frequent_dfs_elem[dfs_id].to_string());
    h_frequent_dfs_supports[dfs_id] = freq_dfs_elems_supports[dfs_id];


  } // for dfs_id


  CUDAFREE(d_prefiltered_dfs_elems, *logger);
  delete [] prefiltered_dfs_elems;
}


void gspan_cuda_freq_mindfs::mainloop(types::embedding_list_columns &embeddings,
                                      types::DFSCode &code,
                                      int support,
                                      dfs_extension_element_set_t backward_edges,
                                      types::RMPath scheduled_rmpath,
                                      extension_map_t &dfs_support_info)
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
                     dfs_array_length);


  TRACE(*logger, "dfs extensions: " << dfs_array_length);
  for(int i = 0; i < dfs_array_length; i++) {
    TRACE(*logger, "dfs: " << h_dfs_elem[i].to_string());
  }

  scheduled_rmpath_local.clear();


  extension_map_t dfs_support_info_out;

  // h_frequent_dfs_elem stores only the FREQUENT extensions
  types::DFS *h_frequent_dfs_elem = 0;
  int *h_frequent_dfs_supports = 0;
  int frequent_candidate_count = 0;
  // filter-out infrequent DFS from h_dfs_elem
  filter_extensions(d_exts_result, exts_result_length, // the extensions
                    h_dfs_elem, dfs_array_length, block_offsets, num_blocks, embeddings.columns_count, // input DFS elements
                    code,
                    dfs_support_info, // output not frequent extensions
                    dfs_support_info_out, // output not frequent extensions
                    h_frequent_dfs_elem, h_frequent_dfs_supports, frequent_candidate_count); // output of the filtering, the frequent extensions

  //INFO(*logger, "h_dfs_elem: " << h_dfs_elem << "; dfs_array_length: " << dfs_array_length);
  //INFO(*logger, "h_dfs_elem: " << h_dfs_elem);

  if(execute_tests) {
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
    types::embedding_list_columns d_new_embeddings(false);
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

      extract_embedding_column(block_start, block_length, h_frequent_dfs_elem[i], d_new_embed_column, new_embed_column_length);
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
                                 new_embed_column_length);

      d_new_embeddings.d_replace_last_column(d_new_embed_column, new_embed_column_length);

      new_backward_edges.insert(h_frequent_dfs_elem[i]);
      backward_edges.insert(h_frequent_dfs_elem[i]);
    }

    if(execute_tests) {
      test_embeddings(embeddings, code, scheduled_rmpath, h_frequent_dfs_elem[i], d_new_embed_column, new_embed_column_length);
    }


    // execute the mainloop recursivelly
    mainloop(d_new_embeddings, new_code, h_frequent_dfs_supports[i], new_backward_edges, scheduled_rmpath_local, dfs_support_info_out);
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



gspan_cuda_freq_mindfs::gspan_cuda_freq_mindfs()
{
  logger = Logger::get_logger("GSPAN_CUDA_FREQ_MINDFS");
}


gspan_cuda_freq_mindfs::~gspan_cuda_freq_mindfs()
{
}

} // namespace gspan_cuda


