#include <gspan_cuda_lists.hpp>
#include <cuda_gspan_ops.hpp>
#include <kernel_execution.hpp>
#include <embedding_lists.hpp>
#include <graph_types.hpp>

#include <algorithm>
#include <cassert>

using types::Graph;

using types::Projected_map3;
using types::Projected_map2;
using types::Projected_map1;
using types::Projected_iterator3;
using types::Projected_iterator2;
using types::Projected_iterator1;

using types::Projected_riterator3;

using types::embedding_list_columns;

namespace gspan_cuda {

gspan_cuda_lists::gspan_cuda_lists()
{
  execute_tests = false;
  logger = Logger::get_logger("GSPAN_CUDA_LISTS");
}

gspan_cuda_lists::~gspan_cuda_lists()
{

}




void gspan_cuda_lists::run_intern2()
{
  types::edge_gid_list3_t root;
  fill_root(root);

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
        types::Projected projected_empty;

        TRACE(*logger, "original projection size: " << tolabel->second.size());
        DEBUG(*logger, memory_checker::print_memory_usage());
        dfs_extension_element_set_t new_backward_edges;
        types::RMPath scheduled_rmpath;
        scheduled_rmpath.push_back(0);
        scheduled_rmpath.push_back(1);
        list_extensions_vec_t extensions;
        mainloop(first_embeddings, code, new_backward_edges, extensions);
        first_embeddings.delete_from_device();
      } // for tolabel
    } // for elabel
  } // for fromlabel
} // gspan_cuda::run_intern2

void gspan_cuda_lists::run_intern(void)
{
  std::map<unsigned int, unsigned int> singleVertexLabel;
  std::map<unsigned int, std::map<unsigned int, unsigned int> > singleVertex;

  // Do single node handling, as the normal gspan DFS code based processing
  // cannot find subgraphs of size |subg|==1.  Hence, we find frequent node
  // labels explicitly.
  for(unsigned int id = 0; id < graph_database.size(); ++id) {
    for(unsigned int nid = 0; nid < graph_database[id].size(); ++nid) {
      if(singleVertex[id][graph_database[id][nid].label] == 0) {
        // number of graphs it appears in
        singleVertexLabel[graph_database[id][nid].label] += 1;
      }

      singleVertex[id][graph_database[id][nid].label] += 1;
    } // for nid
  } // for it


  // All minimum support node labels are frequent 'subgraphs'.
  // singleVertexLabel[nodelabel] gives the number of graphs it appears
  // in.
  for(std::map<unsigned int, unsigned int>::iterator it = singleVertexLabel.begin(); it != singleVertexLabel.end(); ++it) {
    if((*it).second < minimal_support)
      continue;

    unsigned int frequent_label = (*it).first;

    // Found a frequent node label, report it.
    Graph g;
    g.resize(1);
    g[0].label = frequent_label;

    // [graph_id] = count for current substructure
    std::vector<unsigned int> counts(graph_database.size());
    for(std::map<unsigned int, std::map<unsigned int, unsigned int> >::iterator it2 = singleVertex.begin(); it2 != singleVertex.end(); ++it2) {
      counts[(*it2).first] = (*it2).second[frequent_label];
    } // for it2

    std::map<unsigned int, unsigned int> gycounts;
    for(unsigned int n = 0; n < counts.size(); ++n)
      gycounts[n] = counts[n];

    report_single(g, gycounts);
  } // for it




  ///////////////////////////////////////////////////////////////////////
  types::EdgeList edges;
  Projected_map3 root;

  for(unsigned int id = 0; id < graph_database.size(); ++id) {
    Graph &g = graph_database[id];
    for(unsigned int from = 0; from < g.size(); ++from) {
      if(get_forward_root(g, g[from], edges)) {
        for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
          root[g[from].label][(*it)->elabel][g[(*it)->to].label].push(id, *it, 0);
      } // if
    } // for from
  } // for id

  for(Projected_iterator3 fromlabel = root.begin();
      fromlabel != root.end(); ++fromlabel) {
    for(Projected_iterator2 elabel = fromlabel->second.begin();
        elabel != fromlabel->second.end(); ++elabel) {
      for(Projected_iterator1 tolabel = elabel->second.begin();
          tolabel != elabel->second.end(); ++tolabel) {
        unsigned int supp = support(tolabel->second);
        if(supp < minimal_support) continue;

        // Build the initial two-node graph.  It will be grownrecursively within project.
        types::DFSCode code;
        code.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
        DEBUG(*logger, "*************** creating first embeddings for: " << code.to_string());
        if(output) output->output_graph(code, supp);

        types::embedding_list_columns first_embeddings(false);
        create_first_embeddings(code[0], &d_cuda_graph_database, &first_embeddings, exec_config);
        types::Projected projected_empty;

        TRACE(*logger, "original projection size: " << tolabel->second.size());
        DEBUG(*logger, memory_checker::print_memory_usage());
        dfs_extension_element_set_t new_backward_edges;

        list_extensions_vec_t extensions;
        mainloop(first_embeddings, code, new_backward_edges, extensions);
        first_embeddings.delete_from_device();
      } // for tolabel
    } // for elabel
  } // for fromlabel
} //



void gspan_cuda_lists::run()
{
  run_intern2();
}

void gspan_cuda_lists::get_rmpath_extensions(types::embedding_list_columns &embeddings,
                                             types::DFSCode code,
                                             types::RMPath cuda_rmpath,
                                             types::RMPath scheduled_rmpath,
                                             new_extensions_t &ne)
{
  int minlabel = code[0].fromlabel;

  TRACE(*logger, "computing extensions");
  DEBUG(*logger, "cuda_rmpath: " << utils::print_vector(cuda_rmpath));
  DEBUG(*logger, "scheduled_rmpath: " << utils::print_vector(scheduled_rmpath));

  types::RMPath host_rmpath = code.buildRMPath();
  get_all_extensions(&d_cuda_graph_database,
                     &cuda_allocs_get_all_ext,
                     &embeddings,
                     cuda_rmpath,
                     host_rmpath,
                     code,
                     ne.d_exts_result,
                     ne.exts_result_length,
                     minlabel,
                     exec_config,
                     scheduled_rmpath);


  if(ne.exts_result_length == 0) {
    ne.delete_from_device();
    DEBUG(*logger, "new extensions removed from device");
    //CUDAFREE(ne.d_exts_result, *logger);
    return;
  }


  DEBUG(*logger, "computing supports for extensions");
  get_support_for_extensions(d_cuda_graph_database.max_graph_vertex_count,
                             ne.d_exts_result,
                             ne.exts_result_length,
                             ne.dfs_array_length,
                             ne.h_dfs_elem,
                             ne.h_dfs_offsets,
                             ne.h_support,
                             exec_config);

} // gspan_cuda_lists::get_rmpath_extensions


void gspan_cuda_lists::get_rmpath_extensions(types::embedding_list_columns &embeddings,
                                             types::DFSCode code,
                                             types::RMPath cuda_rmpath,
                                             types::RMPath scheduled_rmpath,
                                             list_extensions_vec_t &exts)
{
  new_extensions_t ne;
  get_rmpath_extensions(embeddings, code, cuda_rmpath, scheduled_rmpath, ne);



  for(int i = 0; i < ne.dfs_array_length; i++) {
    if(ne.h_support[i] < minimal_support) continue;
    int curr_dfs_ext_length = -1;
    if(i == ne.dfs_array_length - 1) {
      curr_dfs_ext_length = ne.exts_result_length - ne.h_dfs_offsets[i];
    } else {
      curr_dfs_ext_length = ne.h_dfs_offsets[i + 1] - ne.h_dfs_offsets[i];
    }

    if(ne.h_dfs_elem[i].is_forward()) {
      types::embedding_element *d_new_embed_vertices = 0;
      // copy the embeddings from the extension_element_t on device into the
      // embedding_element, creating new column in the embedding
      CUDAMALLOC(&d_new_embed_vertices, sizeof(types::embedding_element) * curr_dfs_ext_length, *logger);
      cudapp::cuda_computation_parameters copy_params = cudapp::cuda_configurator::get_computation_parameters(curr_dfs_ext_length, 512);

      TRACE(*logger, "dfs: " << ne.h_dfs_elem[i] << "; copying extensions, from: " << 0 << "; to: " << curr_dfs_ext_length << "; offset: " << ne.h_dfs_offsets[i]);
      cudapp::for_each(0, curr_dfs_ext_length, copy_embedding_info(d_new_embed_vertices, ne.d_exts_result, ne.h_dfs_offsets[i]), copy_params);

      //
      embedding_extension_t forward_extension(false, FRWD);
      forward_extension.embedding_column = d_new_embed_vertices;
      forward_extension.col_length = curr_dfs_ext_length;
      forward_extension.dfs_elem = ne.h_dfs_elem[i];
      forward_extension.support = ne.h_support[i];
      TRACE(*logger, "produced forward extension: " << forward_extension.to_string());
      exts.push_back(forward_extension);
    } else {
      // convert the backward extension (stored as extension_element_t) to integer offsets relative
      // to the last column
      int *d_new_embed_backlinks = 0;
      CUDAMALLOC(&d_new_embed_backlinks, sizeof(int) * curr_dfs_ext_length, *logger);

      TRACE(*logger, "processing backward edge, filtering embeddings, embedding count: " << curr_dfs_ext_length);
      cudapp::cuda_computation_parameters copy_params = cudapp::cuda_configurator::get_computation_parameters(curr_dfs_ext_length, 512);
      cudapp::for_each(0, curr_dfs_ext_length, copy_backlink_info(d_new_embed_backlinks, ne.d_exts_result, ne.h_dfs_offsets[i]), copy_params);

      embedding_extension_t backward_extension(false, BKWD);
      backward_extension.dfs_elem = ne.h_dfs_elem[i];
      backward_extension.support = ne.h_support[i];
      backward_extension.filtered_emb_offsets = d_new_embed_backlinks;
      backward_extension.filtered_emb_offsets_length = curr_dfs_ext_length;
      TRACE(*logger, "produced backward extension: " << backward_extension.to_string());
      exts.push_back(backward_extension);
    } // if-else
  } // for i

  ne.delete_from_device();
} // gspan_cuda_lists::get_rmpath_vertex_extensions


void gspan_cuda_lists::get_rmpath_extensions(types::embedding_list_columns &embeddings,
                                             types::DFSCode code,
                                             list_extensions_vec_t &exts)
{
  types::RMPath gspan_rmpath = code.buildRMPath();
  types::RMPath cuda_rmpath = convert_rmpath(gspan_rmpath, code);

  get_rmpath_extensions(embeddings, code, cuda_rmpath, cuda_rmpath, exts);
} // gspan_cuda_lists::get_rmpath_extensions



void gspan_cuda_lists::get_rmpath_vertex_extensions(types::embedding_list_columns &embeddings,
                                                    types::DFSCode code,
                                                    list_extensions_vec_t &exts)
{
  types::RMPath gspan_rmpath = code.buildRMPath();
  types::RMPath scheduled_rmpath;
  types::RMPath cuda_rmpath = convert_rmpath(gspan_rmpath, code);

  scheduled_rmpath.push_back(code[gspan_rmpath.front()].to);
  DEBUG(*logger, "right-most vertex extension for vertex: " << code[gspan_rmpath.front()].to);
  get_rmpath_extensions(embeddings, code, cuda_rmpath, scheduled_rmpath, exts);
} // gspan_cuda_lists::get_rmpath_vertex_extensions




void gspan_cuda_lists::filter_non_minimal_extensions(types::DFSCode code, list_extensions_vec_t &extensions)
{
  list_extensions_vec_t tmp_exts;
  for(int i = 0; i < extensions.size(); i++) {
    types::DFSCode tmp_code = code;
    tmp_code.push_back(extensions[i].dfs_elem);
    if(tmp_code.dfs_code_is_min() == true) {
      tmp_exts.push_back(extensions[i]);
      TRACE(*logger, "checking minimality of " << tmp_code.to_string() << " ..... minimal");
    } else {
      TRACE(*logger, "checking minimality of " << tmp_code.to_string() << " ..... not minimal");
      extensions[i].device_free();
    }
  } // for i
  extensions = tmp_exts;
} // gspan_cuda_lists::filter_non_minimal_extensions

/*
   void gspan_cuda_lists::filter_extensions(types::DFSCode code, list_extensions_vec_t &extensions, list_extensions_vec_t &result, int ext_start, types::DFS last_edge)
   {
   types::RMPath rmpath = code.buildRMPath();
   int max_vid = last_edge.to;//code[rmpath[0]].to;
   int minlabel = last_edge.fromlabel;
   for(int i = ext_start; i < extensions.size(); i++) {
    if(extensions[i].is_forward()) {
      if(extensions[i].dfs_elem.from = max_vid) {
        // pure forward extension
        if(!(minlabel > extensions[i].dfs_elem.tolabel)) {
          result.push_back(extensions[i]);
        }
      } else {
        int dfs_code_idx = 0;
        for(int j = 0; j < rmpath.size(); j++) {
          if(code[rmpath[j]].from == extensions[i].dfs_elem.from) {
            dfs_code_idx = rmpath[j];
            break;
          }
        }

        // extension from rmpath
        if(!(minlabel > extensions[i].dfs_elem.tolabel)) {
          if((code[dfs_code_idx].elabel < extensions[i].dfs_elem.elabel) ||
             (code[dfs_code_idx].elabel == extensions[i].dfs_elem.elabel && code[dfs_code_idx].tolabel <= extensions[i].dfs_elem.tolabel)) {
            result.push_back(extensions[i]);
          }
        }
      }
    } else {
      // backward extensions are already filtered against the labels
      // of the dfs code from the get_all_extensions
    }
   } // for i

   extensions = result;
   }
 */


bool gspan_cuda_lists::possible_extension(types::DFSCode code, types::RMPath rmpath, types::DFS dfs_to_test)
{
  if(dfs_to_test.is_backward()) return true;

  int max_vid = code[rmpath[0]].to;
  int minlabel = code[0].fromlabel;


  TRACE5(*logger, "dfscode: " << code.to_string());
  TRACE5(*logger, "testing dfs_to_test: " << dfs_to_test.to_string());

  if((minlabel > dfs_to_test.tolabel)) {
    TRACE5(*logger, "returning false; minlabel: " << minlabel << "; dfs_to_test.tolabel: " << dfs_to_test.tolabel);
    return false;
  }
  int dfs_code_idx = -1;
  types::DFS dfs_code_tmp;
  for(int j = 0; j < rmpath.size(); j++) {
    if(code[rmpath[j]].from == dfs_to_test.from) {
      dfs_code_idx = rmpath[j];
      break;
    }
  }
  if(dfs_code_idx == -1) {
    TRACE5(*logger, "returning false");
    return false;
  }

  dfs_code_tmp = code[dfs_code_idx];

  TRACE5(*logger, "FORWARD dfs_to_test: " << dfs_to_test.to_string() << "; against: " << dfs_code_tmp.to_string());

  // extension from rmpath
  if((dfs_code_tmp.elabel < dfs_to_test.elabel) ||
     (dfs_code_tmp.elabel == dfs_to_test.elabel && dfs_code_tmp.tolabel <= dfs_to_test.tolabel)) {
    //result.push_back(extensions[i]);
    return true;
  } // if

  TRACE5(*logger, "returning false");
  return false;
}





void gspan_cuda_lists::create_new_extensions(list_extensions_vec_t &extensions,
                                             int start_idx,
                                             list_extensions_vec_t &new_extensions,
                                             embedding_extension_t &d_filtered_last_col,
                                             const types::embedding_list_columns &embeddings,
                                             const types::DFSCode &code,
                                             const types::RMPath &rmpath)
{
  // create new extensions using the intersection functions
  types::DFSCode new_dfscode = code;
  new_dfscode.push_back(extensions[start_idx].dfs_elem);
  types::RMPath new_rmpath = new_dfscode.buildRMPath();

  TRACE5(*logger, "code: " << code.to_string());
  TRACE5(*logger, "extensions to process: ");
  for(int i = 0; i < extensions.size(); i++) {
    TRACE5(*logger, "extension: " << extensions[i].to_string());
  }

  TRACE(*logger, "1st extension: " << extensions[start_idx].to_string());
  for(int new_ext_idx = start_idx; new_ext_idx < extensions.size(); new_ext_idx++) {
    // we do not want to do the self-join on backward edges !
    if(start_idx == new_ext_idx && extensions[new_ext_idx].is_backward()) continue;
    embedding_extension_t result(false);
    // !!!! using the opposite if ordering of if !!!!
    if(extensions[new_ext_idx].ext_type == FRWD) {
      if(extensions[start_idx].ext_type == FRWD) {

        TRACE(*logger, "2nd extension: " << extensions[new_ext_idx].to_string());

        if(get_graph_id_list_intersection_size(extensions[start_idx],extensions[new_ext_idx],d_cuda_graph_database.max_graph_vertex_count) < minimal_support) {
          TRACE(*logger, "not frequent: pre-filtered by graph_id_list intersection");
          continue;
        }

        if(possible_extension(new_dfscode, new_rmpath, extensions[new_ext_idx].dfs_elem) == false) {
          TRACE(*logger, "PRUNED: " << extensions[new_ext_idx].dfs_elem.to_string() << " because of the label conditions");
          continue;
        }
        TRACE(*logger, "2nd extension: " << extensions[new_ext_idx].dfs_elem.to_string());

        // FRWD-FRWD intersection
        intersection_fwd_fwd(extensions[start_idx], extensions[new_ext_idx], result, exec_config);
        result.dfs_elem = extensions[new_ext_idx].dfs_elem;
        int last_dfs_elem = rmpath.front();
        result.dfs_elem.to = code[last_dfs_elem].to + 2;
        result.ext_type = FRWD;
        TRACE(*logger, "computing support for 2nd extension");
        compute_support_for_fwd_ext(result, d_cuda_graph_database.max_graph_vertex_count);
        if(result.support < minimal_support) {
          TRACE(*logger, "not frequent: " << result.to_string());
          result.device_free();
          continue;
        }
        TRACE(*logger, "intersection_fwd_fwd: " << extensions[start_idx].to_string()
              << " \\cap " << extensions[new_ext_idx].to_string()
              << " -> " << result.to_string());
      } else {
        //BKWD-FRWD intersection
        TRACE(*logger, "2nd extension: " << extensions[new_ext_idx].to_string());
        intersection_bwd_fwd(embeddings, d_cuda_graph_database, extensions[start_idx], extensions[new_ext_idx], d_filtered_last_col, result, exec_config);
        TRACE(*logger, "intersection_bwd_fwd: " << extensions[start_idx].to_string()
              << " \\cap " << extensions[new_ext_idx].to_string()
              << " -> " << result.to_string());
        if(result.support < minimal_support) {
          TRACE(*logger, "not frequent: " << result.to_string());
          result.device_free();
          continue;
        }
      } // if-else
      new_extensions.push_back(result);
    } // if
  } // for new_ext_idx
} // gspan_cuda_lists::create_new_extensions


void gspan_cuda_lists::mainloop(types::embedding_list_columns &embeddings,
                                types::DFSCode &code,
                                dfs_extension_element_set_t backward_edges,
                                list_extensions_vec_t extensions)
{
  TRACE(*logger, "=====================================================================================================");
  TRACE(*logger, "=====================================================================================================");
  DEBUG(*logger, "===============  mainloop for: " << code.to_string());
  DEBUG(*logger, "extensions.size(): " << extensions.size());
  const types::RMPath &rmpath = code.buildRMPath();


  list_extensions_vec_t new_extensions;
  if(code.size() > 1) {
    TRACE(*logger, "scanning DB for rm vertex only ...................");
    list_extensions_vec_t tmp;
    get_rmpath_vertex_extensions(embeddings, code, new_extensions);
    for(int i = 0; i < new_extensions.size(); i++) {
      int found = false;
      for(int j = 0; j < extensions.size(); j++) {
        if(new_extensions[i].dfs_elem == extensions[j].dfs_elem) {
          found = true;
          new_extensions[i].device_free();
          break;
        }
      }
      if(found == false) {
        tmp.push_back(new_extensions[i]);
      }
    } // for i

    new_extensions = tmp;
  } else {
    TRACE(*logger, "initializing extensions for DFS code of length 1: whole DB scan");
    get_rmpath_extensions(embeddings, code, new_extensions);
  }

  extensions.insert(extensions.begin(), new_extensions.begin(), new_extensions.end());
  sort(extensions.begin(), extensions.end(), embedding_extension_compare_less_then_t());


  TRACE(*logger, "extension count: " << extensions.size());
  for(int i = 0; i < extensions.size(); i++) {
    TRACE(*logger, "extensions:  " << extensions[i].to_string());
  }

  //filter_non_minimal_extensions(code, extensions);

  std::stringstream ss;
  for(int i = 0; i < extensions.size(); i++) {
    ss << extensions[i].to_string() << "; ";
  }
  DEBUG(*logger, "minimal extensions, size " << extensions.size() << ": " << ss.str());


  if(execute_tests) {
    test_extensions(extensions, embeddings, code);
  }


  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // looping over the extensions
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  TRACE(*logger, "running the gspan for all frequent and minimal extensions");
  for(int ext_idx = 0; ext_idx < extensions.size(); ext_idx++) {
    // check whether the new extension is frequent
    if(extensions[ext_idx].support < minimal_support) {
      TRACE(*logger, "NOT FREQUENT extension: " << extensions[ext_idx].to_string() << "; support: " << extensions[ext_idx].support);
      continue;
    }

    if(backward_edges.find(extensions[ext_idx].dfs_elem) != backward_edges.end()) {
      DEBUG(*logger, "skipping already found backward edge");
      continue;
    }

    // check whether the new extension is minimal
    types::DFSCode new_code = code;
    new_code.push_back(extensions[ext_idx].dfs_elem);
    if(new_code.dfs_code_is_min() == false) {
      TRACE(*logger, "NOT MINIMAL extension: " << extensions[ext_idx].to_string());
      continue;
    }

    if(output) {
      output->output_graph(new_code, extensions[ext_idx].support);
    }

    TRACE(*logger, "PROCESSING extension #" << ext_idx << "; dfs extension: " << extensions[ext_idx].to_string());

    /////////////////////////////////////////////////////////////////////
    // WE SHOULD DO ALL THE INTERSECTIONS AFTER WE DO THE CHEAP CHECKS
    /////////////////////////////////////////////////////////////////////
    list_extensions_vec_t new_exts;
    embedding_extension_t d_filtered_last_col(false);
    // create new extensions using the intersection functions
    create_new_extensions(extensions, ext_idx, new_exts, d_filtered_last_col, embeddings, code, rmpath);


    types::embedding_list_columns d_new_embeddings(false);
    dfs_extension_element_set_t new_backward_edges(backward_edges);

    d_new_embeddings = embeddings.d_get_half_copy();
    if(extensions[ext_idx].ext_type == FRWD) {
      TRACE(*logger, "extending using FWD column: " << extensions[ext_idx].to_string());
      d_new_embeddings.d_extend_by_one_column(extensions[ext_idx].dfs_elem, extensions[ext_idx].embedding_column, extensions[ext_idx].col_length);
    } else {
      TRACE(*logger, "extending using BWD column: " << d_filtered_last_col.to_string());
      TRACE(*logger, "ptr: " << d_filtered_last_col.embedding_column);
      if(d_filtered_last_col.embedding_column == 0) {
        TRACE(*logger, "filter_backward_embeddings " << extensions[ext_idx].to_string());
        filter_backward_embeddings(embeddings, extensions[ext_idx], exec_config);
        TRACE(*logger, "filter_backward_embeddings " << extensions[ext_idx].to_string());
        d_new_embeddings.d_replace_last_column(extensions[ext_idx].embedding_column, extensions[ext_idx].col_length);
      } else {
        d_new_embeddings.d_replace_last_column(d_filtered_last_col.embedding_column, d_filtered_last_col.col_length);
      } // if-else

      // we are processing backward edge, add the backward edge to
      // list of already processed backward edges.
      new_backward_edges.insert(extensions[ext_idx].dfs_elem);
    } // if-else


    types::embedding_element *d_new_embed_vertices = 0;

    mainloop(d_new_embeddings, new_code, new_backward_edges, new_exts);


    d_new_embeddings.d_half_deallocate();
    CUDAFREE(d_new_embed_vertices, *logger);
    d_filtered_last_col.device_free();
  } // for i


  for(int i = 0; i < extensions.size(); i++) {
    extensions[i].device_free();
  }

  TRACE(*logger, "========================================================================");
  DEBUG(*logger, "================= exiting mainloop, for dfs code: " << code.to_string());
  TRACE(*logger, "========================================================================");
} // gspan_cuda_lists::mainloop




void gspan_cuda_lists::test_extensions(list_extensions_vec_t extensions_to_test, types::embedding_list_columns &embeddings, types::DFSCode code)
{
  list_extensions_vec_t correct_extensions;

  get_rmpath_extensions(embeddings, code, correct_extensions);
  //INFO(*logger, "sorting correct extensions");
  //filter_non_minimal_extensions(code, correct_extensions);
  sort(correct_extensions.begin(), correct_extensions.end(), embedding_extension_compare_less_then_t());
  sort(extensions_to_test.begin(), extensions_to_test.end(), embedding_extension_compare_less_then_t());


  if(correct_extensions.size() != extensions_to_test.size()) {
    CRITICAL_ERROR(*logger, "dfs code: " << code.to_string());
    CRITICAL_ERROR(*logger, "correct_extensions.size(): " << correct_extensions.size() << "; extensions_to_test.size(): " << extensions_to_test.size());
    for(int i = 0; i < correct_extensions.size(); i++) {
      ERROR(*logger, "correct_extensions[" << i << "]: " << correct_extensions[i].to_string());
    }

    for(int i = 0; i < extensions_to_test.size(); i++) {
      ERROR(*logger, "extensions_to_test[" << i << "]: " << extensions_to_test[i].to_string());
    }

    throw std::runtime_error("Sizes of correct_extensions and extensions_to_test differ !");
  }


  std::sort(extensions_to_test.begin(), extensions_to_test.end(), embedding_extension_compare_less_then_t());
  std::sort(correct_extensions.begin(), correct_extensions.end(), embedding_extension_compare_less_then_t());


  for(int i = 0; i < extensions_to_test.size(); i++) {
    bool tested = false;

    DEBUG(*logger, "testing " << extensions_to_test[i].str_ext_type() << "; against: " << correct_extensions[i].str_ext_type());
    DEBUG(*logger, "testing " << extensions_to_test[i].to_string() << "; against: " << correct_extensions[i].to_string());

    if(extensions_to_test[i].dfs_elem != correct_extensions[i].dfs_elem) {
      ERROR(*logger, "dfs code: " << code.to_string());
      for(int j = 0; j < extensions_to_test.size(); j++) {
        ERROR(*logger, "extensions_to_test[" << j << "]: " << extensions_to_test[j].to_string());
      }
      for(int j = 0; j < correct_extensions.size(); j++) {
        ERROR(*logger, "correct_extensions[" << j << "]: " << correct_extensions[j].to_string());
      }

      CRITICAL_ERROR(*logger, "DFS elements are not the same !");
      throw std::runtime_error("DFS elements are not the same !");
    }

    if(correct_extensions[i].is_forward() && extensions_to_test[i].is_forward()) {
      DEBUG(*logger, "testing forward extensions");
      tested = true;
      if(correct_extensions[i].col_length != extensions_to_test[i].col_length) {
        CRITICAL_ERROR(*logger, "correct_extensions[i].col_length: " << correct_extensions[i].col_length
                       << "; extensions_to_test[i].col_length: " << extensions_to_test[i].col_length);
        ERROR(*logger, "correct_extensions: " << print_d_array(correct_extensions[i].embedding_column, correct_extensions[i].col_length));
        ERROR(*logger, "extensions_to_test: " << print_d_array(extensions_to_test[i].embedding_column, extensions_to_test[i].col_length));
        throw std::runtime_error("lengths of embedding differs !");
      } // if


      types::embedding_element *correct_column = 0;
      types::embedding_element *test_column = 0;
      copy_d_array_to_h(correct_extensions[i].embedding_column, correct_extensions[i].col_length, correct_column);
      copy_d_array_to_h(extensions_to_test[i].embedding_column, extensions_to_test[i].col_length, test_column);
      bool same = std::equal(correct_column, correct_column + correct_extensions[i].col_length, test_column);
      if(same == false) {
        std::pair<types::embedding_element*, types::embedding_element*> diff;
        diff = std::mismatch(correct_column, correct_column + correct_extensions[i].col_length, test_column);

        CRITICAL_ERROR(*logger, "diference (position) correct: " << (diff.first - correct_column) << "; test: " << diff.second - test_column);
        CRITICAL_ERROR(*logger, "correct array: " << print_h_array(correct_column, correct_extensions[i].col_length));
        CRITICAL_ERROR(*logger, "test array: " << print_h_array(test_column, extensions_to_test[i].col_length));
        throw std::runtime_error("embeddings are not the same !");
      }
      delete [] correct_column;
      delete [] test_column;
    } // if

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if(correct_extensions[i].is_backward() && extensions_to_test[i].is_backward()) {
      tested = true;
      DEBUG(*logger, "testing backward extensions");

      if(correct_extensions[i].filtered_emb_offsets_length != extensions_to_test[i].filtered_emb_offsets_length) {
        CRITICAL_ERROR(*logger, "correct_extensions[i].filtered_emb_offsets_length: " << correct_extensions[i].filtered_emb_offsets_length
                       << "; extensions_to_test[i].filtered_emb_offsets_length: " << extensions_to_test[i].filtered_emb_offsets_length);
        throw std::runtime_error("lengths of embedding differs !");
      } // if


      int *correct_offsets = 0;
      int *test_offsets = 0;
      copy_d_array_to_h(correct_extensions[i].filtered_emb_offsets, correct_extensions[i].filtered_emb_offsets_length, correct_offsets);
      copy_d_array_to_h(extensions_to_test[i].filtered_emb_offsets, extensions_to_test[i].filtered_emb_offsets_length, test_offsets);
      bool same = std::equal(correct_offsets, correct_offsets + correct_extensions[i].filtered_emb_offsets_length, test_offsets);
      if(same == false) {
        std::pair<int*, int*> diff;
        diff = std::mismatch(correct_offsets, correct_offsets + correct_extensions[i].col_length, test_offsets);

        CRITICAL_ERROR(*logger, "diference (position) correct: " << (diff.first - correct_offsets) << "; test: " << diff.second - test_offsets);
        CRITICAL_ERROR(*logger, "correct array: " << print_h_array(correct_offsets, correct_extensions[i].filtered_emb_offsets_length));
        CRITICAL_ERROR(*logger, "test array: " << print_h_array(test_offsets, extensions_to_test[i].filtered_emb_offsets_length));
        throw std::runtime_error("Offsets are not the same !");
      }
      delete [] correct_offsets;
      delete [] test_offsets;
    } // if

    assert(tested);
  } // for i


  for(int i = 0; i < correct_extensions.size(); i++) {
    correct_extensions[i].device_free();
  } // for i

} // gspan_cuda_lists::test_extensions

} // namespace gspan_cuda

