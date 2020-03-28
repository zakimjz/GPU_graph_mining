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

bool gspan_cuda::test_supports2(types::Projected &projected,
                                types::DFSCode code,
                                types::DFS *h_dfs_elem,
                                int *h_support,
                                int size,
                                extension_element_t *d_exts_result,
                                int exts_result_length)
{
  gspan_cuda::dfs_extension_element_map_t all_dfs_elements_projected;
  TRACE(*logger, "projected.size(): " << projected.size());
  get_all_extensions_orig(projected, code, all_dfs_elements_projected);

  TRACE(*logger, "all_dfs_elements_projected.size(): " << all_dfs_elements_projected.size());
  for(gspan_cuda::dfs_extension_element_map_t::iterator it = all_dfs_elements_projected.begin(); it != all_dfs_elements_projected.end(); it++) {
    DEBUG(*logger, "host extensions: " << it->first.to_string() << "; support: " << it->second);
  }
  bool error_found = false;

  for(int i = 0; i < size; i++) {
    if(all_dfs_elements_projected.find(h_dfs_elem[i]) == all_dfs_elements_projected.end()) {
      CRITICAL_ERROR(*logger, "element: " << h_dfs_elem[i].to_string() << " is missing in all_dfs_elements_projected.");
      error_found = true;
    } // if
  } // for i


  for(gspan_cuda::dfs_extension_element_map_t::iterator it = all_dfs_elements_projected.begin(); it != all_dfs_elements_projected.end(); it++) {
    bool found = false;
    int i = 0;
    for(i = 0; i < size; i++) {
      if(it->first == h_dfs_elem[i]) {
        found = true;
        break;
      }
    } // for i

    if(found == false) {
      CRITICAL_ERROR(*logger, "dfs: " << it->first.to_string() << "; support: " << it->second << "; is missing in h_dfs_elem.");
      error_found = true;
    } else {
      if(h_support[i] != it->second) {
        error_found = true;
        CRITICAL_ERROR(*logger, "found dfs: " << it->first.to_string() << "; support: " << it->second
                       << "; cuda computed_support: " << h_support[i] << "; is missing in h_dfs_elem.");
      }
    }
  } // for it

  if(error_found == true) {
    DEBUG(*logger, "==================================================");
    DEBUG(*logger, "printing h_dfs_elem array: ");
    for(int i = 0; i < size; i++) {
      DEBUG(*logger, h_dfs_elem[i].to_string() << "; support: " << h_support[i]);
    }
  }
  return error_found;
}

void gspan_cuda::get_all_extensions_orig(types::Projected &projected, types::DFSCode DFS_CODE, gspan_cuda::dfs_extension_element_map_t &all_dfs_elements)
{
  all_dfs_elements.clear();
  TRACE(*logger, "projected.size(): " << projected.size());

  // We just outputted a frequent subgraph.  As it is frequent enough, so
  // might be its (n+1)-extension-graphs, hence we enumerate them all.
  const types::RMPath &rmpath = DFS_CODE.buildRMPath();
  int minlabel = DFS_CODE[0].fromlabel;
  int maxtoc = DFS_CODE[rmpath[0]].to;

  Projected_map3 new_fwd_root;
  Projected_map2 new_bck_root;
  types::EdgeList edges;

  // Enumerate all possible one edge extensions of the current substructure.
  for(unsigned int n = 0; n < projected.size(); ++n) {

    unsigned int id = projected[n].id;
    DEBUG(*logger, "id: " << id);
    types::PDFS *cur = &projected[n];
    types::History history(graph_database[id], cur);

    // XXX: do we have to change something here for directed edges?

    // backward
    for(int i = (int)rmpath.size() - 1; i >= 1; --i) {
      types::Edge *e = get_backward(graph_database[id], history[rmpath[i]], history[rmpath[0]], history);
      if(e) {
        DEBUG(*logger, "BCKWRD found edge, gid  " << id << ": " <<  maxtoc << " " << DFS_CODE[rmpath[i]].from << " " << graph_database[id].get_vertex_label(e->from) << " " << e->elabel << " " << graph_database[id].get_vertex_label(e->to) << "   |   from: " << e->from << "; to: " << e->to);
        new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
      }
    }

    // pure forward
    // FIXME: here we pass a too large e->to (== history[rmpath[0]]->to
    // into get_forward_pure, such that the assertion fails.
    //
    // The problem is:
    // history[rmpath[0]]->to > graph_database[id].size()
    if(get_forward_pure(graph_database[id], history[rmpath[0]], minlabel, history, edges)) {
      for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
        DEBUG(*logger, "PURE FRWRD found edge, gid  " << id << ": " << maxtoc << " " << (maxtoc + 1) << " "
              << graph_database[id].get_vertex_label((*it)->from) << " " << (*it)->elabel << " " << graph_database[id].get_vertex_label((*it)->to));
        new_fwd_root[maxtoc][(*it)->elabel][graph_database[id][(*it)->to].label].push(id, *it, cur);
        assert((*it)->to < graph_database[id].size());
      }
    }

    // backtracked forward
    for(int i = 0; i < (int)rmpath.size(); ++i) {
      DEBUG(*logger, "get_forward_rmpath for code element: " << DFS_CODE[rmpath[i]]);
      if(get_forward_rmpath(graph_database[id], history[rmpath[i]], minlabel, history, edges)) {
        for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
          DEBUG(*logger, "FRWRD found edge, gid  " << id << ": " << DFS_CODE[rmpath[i]].from << " " << (maxtoc + 1) << " "
                << graph_database[id].get_vertex_label((*it)->from) << " " << (*it)->elabel << " " << graph_database[id].get_vertex_label((*it)->to));
          new_fwd_root[DFS_CODE[rmpath[i]].from][(*it)->elabel][graph_database[id][(*it)->to].label].push(id, *it, cur);
        } // for it
      } // if
    } // for i

  } // for n

  // Test all extended substructures.
  // backward
  for(Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
    for(Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
      int id = elabel->second.front().id;
      //DFS_CODE.push(maxtoc, to->first, -1, elabel->first, -1);
      int from_grph = elabel->second.front().edge->from;
      int to_grph = elabel->second.front().edge->to;
      int from_label = graph_database[id].get_vertex_label(from_grph);
      int to_label = graph_database[id].get_vertex_label(to_grph);
      unsigned int sup = support(elabel->second);
      types::DFS new_dfs(maxtoc, to->first, from_label, elabel->first, to_label);

      DEBUG(*logger, "BCKWRD gid  " << id << "; from_pat: " <<  maxtoc << "; to_pat: " << to->first
            << "; fromlabel: " << from_label << "; elabel: " << elabel->first
            << "; tolabel: " << to_label << "; from_grph: " << from_grph << "; to_grph: " << to_grph //graph_database[id].get_vertex_label(e->to)
            << "; dfs: " << new_dfs.to_string());

      all_dfs_elements.insert(std::make_pair(new_dfs, sup));
    } // for elabel
  } // for to

  // forward
  for(Projected_riterator3 from = new_fwd_root.rbegin();
      from != new_fwd_root.rend(); ++from) {
    for(Projected_iterator2 elabel = from->second.begin();
        elabel != from->second.end(); ++elabel) {
      for(Projected_iterator1 tolabel = elabel->second.begin();
          tolabel != elabel->second.end(); ++tolabel) {
        //DFS_CODE.push(from->first, maxtoc + 1, -1, elabel->first, tolabel->first);

        int id = tolabel->second.front().id;
        int from_label = graph_database[id].get_vertex_label(tolabel->second.front().edge->from);
        int to_label = graph_database[id].get_vertex_label(tolabel->second.front().edge->to);
        assert(tolabel->first == to_label);
        unsigned int sup = support(tolabel->second);
        types::DFS new_dfs(from->first, maxtoc + 1, from_label, elabel->first, tolabel->first);

        DEBUG(*logger, "FRWRD gid: " << id << "; from_pat: " << from->first << "; to_pat: " << (maxtoc + 1)
              << "; from:" << tolabel->second.front().edge->from << "; to: " << tolabel->second.front().edge->to
              << "; from_label: " << from_label << "; elabel: " << elabel->first << "; tolabel: " << tolabel->first << "; graph id: " << tolabel->second.front().id
              << "; dfs: " << new_dfs.to_string());

        all_dfs_elements.insert(std::make_pair(new_dfs, sup));
      }
    }

  } // for from




  return;
}




void gspan_cuda::get_new_projected(types::Projected &old_projected, types::DFSCode DFS_CODE, types::DFS dfs_elem, types::Projected &new_projected)
{
  // We just outputted a frequent subgraph.  As it is frequent enough, so
  // might be its (n+1)-extension-graphs, hence we enumerate them all.
  const types::RMPath &rmpath = DFS_CODE.buildRMPath();
  int minlabel = DFS_CODE[0].fromlabel;
  int maxtoc = DFS_CODE[rmpath[0]].to;

  Projected_map3 new_fwd_root;
  Projected_map2 new_bck_root;
  types::EdgeList edges;

  DEBUG(*logger, "DFS_CODE: " << DFS_CODE.to_string() << "; new dfs elem: " << dfs_elem.to_string());

  // Enumerate all possible one edge extensions of the current substructure.
  for(unsigned int n = 0; n < old_projected.size(); ++n) {
    unsigned int id = old_projected[n].id;
    //DEBUG(*logger, "current graph id: " << id);
    types::PDFS *cur = &old_projected[n];
    types::History history(graph_database[id], cur);

    // backward
    for(int i = (int)rmpath.size() - 1; i >= 1; --i) {
      types::Edge *e = get_backward(graph_database[id], history[rmpath[i]], history[rmpath[0]], history);
      if(e) {
        new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
      }
    }

    // pure forward
    // FIXME: here we pass a too large e->to (== history[rmpath[0]]->to
    // into get_forward_pure, such that the assertion fails.
    //
    // The problem is:
    // history[rmpath[0]]->to > graph_database[id].size()
    if(get_forward_pure(graph_database[id], history[rmpath[0]], minlabel, history, edges)) {
      for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
        new_fwd_root[maxtoc][(*it)->elabel][graph_database[id][(*it)->to].label].push(id, *it, cur);
      }
    }

    // backtracked forward
    for(int i = 0; i < (int)rmpath.size(); ++i) {
      if(get_forward_rmpath(graph_database[id], history[rmpath[i]], minlabel, history, edges)) {
        for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
          new_fwd_root[DFS_CODE[rmpath[i]].from][(*it)->elabel][graph_database[id][(*it)->to].label].push(id, *it, cur);
        } // for it
      } // if
    } // for i
  } // for n
  DEBUG(*logger, "new_bck_root.size(): " << new_bck_root.size());
  DEBUG(*logger, "new_fwd_root.size(): " << new_fwd_root.size());



  // Test all extended substructures.
  // backward
  for(Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
    for(Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
      //DFS_CODE.push(maxtoc, to->first, -1, elabel->first, -1);

      int id = elabel->second.front().id;
      int from_grph = elabel->second.front().edge->from;
      int to_grph = elabel->second.front().edge->to;
      int from_label = graph_database[id].get_vertex_label(from_grph);
      int to_label = graph_database[id].get_vertex_label(to_grph);
      unsigned int sup = support(elabel->second);
      types::DFS new_dfs(maxtoc, to->first, from_label, elabel->first, to_label);

      if(new_dfs == dfs_elem)  {
        DEBUG(*logger, "found");
        new_projected = elabel->second;
        return;
      } // if

      /*
         int to_label = graph_database[elabel->second.front().id][to->first].label;
         DFS_CODE.push(maxtoc, to->first, DFS_CODE.back().tolabel, elabel->first, to_label);
         DEBUG(*logger, "backward code: " << DFS_CODE.to_string() << "; for element: " << dfs_elem.to_string());
         if(DFS_CODE.back() == dfs_elem)  {
         DEBUG(*logger, "found");
         new_projected = elabel->second;
         return;
         } // if
         DFS_CODE.pop();
       */
    } // for elabel
  } // for to

  // forward
  for(Projected_riterator3 from = new_fwd_root.rbegin();
      from != new_fwd_root.rend(); ++from) {
    for(Projected_iterator2 elabel = from->second.begin();
        elabel != from->second.end(); ++elabel) {
      for(Projected_iterator1 tolabel = elabel->second.begin();
          tolabel != elabel->second.end(); ++tolabel) {

        int id = tolabel->second.front().id;
        int from_label = graph_database[id].get_vertex_label(tolabel->second.front().edge->from);
        int to_label = graph_database[id].get_vertex_label(tolabel->second.front().edge->to);
        assert(tolabel->first == to_label);
        unsigned int sup = support(tolabel->second);
        types::DFS new_dfs(from->first, maxtoc + 1, from_label, elabel->first, tolabel->first);


        //DFS_CODE.push(from->first, maxtoc + 1, -1, elabel->first, tolabel->first);
        //DFS_CODE.push(from->first, maxtoc + 1, DFS_CODE.back().tolabel, elabel->first, tolabel->first);
        DEBUG(*logger, "forward code: " << DFS_CODE.to_string() << "; for element: " << dfs_elem.to_string() << "; projected size: " << tolabel->second.size());
        if(new_dfs == dfs_elem)  {
          DEBUG(*logger, "found");
          new_projected = tolabel->second;
          return;
        }
        //DFS_CODE.pop();
      } // for tolabel
    } // for elabel
  } // for from

  DEBUG(*logger, "NOT FOUND");
} // gspan_cuda::get_new_projected











void gspan_cuda::test_supports(types::DFSCode code, types::DFS *h_dfs_elem, int size, extension_element_t *d_exts_result, int exts_result_length)
{
  dfs_extension_element_set_t all_dfs_elements;

  extension_element_t *h_exts_result = new extension_element_t[exts_result_length];
  CUDA_EXEC(cudaMemcpy(h_exts_result, d_exts_result, sizeof(extension_element_t) * exts_result_length, cudaMemcpyDeviceToHost), *logger);

  for(int i = 0; i < exts_result_length; i++) {
    types::DFS dfs_elem(h_exts_result[i].from,
                        h_exts_result[i].to,
                        h_exts_result[i].fromlabel,
                        h_exts_result[i].elabel,
                        h_exts_result[i].tolabel);
    all_dfs_elements.insert(dfs_elem);
  } // for i

  if(all_dfs_elements.size() != size) {
    CRITICAL_ERROR(*logger, "h_dfs_elem does not contain proper number of extensions ! all_dfs_elements.size(): " << all_dfs_elements.size() << "; size: " << size);
  }

  for(int i = 0; i < size; i++) {
    if(all_dfs_elements.find(h_dfs_elem[i]) == all_dfs_elements.end()) {
      CRITICAL_ERROR(*logger, "element: " << h_dfs_elem[i].to_string() << " is missing.");
    } // if
  } // for i

  for(dfs_extension_element_set_t::iterator it = all_dfs_elements.begin(); it != all_dfs_elements.end(); it++) {
    bool found = false;
    for(int i = 0; i < size; i++) {
      if(it->from == h_dfs_elem[i].from &&
         it->to == h_dfs_elem[i].to &&
         it->fromlabel == h_dfs_elem[i].fromlabel &&
         it->elabel == h_dfs_elem[i].elabel &&
         it->tolabel == h_dfs_elem[i].tolabel)
      {
        found = true;
        break;
      }
    } // for i

    if(found == false) {
      CRITICAL_ERROR(*logger, "dfs: " << it->to_string() << " was not found in the extensions.");
    }
  } // for it

  delete [] h_exts_result;
} // gspan_cuda::test_supports



void gspan_cuda::test_embeddings(types::embedding_list_columns &d_embeddings, types::RMPath rmpath, extension_element_t *d_extensions, int extensions_length)
{
  embedding_list_columns h_embeddings(true);
  h_embeddings.copy_from_device(&d_embeddings);

  extension_element_t *h_extensions = new extension_element_t[extensions_length];
  CUDA_EXEC(cudaMemcpy(h_extensions, d_extensions, sizeof(extension_element_t) * extensions_length, cudaMemcpyDeviceToHost), *logger);

  int col_count = h_embeddings.columns_count;
  for(int i = 0; i < h_embeddings.columns_lengths[col_count - 1]; i++) {
    bool ok = check_one_embedding(graph_database, h_cuda_graph_database, rmpath, h_embeddings, i, h_extensions, extensions_length);
    if(ok == false) {
      CRITICAL_ERROR(*logger, "checking extensions of embedding #" << i);
    }
    assert(ok);
  } // for i


  h_embeddings.delete_from_host();
  delete [] h_extensions;
}



void gspan_cuda::test_database()
{
  for(int i = 0; i < graph_database.size(); i++) {
    const Graph &g = graph_database[i];
    for(int from = 0; from < g.vertex_size(); from++) {
      for(int to = 0; to < g.vertex_size(); to++) {
        if(from == to) continue;
        int from_label = g.get_vertex_label(from);
        int to_label = g.get_vertex_label(to);

        types::Edge e1;
        bool e1_found = g[from].find(from, to, e1);
        types::Edge e2;
        bool e2_found = g[to].find(to, from, e2);
        if(e1_found && e2_found && e1.elabel != e2.elabel) {
          CRITICAL_ERROR(*logger, "error: graph " << i << " contains two edges between two same vertices with different edge label");
        }
      } // for to
    } // for from
  } // for i
} // test_database


} // namespace gspan_cuda



