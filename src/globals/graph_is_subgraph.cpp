#include <graph_types.hpp>
#include <logger.hpp>

namespace types {

static Logger *grph_log = Logger::get_logger("GRPH");

/**
 * tests whether other_graph_dfs is subgraph of this.
 */
bool Graph::is_subgraph(DFSCode &other_graph_dfs) const
{
  /*
     types::EdgeList edges;
     Projected_map3 root;

     static DFS_partial_equal dfs_eq;
     static DFS_partial_not_equal dfs_neq;

     // TODO: filter the possibilities of depth-first search branching by other_graph_dfs[0].
     for(unsigned int from = 0; from < this->size() ; ++from) {
     if(get_forward_root(*this, (*this)[from], edges)) {
      for(types::EdgeList::iterator it = edges.begin(); it != edges.end();  ++it) {
        DFS tmp_dfs(0, 1, (*this)[from].label, (*it)->elabel, (*this)[(*it)->to].label);
        if(dfs_eq(tmp_dfs, other_graph_dfs[0])) {
          root[(*this)[from].label][(*it)->elabel][(*this)[(*it)->to].label].push(0, *it, 0);
        } // if(found appropriate starting DFS element)
      } // for it
     } // if
     } // for from


     for(Projected_iterator3 fromlabel = root.begin();
      fromlabel != root.end(); ++fromlabel) {
     for(Projected_iterator2 elabel = fromlabel->second.begin() ;
        elabel != fromlabel->second.end() ; ++elabel) {
      for(Projected_iterator1 tolabel = elabel->second.begin();
          tolabel != elabel->second.end(); ++tolabel) {
        // Build the initial two-node graph.  It will be grownrecursively within project.
        // TODO: actually,  DFS_CODE[i] should be the same as other_graph_dfs[i]
        //       so why I do not use index to other_graph_dfs instead of building a new (parallel) DFS_CODE.
        //DFS_CODE.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
        int depth = 1;
        if(is_subgraph_internal(tolabel->second, other_graph_dfs, depth) == true) return true;
        //DFS_CODE.pop();
      } // for tolabel
     } // for elabel
     } // for fromlabel

   */


  TRACE5(*grph_log, "this graph min dfs code: " << get_min_dfs_code().to_string());
  TRACE5(*grph_log, "other_graph_dfs: " << other_graph_dfs.to_string());

  //DFSCode min_dfs_code;


  //static DFS_partial_equal dfs_eq;
  static DFS_partial_not_equal dfs_neq;

  Projected_map3 root;
  types::EdgeList edges;

  for(unsigned int from = 0; from < this->size(); ++from) {
    if(get_forward_root(*this, (*this)[from], edges)) {
      for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
        DFS tmp_dfs(0, 1, (*this)[from].label, (*it)->elabel, -(*this)[(*it)->to].label);
        if(dfs_neq(tmp_dfs, other_graph_dfs[0])) continue;
        TRACE5(*grph_log, "tmp_dfs: " << tmp_dfs.to_string() << "; other_graph_dfs[0]: " << other_graph_dfs[0].to_string());
        root[(*this)[from].label][(*it)->elabel][(*this)[(*it)->to].label].push(0, *it, 0);
      } // for it
    } // if get_forward_root
  } // for from

  Projected_iterator3 fromlabel = root.begin();
  Projected_iterator2 elabel = fromlabel->second.begin();
  Projected_iterator1 tolabel = elabel->second.begin();

  //min_dfs_code.push(0, 1, fromlabel->first, elabel->first, tolabel->first);

  TRACE5(*grph_log, "#######################################################################################");
  //TRACE4(*grph_log, "computing min dfs code");

  return is_subgraph_internal(tolabel->second, other_graph_dfs, 1);
  //(tolabel->second, dfs_code, min_dfs_code, graph_dfs_code);

  //return false;
} // Graph::is_subgraph


bool Graph::is_subgraph_internal(Projected &projected, DFSCode &other_graph_dfs, int depth) const
{
  if(other_graph_dfs.size() == depth) {
    TRACE5(*grph_log, "other_graph_dfs IS ISOMORPHIC with this.");
    return true;
  }

  const RMPath &rmpath = other_graph_dfs.buildRMPath();
  int minlabel         = other_graph_dfs[0].fromlabel;
  int maxtoc           = other_graph_dfs[rmpath[rmpath.size() - depth]].to;

  static DFS_partial_equal dfs_eq;
  static DFS_partial_not_equal dfs_neq;

  TRACE5(*grph_log, "#########################################################################################");


  TRACE5(*grph_log, "other_graph_dfs: " << other_graph_dfs.to_string());
  TRACE5(*grph_log, "rmpath.size(): " << rmpath.size());
  TRACE5(*grph_log, "depth: " << depth);
  TRACE5(*grph_log, "projected: " << projected.to_string());
  TRACE5(*grph_log, "maxtoc: " << maxtoc);
  TRACE5(*grph_log, "minlabel: " << minlabel);

  //////////////////////////////////////////////////////////////////////////////////////////////
  // TODO: filter the possibilities of depth-first search branching by other_graph_dfs[depth].
  //////////////////////////////////////////////////////////////////////////////////////////////

  // SUBBLOCK 1
  {
    Projected_map1 root;
    bool flg = false;

    // iterate over all backward edges that are connected to the right-most path
    // first found => proceed further in the dfs search for the minimum code.
    int rmpath_last_idx = std::max(1, int(rmpath.size()) - depth + 1); //  + 1
    TRACE5(*grph_log, "get_backward, rmpath.size()-1: " << (rmpath.size() - 1) << "; rmpath_last_idx: " << rmpath_last_idx);
    for(int i = rmpath.size() - 1; !flg  && i >= rmpath_last_idx; --i) {
      TRACE5(*grph_log,  "i: " << i << "; projected.size(): " << projected.size());
      for(unsigned int n = 0; n < projected.size(); ++n) {
        //std::cout << "iterating over n: " << n << std::endl;
        PDFS *cur = &projected[n];
        History history(*this, cur);
        //std::cout << "history: " << history.to_string() << std::endl;
        TRACE5(*grph_log, "rmpath[" << i << "]: " << rmpath[i] << "; rmpath[" << (rmpath_last_idx - 1) << "]: " << rmpath[rmpath_last_idx - 1]);
        TRACE5(*grph_log, "history[rmpath[i]]: " << history[rmpath[i]]->to_string() << "; history[rmpath[rmpath_last_idx+1]]: " << history[rmpath[rmpath_last_idx - 1]]->to_string());
        TRACE5(*grph_log, "edges equal ? " << (history[rmpath[i]] == history[rmpath[rmpath_last_idx - 1]]));
        Edge *e = get_backward(*this, history[rmpath[i]], history[rmpath[rmpath_last_idx - 1]], history);
        if(e == 0) continue;

        DFS tmp_dfs(maxtoc, other_graph_dfs[rmpath[i]].from, -1, e->elabel, -1);
        TRACE5(*grph_log, "get_backward comparing: " << tmp_dfs.to_string() << "; with: " << other_graph_dfs[depth].to_string() << "; result: " << dfs_eq(tmp_dfs, other_graph_dfs[depth]));
        if(dfs_neq(tmp_dfs, other_graph_dfs[depth])) continue;

        root[e->elabel].push(0, e, cur);
        flg = true;
      } // for n
    } // for i

    // if we have found at least one backward edge, get the smallest and append its DFS to the min_dfs_code
    if(flg) {
      Projected_iterator1 elabel = root.begin();
      // TODO: is this correct ?
      return is_subgraph_internal(elabel->second, other_graph_dfs, depth + 1);
    } // if(flg)
  } // SUBBLOCK 1

  // SUBBLOCK 2
  {
    bool flg = false;
    int newfrom = 0;
    Projected_map2 root;
    types::EdgeList edges;

    TRACE5(*grph_log, "pure forward edges");
    // collect all forward edges
    // edges that extends the right-most node
    for(unsigned int n = 0; n < projected.size(); ++n) {
      PDFS *cur = &projected[n];
      History history(*this, cur);
      TRACE5(*grph_log, "get_forward_pure, history: " << history.to_string());
      TRACE5(*grph_log, "rmpath.size()-depth: " << (rmpath.size() - depth));
      TRACE5(*grph_log, "history[rmpath[rmpath.size()-depth]]: " << history[rmpath[rmpath.size() - depth]]->to_string());
      if(get_forward_pure(*this, history[rmpath[rmpath.size() - depth]], minlabel, history, edges)) {
        newfrom = maxtoc;
        for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
          DFS tmp_dfs(newfrom, maxtoc + 1, -1, (*it)->elabel, (*this)[(*it)->to].label);
          TRACE5(*grph_log, "get_forward_pure comparing: " << tmp_dfs.to_string() << "; with: " << other_graph_dfs[depth] << "; result: " << dfs_eq(tmp_dfs, other_graph_dfs[depth]));
          if(dfs_neq(tmp_dfs, other_graph_dfs[depth])) continue;
          flg = true;
          TRACE5(*grph_log, "match: " << tmp_dfs);
          root[(*it)->elabel][(*this)[(*it)->to].label].push(0, *it, cur);
        } // for it
      } // if get_forward_pure
    } // for n


    TRACE5(*grph_log, "rmpath forward edges");
    // edges that extends a node on the right-most path except the right-most node
    for(int i = 0; !flg && i < depth; ++i) {
      TRACE5(*grph_log, "i: " << i);
      for(unsigned int n = 0; n < projected.size(); ++n) {
        PDFS *cur = &projected[n];
        History history(*this, cur);
        int rmpath_idx = rmpath.size() - 1 - i;
        if(get_forward_rmpath(*this, history[rmpath[rmpath_idx]], minlabel, history, edges)) {
          newfrom = other_graph_dfs[rmpath[rmpath_idx]].from;
          for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
            DFS tmp_dfs(newfrom, maxtoc + 1, -1, (*it)->elabel, (*this)[(*it)->to].label);
            TRACE5(*grph_log, "get_forward_rmpath comparing: " << tmp_dfs.to_string() << "; with: " << other_graph_dfs[depth] << "; result: " << dfs_eq(tmp_dfs, other_graph_dfs[depth]));
            if(dfs_neq(tmp_dfs, other_graph_dfs[rmpath[depth]])) continue;
            flg = true;
            root[(*it)->elabel][(*this)[(*it)->to].label].push(0, *it, cur);
          }
        } // if get_forward_rmpath
      } // for n
    } // for i

    if(flg) {
      Projected_iterator2 elabel  = root.begin();
      Projected_iterator1 tolabel = elabel->second.begin();
      // TODO: is this correct ?
      return is_subgraph_internal(tolabel->second, other_graph_dfs, depth + 1);
    } // if(flg)
  } // SUBBLOCK 2

  TRACE5(*grph_log, "============= NON-ISOMORPHIC =============");

  return false;
}

} // namespace types



