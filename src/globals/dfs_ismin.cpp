#include <graph_types.hpp>
#include <logger.hpp>


using types::DFSCode;
using types::RMPath;
using types::Graph;
using types::Projected;
using types::History;

using types::Projected_map3;
using types::Projected_map2;
using types::Projected_map1;
using types::Projected_iterator3;
using types::Projected_iterator2;
using types::Projected_iterator1;


static Logger *dfs_log = Logger::get_logger("DFS");

namespace types {

bool DFSCode::dfs_code_is_min() const
{
  const DFSCode &dfs_code = *this;

  if(dfs_code.size() == 1) {
    return (true);
  }

  Graph graph_dfs_code;
  DFSCode min_dfs_code;

  dfs_code.toGraph(graph_dfs_code);
  //graph_dfs_code.write(cout);
  //min_dfs_code.clear();

  Projected_map3 root;
  types::EdgeList edges;

  for(unsigned int from = 0; from < graph_dfs_code.size(); ++from) {
    if(get_forward_root(graph_dfs_code, graph_dfs_code[from], edges)) {
      for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
        root[graph_dfs_code[from].label][(*it)->elabel][graph_dfs_code[(*it)->to].label].push(0, *it, 0);
      } // for it
    } // if get_forward_root
  } // for from

  Projected_iterator3 fromlabel = root.begin();
  Projected_iterator2 elabel = fromlabel->second.begin();
  Projected_iterator1 tolabel = elabel->second.begin();

  min_dfs_code.push(0, 1, fromlabel->first, elabel->first, tolabel->first);

  TRACE4(*dfs_log, "=======================================================");
  TRACE4(*dfs_log, "computing min dfs code");

  return dfs_code_is_min_internal(tolabel->second, dfs_code, min_dfs_code, graph_dfs_code);
} // dfs_code_is_min


bool DFSCode::dfs_code_is_min_internal(Projected &projected, const DFSCode &dfs_code, DFSCode &min_dfs_code, Graph &graph_dfs_code)
{
  const RMPath &rmpath = min_dfs_code.buildRMPath();
  int minlabel         = min_dfs_code[0].fromlabel;
  int maxtoc           = min_dfs_code[rmpath[0]].to;


  TRACE5(*dfs_log, "--------------------------------------------------------------------------------");
  TRACE5(*dfs_log, "projected: " << projected.to_string());
  TRACE5(*dfs_log, "dfs_code: " << dfs_code.to_string());
  TRACE5(*dfs_log, "min_dfs_code: " << min_dfs_code.to_string());


  static DFS_partial_not_equal dfs_neq;

  // SUBBLOCK 1
  {
    TRACE5(*dfs_log, "backward edges ... ");
    Projected_map1 root;
    bool flg = false;
    int newto = 0;

    for(int i = rmpath.size() - 1; !flg  && i >= 1; --i) {
      for(unsigned int n = 0; n < projected.size(); ++n) {
        PDFS *cur = &projected[n];
        History history(graph_dfs_code, cur);
        Edge *e = get_backward(graph_dfs_code, history[rmpath[i]], history[rmpath[0]], history);
        if(e) {
          TRACE5(*dfs_log, "selected edge: " << e->to_string());
          root[e->elabel].push(0, e, cur);
          newto = min_dfs_code[rmpath[i]].from;
          flg = true;
        } // if e
      } // for n
    } // for i

    if(flg) {
      Projected_iterator1 elabel = root.begin();
      min_dfs_code.push(maxtoc, newto, -1, elabel->first, -1);
      if(dfs_neq(dfs_code[min_dfs_code.size() - 1], min_dfs_code[min_dfs_code.size() - 1])) {
        TRACE5(*dfs_log, "returning FALSE, dfs_code: " << dfs_code.to_string() << "; min_dfs_code: " << min_dfs_code.to_string());
        return false;
      }
      return dfs_code_is_min_internal(elabel->second, dfs_code, min_dfs_code, graph_dfs_code);
    }
  } // SUBBLOCK 1

  // SUBBLOCK 2
  {
    TRACE5(*dfs_log, "forward edges ... ");
    bool flg = false;
    int newfrom = 0;
    Projected_map2 root;
    types::EdgeList edges;

    for(unsigned int n = 0; n < projected.size(); ++n) {
      PDFS *cur = &projected[n];
      History history(graph_dfs_code, cur);
      TRACE5(*dfs_log, "history[rmpath[0]]: " << history[rmpath[0]]->to_string());
      if(get_forward_pure(graph_dfs_code, history[rmpath[0]], minlabel, history, edges)) {
        flg = true;
        newfrom = maxtoc;
        for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
          TRACE5(*dfs_log, "selected edge(frwrd pure): " << (*it)->to_string());
          root[(*it)->elabel][graph_dfs_code[(*it)->to].label].push(0, *it, cur);
        }
      } // if get_forward_pure
    } // for n

    for(int i = 0; !flg && i < (int)rmpath.size(); ++i) {
      for(unsigned int n = 0; n < projected.size(); ++n) {
        PDFS *cur = &projected[n];
        History history(graph_dfs_code, cur);
        if(get_forward_rmpath(graph_dfs_code, history[rmpath[i]], minlabel, history, edges)) {
          flg = true;
          newfrom = min_dfs_code[rmpath[i]].from;
          for(types::EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
            TRACE5(*dfs_log, "selected edge(frwrd rmpa): " << (*it)->to_string());
            root[(*it)->elabel][graph_dfs_code[(*it)->to].label].push(0, *it, cur);
          }
        } // if get_forward_rmpath
      } // for n
    } // for i

    if(flg) {
      Projected_iterator2 elabel  = root.begin();
      Projected_iterator1 tolabel = elabel->second.begin();
      min_dfs_code.push(newfrom, maxtoc + 1, -1, elabel->first, tolabel->first);
      if(dfs_neq(dfs_code[min_dfs_code.size() - 1], min_dfs_code[min_dfs_code.size() - 1])) {
        TRACE5(*dfs_log, "returning FALSE, dfs_code: " << dfs_code.to_string() << "; min_dfs_code: " << min_dfs_code.to_string());
        return false;
      }
      return dfs_code_is_min_internal(tolabel->second, dfs_code, min_dfs_code, graph_dfs_code);
    } // if(flg)
  } // SUBBLOCK 2

  TRACE5(*dfs_log, "returning TRUE, dfs_code: " << dfs_code.to_string() << "; min_dfs_code: " << min_dfs_code.to_string());
  return true;
} // dfs_code_is_min_internal


} // namespace types

