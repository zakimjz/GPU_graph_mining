/*
    $Id: gspan.cpp,v 1.8 2004/05/21 09:27:17 taku-ku Exp $;

   Copyright (C) 2004 Taku Kudo, All rights reserved.
     This is free software with ABSOLUTELY NO WARRANTY.

   This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

   This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
     Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
     02111-1307, USA
 */
#include <gspan.hpp>
#include <iterator>

#include <stdlib.h>
#include <unistd.h>
#include <cstdio>

#include <iostream>

using namespace std;

namespace GSPAN {

gSpan::gSpan(void)
{
  logger = Logger::get_logger("GSPAN_SEQ");
  output = 0;
  minimal_support = -1;
}


void gSpan::set_database(types::graph_database_t &graph_database)
{
  this->graph_database = graph_database;
}


void gSpan::set_min_support(int minsup)
{
  minimal_support = minsup;
}

void gSpan::set_graph_output(graph_output * gout)
{
  output = gout;
}


std::istream &gSpan::read(std::istream &is)
{
  Graph g(directed);

  while(true) {
    g.read(is);
    if(g.empty()) break;
    graph_database.push_back(g);
  } // while

  return is;
}


std::map<unsigned int, unsigned int>
gSpan::support_counts(Projected &projected)
{
  std::map<unsigned int, unsigned int> counts;

  for(Projected::iterator cur = projected.begin();
      cur != projected.end(); ++cur) {
    counts[cur->id] += 1;
  }

  return (counts);
}


unsigned int
gSpan::support(Projected &projected)
{
  unsigned int oid = 0xffffffff;
  unsigned int size = 0;

  for(Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
    if(oid != cur->id) {
      ++size;
    }
    oid = cur->id;
  }

  return size;
}


/* Special report function for single node graphs.
 */
void gSpan::report_single(Graph &g, std::map<unsigned int, unsigned int>& ncount)
{
  unsigned int sup = 0;
  for(std::map<unsigned int, unsigned int>::iterator it = ncount.begin();
      it != ncount.end(); ++it) {
    sup += (*it).second;
  } // for it
} // gSpan::report_single


void gSpan::report(Projected &projected, unsigned int sup)
{
  output->output_graph(DFS_CODE, sup);
  //if(maxpat_max > maxpat_min && DFS_CODE.nodeCount() > maxpat_max)
  //return;
  //if(maxpat_min > 0 && DFS_CODE.nodeCount() < maxpat_min)
  //return;
  /*
     if(where) {
     *os << "<pattern>\n";
     *os << "<id>" << ID << "</id>\n";
     *os << "<support>" << sup << "</support>\n";
     *os << "<what>";
     }
   */
  /*
     if(!enc) {
     Graph g(directed);
     DFS_CODE.toGraph(g);

     if(!where)
     *os << "t # " << ID << " * " << sup;

     *os << '\n';
     g.write(*os);
     } else {
     if(!where)
     *os << '<' << ID << ">    " << sup << " [";

     DFS_CODE.write(*os);
     if(!where) *os << ']';
     }
   */
  /*
     if(where) {
     *os << "</what>\n<where>";
     unsigned int oid = 0xffffffff;
     for(Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
      if(oid != cur->id) {
        if(cur != projected.begin()) *os << ' ';
     *os << cur->id;
      }
      oid = cur->id;
     }
     *os << "</where>\n</pattern>";
     }

     *os << '\n';
     ++ID;
   */
}

/* Recursive subgraph mining function (similar to subprocedure 1
 * Subgraph_Mining in [Yan2002]).
 */
void gSpan::project(Projected &projected)
{
  // Check if the pattern is frequent enough.
  unsigned int sup = support(projected);
  if(sup < minimal_support) return;



  // The minimal DFS code check is more expensive than the support check,
  // hence it is done now, after checking the support.
  if(is_min() == false) {
    return;
  } else {
  }

  DEBUG(*logger, "executing project for code: " << DFS_CODE.to_string() << "; support: " << sup);

  // Output the frequent substructure
  report(projected, sup);


  // In case we have a valid upper bound and our graph already exceeds it,
  // return.  Note: we do not check for equality as the DFS exploration may
  // still add edges within an existing subgraph, without increasing the
  // number of nodes.
  //
  //if(maxpat_max > maxpat_min && DFS_CODE.nodeCount() > maxpat_max) return;


  // We just outputted a frequent subgraph.  As it is frequent enough, so
  // might be its (n+1)-extension-graphs, hence we enumerate them all.
  const RMPath &rmpath = DFS_CODE.buildRMPath();
  int minlabel = DFS_CODE[0].fromlabel;
  int maxtoc = DFS_CODE[rmpath[0]].to;

  Projected_map3 new_fwd_root;
  Projected_map2 new_bck_root;
  types::EdgeList edges;

  // Enumerate all possible one edge extensions of the current substructure.
  for(unsigned int n = 0; n < projected.size(); ++n) {

    unsigned int id = projected[n].id;
    PDFS *cur = &projected[n];
    History history(graph_database[id], cur);

    // XXX: do we have to change something here for directed edges?

    // backward
    for(int i = (int)rmpath.size() - 1; i >= 1; --i) {
      Edge *e = get_backward(graph_database[id], history[rmpath[i]], history[rmpath[0]], history);
      if(e)
        new_bck_root[DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
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

  // Test all extended substructures.
  // backward
  for(Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
    for(Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
      DFS_CODE.push(maxtoc, to->first, -1, elabel->first, -1);
      project(elabel->second);
      DFS_CODE.pop();
    }
  }

  // forward
  for(Projected_riterator3 from = new_fwd_root.rbegin();
      from != new_fwd_root.rend(); ++from) {
    for(Projected_iterator2 elabel = from->second.begin();
        elabel != from->second.end(); ++elabel) {
      for(Projected_iterator1 tolabel = elabel->second.begin();
          tolabel != elabel->second.end(); ++tolabel) {
        DFS_CODE.push(from->first, maxtoc + 1, -1, elabel->first, tolabel->first);
        project(tolabel->second);
        DFS_CODE.pop();
      }
    }
  }

  return;
}

void gSpan::run()
{
  run_intern();
}

void gSpan::run_intern(void)
{
  for(unsigned int id = 0; id < graph_database.size(); ++id) {
    for(unsigned int nid = 0; nid < graph_database[id].size(); ++nid) {
      if(singleVertex[id][graph_database[id][nid].label] == 0) {
        // number of graphs it appears in
        singleVertexLabel[graph_database[id][nid].label] += 1;
      }

      singleVertex[id][graph_database[id][nid].label] += 1;
    } // for nid
  } // for it


  for(std::map<unsigned int, unsigned int>::iterator it =
        singleVertexLabel.begin(); it != singleVertexLabel.end(); ++it) {
    if((*it).second < minimal_support)
      continue;

    unsigned int frequent_label = (*it).first;

    // Found a frequent node label, report it.
    Graph g(directed);
    g.resize(1);
    g[0].label = frequent_label;

    // [graph_id] = count for current substructure
    std::vector<unsigned int> counts(graph_database.size());
    for(std::map<unsigned int, std::map<unsigned int, unsigned int> >::iterator it2 =
          singleVertex.begin(); it2 != singleVertex.end(); ++it2) {
      counts[(*it2).first] = (*it2).second[frequent_label];
    } // for it2

    std::map<unsigned int, unsigned int> gycounts;
    for(unsigned int n = 0; n < counts.size(); ++n)
      gycounts[n] = counts[n];

    report_single(g, gycounts);
  } // for it

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
        // Build the initial two-node graph.  It will be grownrecursively within project.
        DFS_CODE.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
        project(tolabel->second);
        DFS_CODE.pop();
      } // for tolabel
    } // for elabel
  } // for fromlabel
} // void gSpan::run_intern(void)

} // namespace GSPAN


