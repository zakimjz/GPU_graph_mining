#include <graph_types.hpp>
#include <types.hpp>
#include <types.hpp>
#include <stdexcept>
#include <iostream>
#include <graph_repair_tools.hpp>

using namespace std;

void fix_graph(types::Graph &grph)
{
  for(int vid = 0; vid != grph.size(); vid++) {
    std::map<int, int> to_label_map;

    for(int eid = 0; eid != grph[vid].edge.size(); eid++) {
      if(grph[vid].edge[eid].to == grph[vid].edge[eid].from) continue;
      int to = grph[vid].edge[eid].to;
      int elabel = grph[vid].edge[eid].elabel;
      if(to_label_map.find(to) == to_label_map.end()) {
        to_label_map.insert(make_pair(to, elabel));
      } else if(elabel != to_label_map.find(to)->second) {
        //cerr << "error: unable to fix graph, parallel edge has different label" << endl;
        throw std::runtime_error("error: unable to fix graph, parallel edge has different label");
      } // if-else-if
    } // for eid


    std::map<int, int>::iterator itE;
    grph[vid].edge.clear();
    for(itE = to_label_map.begin(); itE != to_label_map.end(); itE++) {
      grph[vid].push(vid, itE->first, itE->second);
    }
  } // for vid

  grph.buildEdge();
}



static bool should_relabel_graph(types::Graph &grph)
{
  for(int vid = 1; vid < grph.size(); vid++) {
    for(int eid = 0; eid < grph[vid].edge.size(); eid++) {
      if(grph[vid].edge[eid].from == 0 || grph[vid].edge[eid].to == 0) {
        return false;
      }
    } // for eid
  } // for vid

  return true;
}



void relabel_graph(types::Graph &grph)
{
  grph.erase(grph.begin());

  for(int vid = 0; vid < grph.size(); vid++) {
    for(int eid = 0; eid < grph[vid].edge.size(); eid++) {
      grph[vid].edge[eid].from -= 1;
      grph[vid].edge[eid].to -= 1;
    } // for eid
  } // for vid
} // relabel_graph



void fix_database(types::graph_database_t &db)
{
  for(int i = 0; i < db.size(); i++) {
    try {
      check_graph(db[i]);
    } catch(...) {
      //cerr << "error while processing graph " << i << endl;
      fix_graph(db[i]);
    }

    if(should_relabel_graph(db[i])) {
      relabel_graph(db[i]);
    }
  } // for i
} // fix_database


