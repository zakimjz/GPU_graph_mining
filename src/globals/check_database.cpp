#include <graph_types.hpp>
#include <types.hpp>
#include <types.hpp>
#include <stdexcept>
#include <iostream>
#include <sstream>

using namespace std;

void check_graph(const types::Graph &grph)
{
  for(int vid = 0; vid != grph.size(); vid++) {
    std::set<int> to_set;
    for(int eid = 0; eid != grph[vid].edge.size(); eid++) {
      int to = grph[vid].edge[eid].to;
      if(to_set.find(to) == to_set.end()) {
        to_set.insert(to);
      } else {
        cerr << "error while processing vertex " << vid << "; eid: " << eid << "; edge count: " << grph[vid].edge.size() << endl;
        throw runtime_error("edge already exists");
      } // if

      if(grph[vid].edge[eid].to == grph[vid].edge[eid].from) {
        cerr << "error while processing edge: " << grph[vid].edge[eid].to << ", " << grph[vid].edge[eid].from << endl;
        throw runtime_error("selfloop");
      } // if
    } // for eid
  } // for vid
} // check_graph



void check_database(const types::graph_database_t &db)
{
  for(int i = 0; i < db.size(); i++) {
    try {
      check_graph(db[i]);
    } catch(...) {
      cerr << "error while processing graph " << i << endl;
      throw;
    }
  } // for i
} // check_database



void print_database_statistics(const types::graph_database_t &db)
{
  std::set<int> vlabels;
  std::set<int> elabels;

  for(int gid = 0; gid < db.size(); gid++) {
    const types::Graph &grph = db[gid];
    for(int vid = 0; vid < grph.size(); vid++) {
      vlabels.insert(grph[vid].label);
      if(grph[vid].label > 30 || grph[vid].label < 0) {
        std::cout << "vertex label gid: " << gid << "; vid: " << vid << std::endl;
      }

      for(int eid = 0; eid < grph[vid].edge.size(); eid++) {
        elabels.insert(grph[vid].edge[eid].elabel);
        if(grph[vid].edge[eid].elabel > 30 || grph[vid].edge[eid].elabel < 0) {
          std::cout << "gid: " << gid << std::endl;
        }
      } // for eid
    } // for vid
  } // for gid

  std::stringstream ss;
  for(std::set<int>::iterator itV = vlabels.begin(); itV != vlabels.end(); itV++) {
    ss << *itV << " ";
  } // for itV
  std::cout << "vlabels: " << ss.str() << endl;

  ss.str("");
  for(std::set<int>::iterator itE = elabels.begin(); itE != elabels.end(); itE++) {
    ss << *itE << " ";
  } // for itE
  std::cout << "elabels: " << ss.str() << endl;

} // print_database_statistics


