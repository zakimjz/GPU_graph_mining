#ifndef __GRAPH_TYPES_HPP__
#define __GRAPH_TYPES_HPP__

#include <types.hpp>
#include <dfs_code.hpp>
#include <set>
#include <map>
#include <string>

#include <cuda_graph_types.hpp>

namespace types {

typedef int edge_label_t;
typedef int vertex_label_t;

typedef std::set<edge_label_t> edge_label_set_t;
typedef std::set<vertex_label_t> vertex_label_set_t;

// Originaly from Taku Kudo's implementation of the gSpan algorithm.
struct Edge {
  int from;
  int to;
  int elabel;
  unsigned int id;
  Edge() : from(0), to(0), elabel(0), id(0) {
  }
  std::string to_string() const;
};

class Vertex {
public:
  typedef std::vector<Edge>::iterator edge_iterator;
  typedef std::vector<Edge>::const_iterator const_edge_iterator;

  int label;
  std::vector<Edge> edge;

  void push(int from, int to, int elabel) {
    edge.resize(edge.size() + 1);
    edge[edge.size() - 1].from = from;
    edge[edge.size() - 1].to = to;
    edge[edge.size() - 1].elabel = elabel;
    return;
  }

  bool find(int from, int to, Edge &result) const {
    for(int i = 0; i < edge.size(); i++) {
      if(edge[i].from == from && edge[i].to == to) {
        result = edge[i];
        return true;
      }
    } // for i
    return false;
  } // find


  static size_t get_serialized_size(const Vertex &vrtx);
  static size_t get_serialized_size(char *buffer, size_t buffer_size);
  static size_t serialize(const Vertex &vrtx, char *buffer, size_t buffer_size);
  static size_t deserialize(Vertex &vrtx, char *buffer, size_t buffer_size);
};

class Graph : public std::vector<Vertex> {
private:
  unsigned int edge_size_;
public:
  typedef std::vector<Vertex>::iterator vertex_iterator;

  Graph(bool _directed);
  bool directed;

  //  int y; // class label
  unsigned int edge_size() const {
    return edge_size_;
  }
  unsigned int vertex_size() const {
    return (unsigned int)size();
  }                                                                 // wrapper
  void buildEdge();
  std::istream &read(std::istream &);  // read
  std::istream &read_fsg(std::istream &);  // read
  std::ostream &write(std::ostream &);  // write
  void check(void);

  Graph();
  types::DFSCode get_min_dfs_code() const;
  int get_vertex_label(int vid) const {
    return at(vid).label;
  }

  bool is_subgraph(DFSCode &other_graph_dfs) const;

  std::string to_string() const;

  static size_t get_serialized_size(const Graph &grph);
  static size_t get_serialized_size(char *buffer, size_t buffer_size);
  static size_t serialize(const Graph &grph, char *buffer, size_t buffer_size);
  static size_t deserialize(Graph &grph, char *buffer, size_t buffer_size);

  static size_t get_serialized_size(const graph_database_t &grph_db);
  static size_t get_serialized_size_db(char *buffer, size_t buffer_size);
  static size_t serialize(const graph_database_t &grph_db, char *buffer, size_t buffer_size);
  static size_t deserialize(graph_database_t &grph_db, char *buffer, size_t buffer_size);


  /*
     static size_t get_serialized_size(const graph_database_t &grph_db, const std::set<int> &tids);
     static size_t get_serialized_size(const graph_database_t &grph_db, const std::set<int> &tids, char *buffer, size_t buffer_size);
     static size_t serialize(const graph_database_t &grph_db, const std::set<int> &tids, char *buffer, size_t buffer_size);
     static size_t deserialize(graph_database_t &grph_db, const std::set<int> &tids, char *buffer, size_t buffer_size);
   */

protected:
  void get_min_dfs_code_internal(Projected &projected, DFSCode &min_dfs_code) const;
  bool is_subgraph_internal(Projected &projected, DFSCode &other_graph_dfs, int depth) const;
};













struct graph_database_cuda;


struct PDFS {
  unsigned int id;      // ID of the original input graph
  Edge        *edge;
  PDFS        *prev;
  PDFS() : id(0), edge(0), prev(0) {
  };
  std::string to_string() const;
  std::string to_string_projection(types::graph_database_t &gdb, types::graph_database_cuda &cgdb) const;
};


/**
 * Stores information of edges/nodes that were already visited in the
 * current DFS branch of the search.
 */
class History : public std::vector<Edge*> {
private:
  std::vector<int> edge;
  std::vector<int> vertex;

public:
  bool hasEdge(unsigned int id) {
    return (bool)edge[id];
  }
  bool hasVertex(unsigned int id) {
    return (bool)vertex[id];
  }
  void build(const Graph &, PDFS *);
  History() {
  }
  History(const Graph &g, PDFS *p) {
    build(g, p);
  }
  std::string to_string() const;
};

class Projected : public std::vector<PDFS> {
public:
  void push(int id, Edge *edge, PDFS *prev) {
    resize(size() + 1);
    PDFS &d = (*this)[size() - 1];
    d.id = id;
    d.edge = edge;
    d.prev = prev;
  }
  std::string to_string() const;
};


typedef std::map<int, std::map <int, std::map <int, Projected> > >           Projected_map3;
typedef std::map<int, std::map <int, Projected> >                            Projected_map2;
typedef std::map<int, Projected>                                             Projected_map1;
typedef std::map<int, std::map <int, std::map <int, Projected> > >::iterator Projected_iterator3;
typedef std::map<int, std::map <int, Projected> >::iterator Projected_iterator2;
typedef std::map<int, Projected>::iterator Projected_iterator1;
typedef std::map<int, std::map <int, std::map <int, Projected> > >::reverse_iterator Projected_riterator3;



typedef std::vector<int> graph_id_list_t;
typedef std::map<int, graph_id_list_t>   edge_gid_list1_t;
typedef std::map<int, edge_gid_list1_t>  edge_gid_list2_t;
typedef std::map<int, edge_gid_list2_t>  edge_gid_list3_t;


bool  get_forward_pure(const Graph &, Edge *,  int, History&, types::EdgeList &);
bool  get_forward_rmpath(const Graph &, Edge *,  int,  History&, types::EdgeList &);
bool  get_forward_root(const Graph &, const Vertex &, types::EdgeList &);
Edge *get_backward(const Graph &, Edge *,  Edge *, History&);



} // namespace types

#endif

