#ifndef __ADJ_MATRIX_HPP__
#define __ADJ_MATRIX_HPP__

#include <graph_types.hpp>
#include <string>
#include <logger.hpp>


namespace types {

/**
 * this struct stores the database in the array format.
 *
 *
 */
struct graph_database_cuda {

  bool located_on_host; //< true if this database is stored on host, false otherwise.


  int edges_sizes;

  /**
   * edges contains the whole neigborhood of all vertices. The offset
   * of vertex vid into the array edges is stored at
   * vertex_offsets[vid]. Has the size edges_sizes.
   */
  int *edges;
  int *edges_labels; //< contains labels of the edges, has the same size as the array edges.
  int *vertex_labels; //< contains vertex labels, has the same size as vertex_offsets.

  /**
   * stores the vertex offsets. Each vertex id maps to one offset,
   * i.e., vertex_offsets[vid]. If vertex_offsets[vid] == -1 the vid
   * is invalid vertex id.  the size of this array is
   * max_graph_vertex_count*db_size
   */
  int *vertex_offsets;

  int max_graph_vertex_count; //< max_graph_vertex_count stores the maximum number of vertices on a graph in this database.
  int vertex_count; //< total number of valid vertices.
  int db_size; //< number of graphs in this array


  graph_database_cuda(bool located_on_host) {
    edges = 0;
    edges_labels = 0;
    vertex_offsets = 0;
    vertex_labels = 0;
    max_graph_vertex_count = -1;
    vertex_count = -1;
    db_size = -1;
    //d_thread_vertex_mapping = 0;
    this->located_on_host = located_on_host;
  } // graph_database_cuda

  static graph_database_cuda create_from_host_representation(const types::graph_database_t &gdb);
  void copy_to_device(graph_database_cuda *device_gdb);
  void copy_from_device(graph_database_cuda *device_gdb);
  graph_database_cuda host_copy();

  std::string to_string() const;
  std::string to_string_no_content() const;

  void delete_from_host();
  void delete_from_device();

  __host__
  std::string graph_to_string(int grph_id) const;

  graph_database_cuda & operator=(const graph_database_cuda &other);

  void convert_to_host_representation(types::graph_database_t &gdb);
  void shrink_database(int minimal_support);


  int size() const {
    return db_size;
  }


  __host__ __device__
  int graph_vertex_count(int gid) const {
    for(int i = 0; i < max_graph_vertex_count; i++) {
      //  int *vertex_offsets;
      if(vertex_offsets[gid * max_graph_vertex_count + i] == -1) return i;
    } // for i
    return -1;
  } // graph_database_cuda::graph_vertex_count


  __host__ __device__
  bool vertex_is_valid(int idx) const {
    return idx < max_graph_vertex_count * db_size && (vertex_offsets[idx] != -1);
  }


  __host__ __device__
  int get_next_valid_vertex_idx(int vid) const {
    if(vid ==  max_graph_vertex_count * db_size - 1) return vid + 1;
    if(vertex_offsets[vid + 1] == -1) {
      return vid + max_graph_vertex_count - (vid % max_graph_vertex_count);
    } // if
    return vid + 1;
  } // d_get_next_valid_vertex_idx


  __host__ __device__
  int get_vertex_degree(int vid) const {
    if(!vertex_is_valid(vid)) return 0;
    int next_vid = get_next_valid_vertex_idx(vid);
    return vertex_offsets[next_vid] - vertex_offsets[vid];
  }

  __host__ __device__
  int get_vertex_label(int vid) const {
    return vertex_labels[vid];
  }


  __host__ __device__
  int get_graph_id(int vid) const {
    return vid / max_graph_vertex_count;
  }

  __host__ __device__
  int get_host_graph_vertex_id(int d_vid) const {
    return d_vid % max_graph_vertex_count;
  }


  __host__ __device__
  int get_device_graph_vertex_id(int h_vid, int grph_id) const {
    return grph_id * max_graph_vertex_count + h_vid;
  }

  __host__ __device__
  int get_neigh_offsets(int vid) const {
    return vertex_offsets[vid];
  }

  __host__ __device__
  int get_edge_label(int from, int to) const {
    int neigh_offset = vertex_offsets[from];
    if(vertex_is_valid(from) == -1) return -1;
    int degree = get_vertex_degree(from);
    for(int i = 0; i < degree; i++) {
      if(edges[neigh_offset + i] == to) return edges_labels[neigh_offset + i];
    } // for i
    return -1;
  } // get_edge_label

  __host__ __device__
  int translate_to_device(int gid, int vid) const {
    return gid * max_graph_vertex_count + vid;
  }

  bool is_on_host() {
    return located_on_host;
  }
  bool is_on_device() {
    return located_on_host;
  }

  bool test_database() const;
};


}

#endif

