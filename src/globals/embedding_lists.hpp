#ifndef __EMBEDDING_LISTS_HPP__
#define __EMBEDDING_LISTS_HPP__

#include <cuda_graph_types.hpp>
#include <graph_types.hpp>
#include <sstream>

#include <ostream>

namespace types {

struct embedding_element {
  int vertex_id;
  int back_link;

  embedding_element(int vertex_id, int back_link){
    this->vertex_id = vertex_id;
    this->back_link = back_link;
  } //embedding_element

  embedding_element() {
    vertex_id = -1;
    back_link = -1;
  } // embedding_element

  std::string to_string() const {
    std::stringstream ss;
    ss << "(" << vertex_id << "," << back_link << ")";
    return ss.str();
  } // to_string

  bool operator==(const embedding_element &other) {
    return vertex_id == other.vertex_id && back_link == other.back_link;
  }
  friend std::ostream &operator<<(std::ostream &o, const embedding_element &e) {
    o << e.to_string();
    return o;
  }
};

struct embedding_list_columns {
  bool located_on_host;
  int columns_count;
  types::DFS *dfscode;
  int dfscode_length;

  embedding_element **columns;

  int *columns_lengths;

  embedding_list_columns(bool located_on_host) {
    columns_count = -1;
    dfscode = 0;
    dfscode_length = 0;
    columns = 0;
    columns_lengths = 0;
    this->located_on_host = located_on_host;
  } // embedding_list_columns


  void h_extend_by_one_column(types::DFS dfs_elem, embedding_element *new_col, int new_col_length);
  void d_extend_by_one_column(types::DFS dfs_elem, embedding_element *new_col, int new_col_length);
  void d_replace_last_column(embedding_element *new_col, int new_col_length);


  embedding_list_columns d_get_half_copy();
  void d_half_deallocate();
  std::string to_string() const;
  std::string embedding_to_string(int row) const;

  std::string to_string_with_labels(const types::graph_database_cuda &gdb, const types::DFSCode &code) const;

  int get_embedding_count() {
    if(located_on_host == false) {
      abort();
    }
    return columns_lengths[columns_count - 1];
  }

  // for testing and debugging purposes
  void copy_to_device(embedding_list_columns *elc);
  void copy_from_device(embedding_list_columns *elc);
  void delete_from_host();
  void delete_from_device();
};

} // namespace types


#endif

