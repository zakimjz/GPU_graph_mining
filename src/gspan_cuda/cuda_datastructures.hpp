#ifndef __CUDA_DATASTRUCTURES_HPP__
#define __CUDA_DATASTRUCTURES_HPP__


#include <embedding_lists.hpp>
#include <cuda_tools.hpp>

#define FRWD 0
#define BKWD 1

namespace gspan_cuda {

struct extension_element_t {
  __device__ __host__

  extension_element_t(){

  }

  extension_element_t(const extension_element_t &ee)  {
    from =  ee.from;
    to = ee.to;
    fromlabel = ee.fromlabel;
    elabel = ee.elabel;
    tolabel = ee.tolabel;
    from_grph = ee.from_grph;
    to_grph = ee.to_grph;
    row = ee.row;
    //col = ee.col;
  }

  extension_element_t(int from, int to, int fromlabel, int elabel, int tolabel, int from_grph, int to_grph, int row) {
    this->from = from;
    this->to = to;
    this->fromlabel = fromlabel;
    this->elabel = elabel;
    this->tolabel = tolabel;

    this->from_grph = from_grph; // from in graph numbering
    this->to_grph = to_grph;   // to in graph numbering

    this->row = row;
  }


  int from;
  int to;
  int fromlabel;
  int elabel;
  int tolabel;

  int from_grph; // from in graph numbering
  int to_grph;   // to in graph numbering

  int row;

  __host__ __device__
  bool is_backward() const {
    return from > to;
  }

  __host__ __device__
  bool is_forward() const {
    return from < to;
  }



  //__device__ __host__
  std::string to_string() const {

    std::stringstream ss;
    ss << "[ (" << from_grph << "," << to_grph << ") (" << from << "," << to << ": " << fromlabel << "," << elabel << "," << tolabel << ") (" << row << ") ]";
    return ss.str();
  }

  friend ostream &operator<<(ostream &out, const extension_element_t &elem) {
    //out << "EE(" << elem.from << "," << elem.to << "," << elem.fromlabel << "," << elem.elabel << "," << elem.tolabel
    //<< "," << elem.from_grph << "," << elem.to_grph << "," << elem.row << "," << elem.col << ")";
    out << elem.to_string();
    return out;
  }
};


template<typename T>
struct dfs_comparator {

  __device__ __host__
  dfs_comparator() {
  }

  __device__ __host__
  int operator()(const T &elem1, const T &elem2) const {

    //(i,j) != (x,y)

    if(elem1.from != elem2.from || elem1.to != elem2.to) { //( i != x || j !=y )

      //if elem1 is a forwward edge
      if(elem1.from < elem1.to) {
        //if elem2 is a forward edge
        if(elem2.from < elem2.to) {
          //(j < y || ( j == y && i > x ) )
          if(elem1.to < elem2.to || ( elem1.to == elem2.to && elem1.from > elem2.from ))
            return -1;

        }else{      //if elem2 is a backward edge

          if(elem1.from < elem2.to)          // (i < y)
            return -1;
        }

      }else{      //if elem1 is a backward edge

        //if elem2 is a backward edge
        if(elem2.from > elem2.to) {
          //(i < x || ( i == x && j < y ) )
          if(elem1.from < elem2.from || ( elem1.from == elem2.from && elem1.to < elem2.to ))
            return -1;

        }else{      //if elem2 is a forward edge

          if(elem1.to <= elem2.from)          // (j <= x)
            return -1;
        }

      }
    }else{ // ( i == x && j == y)

      if ( elem1.fromlabel < elem2.fromlabel) return -1;
      else if ( elem1.fromlabel > elem2.fromlabel) return 1;
      else {   //elem1.tolabel == elem2.tolabel

        if ( elem1.elabel < elem2.elabel) return -1;
        else if ( elem1.elabel > elem2.elabel) return 1;
        else {

          if ( elem1.tolabel < elem2.tolabel) return -1;
          else if ( elem1.tolabel > elem2.tolabel) return 1;
          else return 0;  //elem1 = elem2
        }
      }

    }

    //elem1 > elem2
    return 1;


  } // operator()
};

// in order to compare the extensions, I need to have the ordering
// according to the pattern, not according to the graph in which the
// pattern is embedded !
struct extension_element_comparator_neel {

  int max_node;
  __device__ __host__
  extension_element_comparator_neel(const extension_element_comparator_neel &other) {
    max_node = other.max_node;
  }

  __device__ __host__
  extension_element_comparator_neel(int max_node) {
    this->max_node = max_node;
  }

  __device__ __host__
  bool operator()(const extension_element_t &elem1, const extension_element_t &elem2) const {

    dfs_comparator<extension_element_t> comp;
    if(comp(elem1,elem2) == -1) return true;   //elem1 < elem2
    else if(comp(elem1,elem2) ==  1) return false;  //elem1 > elem2
    else{ //elem1 == elem2

      /* if the elem1 is from g1 and elem2 is from g2
         if g1 < g2 return  true, if g1 > g2 return false */
      if( elem1.from_grph < elem2.from_grph )
        return true;
      else if ( elem1.from_grph > elem2.from_grph )
        return false;
    }

    return false;

  } // operator()
};



struct extension_element_comparator_kesslr2 {

  int max_node;
  __device__ __host__
  extension_element_comparator_kesslr2(const extension_element_comparator_neel &other) {
    max_node = other.max_node;
  }

  __device__ __host__
  extension_element_comparator_kesslr2(int max_node) {
    this->max_node = max_node;
  }

  __device__ __host__
  bool operator()(const extension_element_t &elem1, const extension_element_t &elem2) const {
    if(elem1.from < elem2.from) return true;
    if(elem1.from > elem2.from) return false;

    if(elem1.to < elem2.to) return true;
    if(elem1.to > elem2.to) return false;

    if(elem1.fromlabel < elem2.fromlabel) return true;
    if(elem1.fromlabel > elem2.fromlabel) return false;

    if(elem1.elabel < elem2.elabel) return true;
    if(elem1.elabel > elem2.elabel) return false;

    if(elem1.tolabel < elem2.tolabel) return true;
    if(elem1.tolabel > elem2.tolabel) return false;

    if(elem1.row < elem2.row) return true;
    if(elem1.row > elem2.row) return false;

    if(elem1.from_grph < elem2.from_grph) return true;
    if(elem1.from_grph > elem2.from_grph) return false;

    if(elem1.to_grph < elem2.to_grph) return true;
    if(elem1.to_grph > elem2.to_grph) return false;

    return false;
  } // operator()
};


struct backlink_offset_t {

  int back_link;
  int offset;
  int len;

  __device__ __host__
  bool operator<(const backlink_offset_t &b2) const {
    return back_link < b2.back_link;
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "( " << offset << " " << back_link << " ) ";
    return ss.str();
  }

};

struct embedding_extension_t {
  types::embedding_element *embedding_column;
  backlink_offset_t *backlink_offsets;
  int col_length;
  int num_backlinks;
  types::DFS dfs_elem;
  int ext_type;  // FRWD, BKWD
  bool located_on_host;
  int *filtered_emb_offsets;  //filtered offsets relative to first input column in the intersection operations
  int filtered_emb_offsets_length;
  int support;
  int *graph_id_list;

  __device__ __host__
  embedding_extension_t(bool located_on_host) {
    init();
    this->located_on_host = located_on_host;
  }

  __device__ __host__
  embedding_extension_t(bool located_on_host, int type) {
    init();
    this->located_on_host = located_on_host;
    ext_type = type;
  }


  __device__ __host__ void init() {
    embedding_column = 0;
    backlink_offsets = 0;
    col_length = -1;
    num_backlinks = -1;
    ext_type = -1;
    filtered_emb_offsets = 0;
    filtered_emb_offsets_length = -1;
    support = -1;
    graph_id_list = 0;
  }

  void device_free() {
    TRACE(*Logger::get_logger("EMB_EXT"), "deallocating: " << backlink_offsets);
    CUDAFREE(embedding_column, *Logger::get_logger("EMB_EXT"));
    CUDAFREE(filtered_emb_offsets, *Logger::get_logger("EMB_EXT"));
    CUDAFREE(backlink_offsets, *Logger::get_logger("EMB_EXT"));
    CUDAFREE(graph_id_list, *Logger::get_logger("EMB_EXT"));
    embedding_column = 0;
    filtered_emb_offsets = 0;
    backlink_offsets = 0;
    graph_id_list = 0;
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "[" << dfs_elem.to_string() << ", support: " << support;
    if(ext_type == FRWD) ss << "; FRWD; ";
    else ss << "; BKWD; ";

    if(is_forward()) ss << ", col length: " << col_length;
    else ss << "; filtered_emb_offsets_length: " << filtered_emb_offsets_length;
    ss << ", num_backlinks: " << num_backlinks;
    ss << "]";
    return ss.str();
  }

  std::string to_string_neel() const {
    std::stringstream ss;

    if(ext_type == FRWD) ss << "FRWD ";
    else ss << "BKWD ";

    for(int i = 0; i < col_length; i++) {
      ss << "num_cols=" << col_length << ", " << "num_blinks=" << num_backlinks << " :";
      ss << " [ " << embedding_column[i].to_string() << " " << backlink_offsets[i].to_string() << " ] ";
    }
    return ss.str();
  }

  bool is_backward() const {
    return ext_type == BKWD;
  }

  bool is_forward() const {
    return ext_type == FRWD;
  }

  std::string str_ext_type() {
    if(ext_type == BKWD) return "BKWD";
    if(ext_type == FRWD) return "FRWD";
    return "UNKNOWN";
  }
};



struct embedding_extension_dfs_elem_less_then_comparator_t {
  dfs_comparator<types::DFS> comp;
  bool operator()(const embedding_extension_t &ee1, const embedding_extension_t &ee2) const {
    return comp(ee1.dfs_elem, ee2.dfs_elem);
  }
};

struct embedding_extension_dfs_elem_greater_then_comparator_t {
  dfs_comparator<types::DFS> comp;
  bool operator()(const embedding_extension_t &ee1, const embedding_extension_t &ee2) const {
    return comp(ee2.dfs_elem, ee1.dfs_elem);
  }
};




struct cuda_allocs_for_get_all_extensions {
  int *d_rmpath, d_rmpath_size;
  int *d_scheduled_rmpath_cols, d_scheduled_rmpath_cols_size;
  int *valid_vertex_indices, valid_vertex_indices_size;
  int *extension_offsets, extension_offsets_size;
  int *d_frwd_edge_labels, d_frwd_edge_labels_size;
  int *d_frwd_vertex_labels, d_frwd_vertex_labels_size;
  extension_element_t *extension_array;
  int extension_array_size;
  Logger *logger;

  cuda_allocs_for_get_all_extensions(){
  }

  void init(){

    d_rmpath = 0;
    d_rmpath_size = 0;

    d_scheduled_rmpath_cols = 0;
    d_scheduled_rmpath_cols_size = 0;

    valid_vertex_indices = 0;
    valid_vertex_indices_size = 0;

    extension_offsets = 0;
    extension_offsets_size = 0;

    d_frwd_edge_labels = 0;
    d_frwd_edge_labels_size = 0;

    d_frwd_vertex_labels = 0;
    d_frwd_vertex_labels_size = 0;

    extension_array = 0;
    extension_array_size = 0;

    logger = Logger::get_logger("CUDA_ALLOCS_GET_EXT");

  }


  int *get_d_rmpath(int len){

    if(len > d_rmpath_size) {
      if(d_rmpath != 0) {
        CUDAFREE(d_rmpath, *logger);
      }
      CUDAMALLOC(&d_rmpath, sizeof(int) * len, *logger);
      d_rmpath_size = len;
    }

    return d_rmpath;
  }

  int *get_d_scheduled_rmpath_cols(int len){

    if(len > d_scheduled_rmpath_cols_size) {
      if(d_scheduled_rmpath_cols != 0) {
        CUDAFREE(d_scheduled_rmpath_cols, *logger);
      }
      CUDAMALLOC(&d_scheduled_rmpath_cols, sizeof(int) * len, *logger);
      d_scheduled_rmpath_cols_size = len;
    }

    return d_scheduled_rmpath_cols;
  }

  int *get_valid_vertex_indices(int len){

    if(len > valid_vertex_indices_size) {
      if(valid_vertex_indices != 0) {
        CUDAFREE(valid_vertex_indices, *logger);
      }
      valid_vertex_indices = 0;
      CUDAMALLOC(&valid_vertex_indices, sizeof(int) * len, *logger);
      valid_vertex_indices_size = len;
    }

    return valid_vertex_indices;
  }


  int *get_extension_offsets(int len){

    if(len > extension_offsets_size) {
      if(extension_offsets != 0) {
        CUDAFREE(extension_offsets, *logger);
      }
      CUDAMALLOC(&extension_offsets, sizeof(int) * len, *logger);
      extension_offsets_size = len;
    }

    return extension_offsets;
  }

  int *get_d_frwd_edge_labels(int len){

    if(len > d_frwd_edge_labels_size) {
      if(d_frwd_edge_labels != 0) {
        CUDAFREE(d_frwd_edge_labels, *logger);
      }
      CUDAMALLOC(&d_frwd_edge_labels, sizeof(int) * len, *logger);
      d_frwd_edge_labels_size = len;
    }

    return d_frwd_edge_labels;
  }

  int *get_d_frwd_vertex_labels(int len){

    if(len > d_frwd_vertex_labels_size) {
      if(d_frwd_vertex_labels != 0) {
        CUDAFREE(d_frwd_vertex_labels, *logger);
      }
      CUDAMALLOC(&d_frwd_vertex_labels, sizeof(int) * len, *logger);
      d_frwd_vertex_labels_size = len;
    }

    return d_frwd_vertex_labels;
  }

  extension_element_t *get_extension_array(int len){

    if(len > extension_array_size) {
      if(extension_array != 0) {
        CUDAFREE(extension_array, *logger);
      }
      CUDAMALLOC(&extension_array, sizeof(extension_element_t) * len, *logger);
      extension_array_size = len;
    }

    return extension_array;

  }


  void device_free(){
    CUDAFREE(d_rmpath, *logger);
    CUDAFREE(d_scheduled_rmpath_cols, *logger);
    CUDAFREE(valid_vertex_indices, *logger);
    CUDAFREE(extension_offsets, *logger);
    CUDAFREE(d_frwd_edge_labels, *logger);
    CUDAFREE(d_frwd_vertex_labels, *logger);
    if(extension_array != 0) {
      CUDAFREE(extension_array, *logger);
    }
  }




};



} // namespace gspan_cuda


#endif


