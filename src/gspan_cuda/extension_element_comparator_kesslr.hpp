#ifndef __EXTENSION_ELEMENT_COMPARATOR_KESSLR_HPP__
#define __EXTENSION_ELEMENT_COMPARATOR_KESSLR_HPP__

#include <cuda_graph_types.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>


namespace gspan_cuda {

// in order to compare the extensions, I need to have the ordering
// according to the pattern, not according to the graph in which the
// pattern is embedded !
struct extension_element_comparator_kesslr {
  extension_element_t *array;
  int length;
  int max_node;
  __device__ __host__
  extension_element_comparator_kesslr(const extension_element_comparator_kesslr &other) {
    max_node = other.max_node;
    length = other.length;
    array = other.array;
  }

  __device__ __host__
  extension_element_comparator_kesslr(int max_node) {
    this->max_node = max_node;
    array = 0;
    length = 0;
  }

  __device__ __host__
  extension_element_comparator_kesslr(int max_node, extension_element_t *array, int length) {
    this->max_node = max_node;
    this->array = array;
    this->length = length;
  }

  __device__ __host__
  bool compare_dfs_part(const extension_element_t &elem1, const extension_element_t &elem2) const {
    if(elem1.is_backward() && elem2.is_forward()) return true;
    if(elem1.is_forward() && elem2.is_backward()) return false;

    if(elem1.is_backward() && elem2.is_backward()) {
      if(elem1.to < elem2.to) return true;
      if(elem1.to == elem2.to && elem1.elabel < elem2.elabel) return true;
      return false;
    } // if

    // we have two forward edges
    if(elem1.from > elem2.from) return true;
    if(elem1.from < elem2.from) return false;
    if(elem1.fromlabel < elem2.fromlabel) return true;
    if(elem1.fromlabel == elem2.fromlabel && elem1.elabel < elem2.elabel) return true;
    if(elem1.fromlabel == elem2.fromlabel && elem1.elabel == elem2.elabel && elem1.tolabel < elem2.tolabel) return true;
    return false;
  }

  __device__ __host__
  bool operator()(const extension_element_t &elem1, const extension_element_t &elem2) const {
    if(array != 0) {
      if(&elem1 < array || array + length <= &elem1) {
        printf("got element at offset: %d\n", (int)(&elem1 - array));
      }
      if(&elem2 < array || array + length <= &elem2) {
        printf("got element at offset: %d\n", (int)(&elem2 - array));
      }

    }

    if(elem1.from == elem2.from &&
       elem1.to == elem2.to &&
       elem1.fromlabel == elem2.fromlabel &&
       elem1.elabel == elem2.elabel &&
       elem1.tolabel == elem2.tolabel &&
       elem1.from_grph == elem2.from_grph &&
       elem1.to_grph == elem2.to_grph &&
       elem1.row == elem2.row) {
      return (elem1.from_grph / max_node) < (elem2.from_grph / max_node);
    }


    return compare_dfs_part(elem1, elem2);
  }
};


} // namespace gspan_cuda



#endif

