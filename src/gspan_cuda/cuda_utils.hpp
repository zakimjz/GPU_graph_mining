#ifndef __CUDA_UTILS_HPP__
#define __CUDA_UTILS_HPP__

#include <cuda_datastructures.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_configurator.hpp>
#include <embedding_lists.hpp>
#include <cuda_datastructures.hpp>
#include <cuda_segmented_scan.hpp>

namespace gspan_cuda {

void extract_embedding_column(extension_element_t *d_exts,
                              int d_exts_size,
                              types::DFS dfs_elem,
                              types::embedding_element *&emb_col,
                              int &emb_col_size,
                              cuda_segmented_scan *scanner);

void compact_labels(types::graph_database_cuda &gdb, std::set<int> &vertex_label_set, std::set<int> &edge_label_set);

} // namespace gspan_cuda


#endif

