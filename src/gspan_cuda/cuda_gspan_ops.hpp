#ifndef __CUDA_GSPAN_OPS_HPP__
#define __CUDA_GSPAN_OPS_HPP__

#include <cuda_graph_types.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>
#include <cuda_configurator.hpp>
#include <cuda_tools.hpp>

#include <cuda_datastructures.hpp>
#include <extension_element_comparator_kesslr.hpp>

#include <cuda_segmented_reduction.hpp>
#include <cuda_segmented_scan.hpp>

#include <cuda_functors.hpp>
#include <cuda_copy.hpp>


namespace gspan_cuda {




typedef extension_element_comparator_neel extension_element_comparator;

void get_all_extensions(types::graph_database_cuda *gdb,
                        cuda_allocs_for_get_all_extensions *allocs,
                        types::embedding_list_columns *embeddings,
                        types::RMPath cuda_rmpath,
                        types::RMPath host_rmpath,
                        types::DFSCode code,
                        extension_element_t *&exts_result,
                        int &exts_result_length,
                        int minlabel,
                        cudapp::cuda_configurator *exec_conf,
                        types::RMPath scheduled_rmpath_cols);

void create_first_embeddings(types::DFS first_dfs,
                             types::graph_database_cuda *gdb,
                             types::embedding_list_columns *embeddings,
                             cudapp::cuda_configurator *exec_conf);

void get_support_for_extensions(int max_vertex_count,
                                gspan_cuda::extension_element_t *h_exts_array,
                                int exts_array_length,
                                int &dfs_array_length,
                                types::DFS *&dfs_array,
                                int *&dfs_offsets,
                                int *&support,
                                cudapp::cuda_configurator *exec_conf);

void extract_extensions(int num_edge_labels,
                        int num_vertex_labels,
                        int num_columns,
                        gspan_cuda::extension_element_t *d_exts_array,
                        int exts_array_length,
                        int *&d_block_offsets,
                        int &d_block_offsets_length,
                        types::DFS *&d_dfs_array,
                        int &d_dfs_array_length,
                        int *&d_fwd_flags,
                        int *&d_bwd_flags,
                        types::DFS *&d_fwd_block_dfs_array,
                        types::DFS *&d_bwd_block_dfs_array,
                        int *&block_offsets,
                        types::DFS *&dfs_array,
                        int &dfs_array_length,
                        cuda_copy<types::DFS, is_one_array> *cucpy);

int compute_support(extension_element_t *d_exts, int d_exts_size, types::DFS dfs_elem, int db_size, int max_graph_vertex_count);

void filter_backward_embeddings(types::embedding_list_columns &embeds_in,
                                embedding_extension_t &backward_extension,
                                cudapp::cuda_configurator *exec_conf);

void filter_backward_embeddings(types::embedding_element *&input_embeddings,
                                int input_embeddings_length,
                                types::embedding_element *&filtered_embeddings,
                                int &filtered_embeddings_length,
                                int *d_input_offsets,
                                int d_input_offsets_length,
                                cudapp::cuda_configurator *exec_conf);

void filter_backward_embeddings(types::embedding_element *&input_embeddings,
                                int input_embeddings_length,
                                types::embedding_element *&filtered_embeddings,
                                int &filtered_embeddings_length,
                                int ext_index,
                                extension_element_t *exts,
                                int exts_length,
                                int *h_dfs_offsets,
                                types::DFS *h_dfs_elem,
                                int dfs_array_length,
                                cuda_segmented_scan *scanner,
                                cudapp::cuda_configurator *exec_conf);

void filter_backward_embeddings(types::embedding_list_columns &embeds_in,
                                int ext_index,
                                extension_element_t *exts,
                                int exts_length,
                                int *h_dfs_offsets,
                                types::DFS *h_dfs_elem,
                                int dfs_array_length,
                                types::embedding_element *&filtered_embeddings,
                                int &filtered_embeddings_length,
                                cuda_segmented_scan *scanner,
                                cudapp::cuda_configurator *exec_conf);

void filter_backward_embeddings(types::embedding_list_columns &embeds_in,
                                extension_element_t *d_exts,
                                int exts_length,
                                types::DFS dfs_elem,
                                types::embedding_element *&new_column,
                                int &new_column_length,
                                cuda_segmented_scan *scanner);


void intersection_fwd_fwd(embedding_extension_t &d_embedding_col_1,
                          embedding_extension_t &d_embedding_col_2,
                          embedding_extension_t &d_embedding_col_result,
                          cudapp::cuda_configurator *exec_conf);

void intersection_fwd_bwd(embedding_extension_t d_embedding_col_1,
                          embedding_extension_t d_embedding_col_2,
                          embedding_extension_t &d_embedding_col_result);


void intersection_bwd_fwd(const types::embedding_list_columns &d_embeddings,
                          const types::graph_database_cuda &cuda_gdb,
                          embedding_extension_t &d_embedding_col_1,
                          embedding_extension_t &d_embedding_col_2,
                          embedding_extension_t &filtered_last_col_emb,
                          embedding_extension_t &d_result,
                          cudapp::cuda_configurator *exec_conf);

void exp_intersection_bwd_bwd(embedding_extension_t d_embedding_col_1,
                              embedding_extension_t d_embedding_col_2,
                              embedding_extension_t &d_embedding_col_result);

int compute_support_for_fwd_ext(embedding_extension_t &d_emb,
                                int max_vertex_id);

int compute_support_for_bwd_ext(embedding_extension_t &d_emb,
                                int max_vertex_id);

int get_graph_id_list_intersection_size(embedding_extension_t &d_emb1,
                                        embedding_extension_t &d_emb2,
                                        int max_vertex_id);

bool check_equals_embedding_elements(types::embedding_element *e1,
                                     types::embedding_element *e2,
                                     int len1,
                                     int len2);


void extract_extensions(int num_edge_labels,
                        int num_vertex_labels,
                        int num_columns,
                        extension_element_t *d_exts_array,
                        int exts_array_length,
                        int *&block_offsets,
                        int &num_blocks,
                        types::DFS *&dfs_array,
                        int &dfs_array_length);

void remap_database_graph_ids(extension_element_t *d_exts,
                              int exts_size,
                              int max_graph_vertex_count,
                              int *&d_graph_boundaries_scan,
                              int &d_graph_boundaries_scan_length,
                              int &mapped_db_size,
                              cuda_segmented_scan *scanner);

int compute_support_remapped_db(extension_element_t *d_exts,
                                int exts_size,
                                types::DFS dfs_elem,
                                int max_graph_vertex_count,
                                int *d_graph_flags,
                                int *&d_graph_boundaries_scan,
                                int &graph_boundaries_scan_length,
                                bool &compute_boundaries,
                                int &mapped_db_size,
                                cuda_segmented_scan *scanner);

void compute_support_remapped_db_multiple_dfs(extension_element_t *d_exts,
                                              int exts_size,
                                              types::DFS *dfs_elem,
                                              int dfs_elem_length,
                                              int *&supports,
                                              int max_graph_vertex_count,
                                              int *&d_graph_flags,
                                              int &d_graph_flags_length,
                                              int *&d_graph_boundaries_scan,
                                              int &d_graph_boundaries_scan_length,
                                              int &mapped_db_size,
                                              cuda_segmented_reduction *reduction,
                                              cuda_segmented_scan *scanner);

void remap_database_graph_ids_mult(extension_element_t *d_exts,
                                   int exts_size,
                                   int max_graph_vertex_count,
                                   int *d_graph_boundaries_scan,
                                   int &mapped_db_size,
                                   cuda_segmented_scan *scanner);

void compute_support_remapped_db_multiple_dfs_blocks(extension_element_t *d_exts,
                                                     int exts_size,
                                                     types::DFS *dfs_elem,
                                                     int dfs_elem_length,
                                                     int *&supports,
                                                     int max_graph_vertex_count,
                                                     int *&d_graph_flags,
                                                     int &d_graph_flags_length,
                                                     int *&d_graph_boundaries_scan,
                                                     int &d_graph_boundaries_scan_length,
                                                     int &mapped_db_size,
                                                     cuda_segmented_reduction *reduction);



} // namespace gspan_cuda



#endif


