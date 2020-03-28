#include <gspan_cuda_mult_block.hpp>

namespace gspan_cuda {

gspan_cuda_mult_block::gspan_cuda_mult_block()
{
  execute_tests = false;
  logger = Logger::get_logger("GSPAN_CUDA_MULT_BLOCK");

  compute_support_max_memory_size = 1024 * 1024 * 20;
  //dfs_count_per_compute_support_call = 50;

  d_graph_boundaries_scan = 0;
  d_graph_boundaries_scan_length = 0;

  d_prefiltered_dfs_elems = 0;
  d_prefiltered_dfs_elems_array_size = 0;
}


gspan_cuda_mult_block::~gspan_cuda_mult_block()
{
  CUDAFREE(d_graph_boundaries_scan, *logger);
  CUDAFREE(d_prefiltered_dfs_elems, *logger);
}


bool gspan_cuda_mult_block::should_filter_non_min(const types::DFS &dfs_elem, const types::DFSCode &code) const
{
  return true;
}

void gspan_cuda_mult_block::fill_supports(extension_element_t *d_exts,
                                          int exts_array_length,
                                          types::DFS *h_dfs_elems,
                                          types::DFS *d_dfs_elems,
                                          int dfs_elem_length,
                                          int max_graph_vertex_count,
                                          int mapped_db_size,
                                          std::vector<types::DFS> &freq_dfs_elems,
                                          std::vector<int> &freq_dfs_elems_supports)
{
  TRACE2(*logger, "--------------------------------------------------------------------");
  for(int i = 0; i < dfs_elem_length; i++) {
    TRACE3(*logger, "should compute support for dfs: " << h_dfs_elems[i].to_string());
  }
  int dfs_per_cs_call = compute_support_max_memory_size / mapped_db_size;
  //int dfs_per_cs_call = 5;//compute_support_max_memory_size / mapped_db_size;



  if(dfs_per_cs_call > dfs_elem_length) dfs_per_cs_call = dfs_elem_length;
  int last_block_size = dfs_elem_length % dfs_per_cs_call;
  int compute_support_calls = dfs_elem_length / dfs_per_cs_call + (last_block_size > 0 ? 1 : 0);
  TRACE2(*logger, "compute_support_calls: " << compute_support_calls);
  TRACE2(*logger, "dfs_per_cs_call: " << dfs_per_cs_call);
  TRACE2(*logger, "dfs_elem_length: " << dfs_elem_length);

  for(int i = 0; i < compute_support_calls; i++) {
    TRACE2(*logger, "------------ computing support for dfs block " << i);
    int dfs_in_this_block = dfs_per_cs_call;
    if(i == compute_support_calls - 1 && dfs_elem_length % dfs_per_cs_call != 0) {
      dfs_in_this_block = dfs_elem_length % dfs_per_cs_call;
    }

    int *supports = 0;
    int dfs_start_idx = dfs_per_cs_call * i;
    TRACE2(*logger, "dfs_start_idx: " << dfs_start_idx);
    TRACE2(*logger, "dfs_in_this_block: " << dfs_in_this_block);

    /*
    compute_support_remapped_db_multiple_dfs_blocks(d_exts,
                                                    exts_array_length,
                                                    d_dfs_elems + dfs_start_idx,
                                                    dfs_in_this_block,
                                                    supports,
                                                    max_graph_vertex_count,
                                                    d_graph_flags,
                                                    d_graph_flags_length,
                                                    d_graph_boundaries_scan,
                                                    d_graph_boundaries_scan_length,
                                                    mapped_db_size,
                                                    reduction);
     */	
    
       compute_support_remapped_db_multiple_dfs(d_exts,
                                             exts_array_length,
                                             d_dfs_elems + dfs_start_idx,
                                             dfs_in_this_block,
                                             supports,
                                             max_graph_vertex_count,
                                             d_graph_flags,
                                             d_graph_flags_length,
                                             d_graph_boundaries_scan,
                                             d_graph_boundaries_scan_length,
                                             mapped_db_size,
					     reduction,
					     scanner);
     
    for(int dfs_id = 0; dfs_id < dfs_in_this_block; dfs_id++) {
      if(supports[dfs_id] >= minimal_support) {
        freq_dfs_elems_supports.push_back(supports[dfs_id]);
        freq_dfs_elems.push_back(h_dfs_elems[dfs_start_idx + dfs_id]);
        TRACE2(*logger, "dfs: " << h_dfs_elems[dfs_start_idx + dfs_id].to_string() << " is frequent");
      } else {
        TRACE2(*logger, "dfs: " << h_dfs_elems[dfs_start_idx + dfs_id].to_string() << " is NOT frequent");
      }
    } // for dfs_id

    delete [] supports;
  } // for i
}




/**
 * Precondition: content of h_dfs_elem is sorted by
 * types::DFS::from. That is: each block in d_exts_result has a block
 * in h_dfs_elem.
 *
 *
 */
void gspan_cuda_mult_block::filter_extensions(extension_element_t *d_exts_result,
                                              int exts_result_length,
                                              types::DFS *h_dfs_elem,
                                              int dfs_array_length,
                                              int *ext_block_offsets,
                                              int ext_num_blocks,
                                              int col_count,
                                              types::DFSCode code,
                                              extension_set_t &not_frequent_extensions,
                                              types::DFS *&h_frequent_dfs_elem,
                                              int *&h_frequent_dfs_supports,
                                              int &frequent_candidate_count)
{
  TRACE4(*logger, "#############################################################################################");
  TRACE4(*logger, "#############################################################################################");
  TRACE4(*logger, "filtering extensions, dfs_array_length: " << dfs_array_length);


  // Create list of extensions per extension_element_t block;
  std::vector<int> prefiltered_dfs_elems_offsets(ext_num_blocks + 1, -1);
  std::vector<int> prefiltered_dfs_elems_sizes(ext_num_blocks, 0);
  types::DFS *prefiltered_dfs_elems = new types::DFS[dfs_array_length];
  int prefiltered_dfs_elems_size = 0;

  // prefilter the h_dfs_elem array against the not_frequent_extensions.
  for(int i = 0; i < dfs_array_length; i++) {
    TRACE4(*logger, "h_dfs_elem[" << i << "]: " << h_dfs_elem[i].to_string());
    if(not_frequent_extensions.find(h_dfs_elem[i]) == not_frequent_extensions.end() && should_filter_non_min(h_dfs_elem[i], code)) {
      prefiltered_dfs_elems[prefiltered_dfs_elems_size] = h_dfs_elem[i];
      prefiltered_dfs_elems_size++;
    } // if
  } // for i

  if(prefiltered_dfs_elems_size == 0) return;




  // create blocks of dfs elements: offsets and sizes. The dfs
  // elements are stored in prefiltered_dfs_elems. The created
  // offsets/sizes are related to prefiltered_dfs_elems.
  int last_from = prefiltered_dfs_elems[0].from;
  prefiltered_dfs_elems_offsets[prefiltered_dfs_elems[0].from] = 0;
  for(int i = 0; i < prefiltered_dfs_elems_size; i++) {
    if(prefiltered_dfs_elems[i].from != last_from) {
      prefiltered_dfs_elems_offsets[prefiltered_dfs_elems[i].from] = i;
      TRACE4(*logger, "new dfs block: " << prefiltered_dfs_elems[i].from << " starts at: " << i);
      last_from = prefiltered_dfs_elems[i].from;
    } // if
  } // for i
  prefiltered_dfs_elems_offsets.back() = prefiltered_dfs_elems_size;


  //Now iterator through dfs extensions of each ext block and find the max_mapped_db_size;
  int max_mapped_db_size = 0;

  for(int i = 0; i < prefiltered_dfs_elems_offsets.size() - 1; i++) {
    if(prefiltered_dfs_elems_offsets[i] == -1) {
      prefiltered_dfs_elems_sizes[i] = 0;
      continue;
    }
    prefiltered_dfs_elems_sizes[i] = get_block_size(i, prefiltered_dfs_elems_offsets.data(), prefiltered_dfs_elems_offsets.size() - 1);
    TRACE4(*logger, "block size: " << prefiltered_dfs_elems_sizes[i]);
  } // for i


  // copy the prefiltered dfs elements to device

  //allocate first if necessary
  //types::DFS *d_prefiltered_dfs_elems = 0;
  if(prefiltered_dfs_elems_size > d_prefiltered_dfs_elems_array_size) {
    if(d_prefiltered_dfs_elems != 0)
      CUDAFREE(d_prefiltered_dfs_elems, *logger);

    CUDAMALLOC(&d_prefiltered_dfs_elems, sizeof(types::DFS) * prefiltered_dfs_elems_size, *logger);
    d_prefiltered_dfs_elems_array_size = prefiltered_dfs_elems_size;
  }

  CUDA_EXEC(cudaMemcpy(d_prefiltered_dfs_elems, prefiltered_dfs_elems, sizeof(types::DFS) * prefiltered_dfs_elems_size, cudaMemcpyHostToDevice), *logger);

  // copy the ext_block offsets to device
  //int *d_ext_block_offsets;
  //CUDAMALLOC(&d_ext_block_offsets, sizeof(int) * ext_num_blocks, *logger);
  //CUDA_EXEC(cudaMemcpy(d_ext_block_offsets, ext_block_offsets, sizeof(int) * ext_num_blocks, cudaMemcpyHostToDevice), *logger);


  std::vector<types::DFS> freq_dfs_elems;
  std::vector<int> freq_dfs_elems_supports;

  //allocate the graph_boundaries_scan_array if necessary
  if(d_graph_boundaries_scan_length < exts_result_length) {
    if(d_graph_boundaries_scan != 0) {
      CUDAFREE(d_graph_boundaries_scan, *logger);
    }
    CUDAMALLOC(&d_graph_boundaries_scan, sizeof(int) * (exts_result_length), *logger);
    d_graph_boundaries_scan_length = exts_result_length;
  }
  //CUDA_EXEC(cudaMemset(d_graph_boundaries_scan, 0, sizeof(int) * exts_size), *logger);


  // decide the size of the memory used for storing flags for various
  // dbs allocate the memory and start calling the
  // compute_support_remapped_db_multiple_dfs this step involves call
  // to remap_database_graph_ids.

  TRACE4(*logger, "prefiltered_dfs_elems_offsets.size(): " << prefiltered_dfs_elems_offsets.size());
  for(int block_id = 0; block_id < prefiltered_dfs_elems_sizes.size(); block_id++) {
    if(ext_block_offsets[block_id] == -1) continue;

    TRACE2(*logger, "processing ext block: " << block_id << " ==================================================");

    int dfs_in_this_block = prefiltered_dfs_elems_sizes[block_id];
    int dfs_block_start = prefiltered_dfs_elems_offsets[block_id];
    if(dfs_in_this_block == 0) continue;

    TRACE2(*logger, "dfs_block_start: " << dfs_block_start);

    int mapped_db_size = 0;
    int ext_bo_start = ext_block_offsets[block_id];
    int ext_bo_length = get_block_size(block_id, ext_block_offsets, ext_num_blocks);
    TRACE4(*logger, "ext_bo_start: " << ext_bo_start << "; ext_bo_length: " << ext_bo_length);
    remap_database_graph_ids_mult(d_exts_result + ext_bo_start,
                                  ext_bo_length,
                                  h_cuda_graph_database.max_graph_vertex_count,
                                  d_graph_boundaries_scan + ext_bo_start,
                                  mapped_db_size,
                                  scanner);
    if(max_mapped_db_size < mapped_db_size)
      max_mapped_db_size = mapped_db_size;

  } // for (block_id)



  TRACE(*logger, "FILL SUPPORTS");
  fill_supports(d_exts_result,
                exts_result_length,
                prefiltered_dfs_elems,
                d_prefiltered_dfs_elems,
                prefiltered_dfs_elems_size,
                h_cuda_graph_database.max_graph_vertex_count,
                max_mapped_db_size,
                freq_dfs_elems,
                freq_dfs_elems_supports);


  // copy the content of freq_dfs_elems to the output array
  h_frequent_dfs_elem = new types::DFS[freq_dfs_elems_supports.size()];
  h_frequent_dfs_supports = new int[freq_dfs_elems_supports.size()];
  frequent_candidate_count = freq_dfs_elems_supports.size();
  TRACE4(*logger, "filtering dfs extensions using support: " << minimal_support);
  for(int dfs_id = 0; dfs_id < freq_dfs_elems_supports.size(); dfs_id++) {
    TRACE4(*logger, "frequent: " << freq_dfs_elems[dfs_id].to_string());
    h_frequent_dfs_elem[dfs_id] = freq_dfs_elems[dfs_id];
    TRACE4(*logger, "copied: " << h_frequent_dfs_elem[dfs_id].to_string());
    h_frequent_dfs_supports[dfs_id] = freq_dfs_elems_supports[dfs_id];
  } // for dfs_id


  //CUDAFREE(d_prefiltered_dfs_elems, *logger);
  //CUDAFREE(d_ext_block_offsets, *logger);
  delete [] prefiltered_dfs_elems;
}



} // namespace gspan_cuda

