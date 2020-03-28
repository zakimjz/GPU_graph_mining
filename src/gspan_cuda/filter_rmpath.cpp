#include <cuda_graph_types.hpp>
#include <algorithm>
#include <cassert>
#include <graph_types.hpp>
#include <dfs_code.hpp>

#include <gspan_cuda.hpp>

#include <map>


namespace gspan_cuda {


/**
 * @param gspan_rmpath_in is the input rmpath that is supposed to be filtered.
 * @param gspan_rmpath_out is the filtered rmpath
 * @param h_dfs_elem contains, in the from field, the columns that are allowed. That is: let A = {h_dfs_elem[i].from | all i}
 *                   then gspan_rmpath_out = gspan_rmpath_in \cap A
 *
 */
void gspan_cuda::filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::DFS *h_dfs_elem, int *supports, int h_dfs_elem_count)
{
  std::set<int> rmpath_vertices_with_extension;

  for(int i = 0; i < h_dfs_elem_count; i++) {
    if(supports[i] >= minimal_support) {
      rmpath_vertices_with_extension.insert(h_dfs_elem[i].from);
    }
  } // for i

  rmpath_vertices_with_extension.insert(gspan_rmpath_in.back());

  gspan_rmpath_out.clear();
  for(int i = 0; i < gspan_rmpath_in.size(); i++) {
    if(rmpath_vertices_with_extension.find(gspan_rmpath_in[i]) != rmpath_vertices_with_extension.end()) {
      gspan_rmpath_out.push_back(gspan_rmpath_in[i]);
    } // if
  } // for i
} // filter_rmpath


/**
 * @param gspan_rmpath_in is the input rmpath that is supposed to be filtered.
 * @param gspan_rmpath_out is the filtered rmpath
 * @param gspan_rmpath_has_extension are the allowed columns, i.e., for each gspan_rmpath_out = gspan_rmpath_in \cap gspan_rmpath_has_extension
 *
 */
void gspan_cuda::filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::RMPath &gspan_rmpath_has_extension)
{
  std::set<int> rmpath_vertices_with_extension;

  for(int i = 0; i < gspan_rmpath_has_extension.size(); i++) {
    rmpath_vertices_with_extension.insert(gspan_rmpath_has_extension[i]);
  } // for i

  rmpath_vertices_with_extension.insert(gspan_rmpath_in.back());


  gspan_rmpath_out.clear();
  for(int i = 0; i < gspan_rmpath_in.size(); i++) {
    if(rmpath_vertices_with_extension.find(gspan_rmpath_in[i]) != rmpath_vertices_with_extension.end()) {
      gspan_rmpath_out.push_back(gspan_rmpath_in[i]);
    } // if
  } // for i
} // filter_rmpath

} // namespace gspan_cuda

