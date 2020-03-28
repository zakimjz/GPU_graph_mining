#include <gspan_cuda_mindfs.hpp>





namespace gspan_cuda {

bool gspan_cuda_mindfs::should_filter_non_min(const types::DFS &dfs_elem, const types::DFSCode &code) const
{
  INFO(*logger, "gspan_cuda_mindfs::should_filter_non_min");
  types::DFSCode new_code = code;
  new_code.push_back(dfs_elem);
  bool ismin = new_code.dfs_code_is_min();

  //INFO(*logger, "gspan_cuda_mindfs::should_filter_non_min; ismin: " << ismin);

  return ismin;
}


void gspan_cuda_mindfs::filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::DFS *h_dfs_elem, int *supports, int h_dfs_elem_count)
{
  gspan_rmpath_out = gspan_rmpath_in;
}


void gspan_cuda_mindfs::filter_rmpath(const types::RMPath &gspan_rmpath_in, types::RMPath &gspan_rmpath_out, types::RMPath &gspan_rmpath_has_extension)
{
  gspan_rmpath_out = gspan_rmpath_in;
}


gspan_cuda_mindfs::gspan_cuda_mindfs()
{
  logger = Logger::get_logger("GSPAN_CUDA_MINDFS");
}


gspan_cuda_mindfs::~gspan_cuda_mindfs()
{
}

} // namespace gspan_cuda


