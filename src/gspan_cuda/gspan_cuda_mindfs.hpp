#ifndef __GSPAN_CUDA_MINDFS_HPP__
#define __GSPAN_CUDA_MINDFS_HPP__

#include <gspan_cuda_no_sort_block.hpp>
#include <logger.hpp>

namespace gspan_cuda {

/**
 * Experimental version (does not works). Push the minimality checking
 * before computation of support. THIS VERSION DOES NOT WORKS !
 *
 */
class gspan_cuda_mindfs : public gspan_cuda_no_sort_block {
protected:
  Logger *logger;


  virtual void filter_rmpath(const types::RMPath &gspan_rmpath_in,
                             types::RMPath &gspan_rmpath_out,
                             types::DFS *h_dfs_elem,
                             int *supports,
                             int h_dfs_elem_count);

  virtual void filter_rmpath(const types::RMPath &gspan_rmpath_in,
                             types::RMPath &gspan_rmpath_out,
                             types::RMPath &gspan_rmpath_has_extension);

  virtual bool should_filter_non_min(const types::DFS &dfs_elem, const types::DFSCode &code) const;

public:
  gspan_cuda_mindfs();
  virtual ~gspan_cuda_mindfs();
};

} // namespace gspan_cuda

#endif

