#ifndef __CUDA_SEGMENTED_REDUCTION_HPP__
#define __CUDA_SEGMENTED_REDUCTION_HPP__

#include <logger.hpp>
#include <cuda.h>
#include <vector>
#include <cucpp.h>
#include <scan_types.hpp>
#include <cuda_tools.hpp>
#include <cuda_segmented_op.hpp>

#include <vector>


class cuda_segmented_reduction : public cuda_segmented_op {
protected:
  void segmented_reduce_generic(uint *d_packed_array, int count,
                                uint *d_seg_results, uint *d_segmented_sizes, uint *d_segment_count, uint num_segments,
                                reduce_type_t inclusive);

public:
  cuda_segmented_reduction();
  ~cuda_segmented_reduction();

  void reduce(uint *d_array_packed, int count, std::vector<uint> segment_count, std::vector<uint> segment_sizes, std::vector<uint> &results, reduce_type_t inclusive);
  void reduce_inclusive(uint *d_array_packed, int count, std::vector<uint> segment_count, std::vector<uint> segment_sizes, std::vector<uint> &results);
  void reduce_exclusive(uint *d_array_packed, int count, std::vector<uint> segment_count, std::vector<uint> segment_sizes, std::vector<uint> &results);

};



#endif


