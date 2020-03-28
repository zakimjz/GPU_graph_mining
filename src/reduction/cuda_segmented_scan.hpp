#ifndef __CUDA_SEGMENTED_SCAN_HPP__
#define __CUDA_SEGMENTED_SCAN_HPP__

#include <cuda_segmented_op.hpp>


class cuda_segmented_scan : public cuda_segmented_op {
public:
  cuda_segmented_scan();
  ~cuda_segmented_scan();
  void scan(uint *d_array_packed, uint *d_array_out, int count, reduce_type_t inclusive);
  void global_scan(uint *d_array_in, uint *d_array_out, int count, reduce_type_t inclusive);

  void scan(uint *d_array_packed, int count, reduce_type_t inclusive) {
    scan(d_array_packed, d_array_packed, count, inclusive);
  }


  void exclusive_scan(uint *d_array_packed, int count) {
    scan(d_array_packed, count, EXCLUSIVE);
  }

  void inclusive_scan(uint *d_array_packed, int count) {
    scan(d_array_packed, count, INCLUSIVE);
  }

};



#endif


