#ifndef __CUDA_SEGMENTED_OP_HPP
#define __CUDA_SEGMENTED_OP_HPP

#include <logger.hpp>
#include <cuda_tools.hpp>
#include <cucpp.h>
#include <scan_types.hpp>


enum reduce_type_t {
  EXCLUSIVE = 0,
  INCLUSIVE = 1
};


class cuda_segmented_op {

protected:
  Logger *logger;
  uint *blockScanMem;
  uint blockScanMem_size;
  uint *headFlagsMem;
  uint headFlagsMem_size;
  int2 *ranges;
  uint ranges_size;
  std::vector<int2> ranges_vec;

  uint *d_reduce_result;
  uint d_reduce_result_size;


  uint *d_segmented_count;
  uint d_segmented_count_size;

  uint *d_segmented_sizes;
  uint d_segmented_sizes_size;


  //template<class T>
  //void copy_vector_to_device(T *&where, uint &where_size, const std::vector<T> &vec);


  void SetBlockRanges(KernelParams params, int count);
  void copy_ranges_to_device();
  void allocate_block_memory(uint numBlocks);


public:
  cuda_segmented_op();
  virtual ~cuda_segmented_op();


};

#endif

