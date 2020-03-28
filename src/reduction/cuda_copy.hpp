#ifndef __CUDA_COPY_HPP__
#define __CUDA_COPY_HPP__

#include <logger.hpp>
#include <cuda_tools.hpp>
#include <kernelparams.h>
#include <cuda_segmented_op.hpp>
#include <stdio.h>

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

// Use a 33-slot stride for shared mem transpose.
#define WARP_STRIDE (WARP_SIZE + 1)

typedef unsigned int uint;

#define DEVICE extern "C" __forceinline__ __device__
#define DEVICE2 __forceinline__ __device__

#define ROUND_UP(x, y) (~(y - 1) & (x + y - 1))

#define LOG_BASE_2(x) \
  ((1 == x) ? 0 : \
   ((2 == x) ? 1 : \
    ((4 == x) ? 2 : \
     ((8 == x) ? 3 : \
      ((16 == x) ? 4 : \
       ((32 == x) ? 5 : 0) \
      ) \
     ) \
    ) \
   ) \
  )



#include <scancommon.cu>
#include <copyif.cu>


template<class T, class predicate>
class cuda_copy : public cuda_segmented_op {
protected:
  uint *remap_array;
  uint remap_array_length;
  Logger *logger;

  void remap_input_array(uint *d_remap_index_array, int count, reduce_type_t inclusive, predicate pred);

public:

  struct copy_kernel {
    uint *remap_array;
    T *from_array;
    T *to_array;
    predicate pred;

    copy_kernel(T *from_array, T *to_array, uint *remap_array, predicate pred) : pred(pred) {
      this->remap_array = remap_array;
      this->from_array = from_array;
      this->to_array = to_array;
    }

    __device__ __host__
    void operator()(int idx) {
      if(pred(idx)) {
        uint dest_id = remap_array[idx];
        to_array[dest_id] = from_array[idx];
      } // if
    } // operator()
  };


  cuda_copy();
  ~cuda_copy();

  int copy_if(T *array_begin, int length, T *to_array_begin, predicate p);
};



#include <cuda_copy.tcc>



#endif

