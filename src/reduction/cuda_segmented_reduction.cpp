#include <cuda_segmented_reduction.hpp>
#include <cuda_segmented_scan.hpp>

#include <util.h>
#include <kernelparams.h>
#include <cuda_code.hpp>



cuda_segmented_reduction::cuda_segmented_reduction()
{
  d_reduce_result = 0;
  d_reduce_result_size = 0;

  d_segmented_count = 0;
  d_segmented_count_size = 0;

  d_segmented_sizes = 0;
  d_segmented_sizes_size = 0;
}

cuda_segmented_reduction::~cuda_segmented_reduction()
{
  CUDAFREE(d_reduce_result, *logger);
  CUDAFREE(d_segmented_count, *logger);
  CUDAFREE(d_segmented_sizes, *logger);

}




////////////////////////////////////////////////////////////////////////////////
// Generic scan - single implementation for all four scan types. Switches
// over scan types when building the kernel's call stack.

void cuda_segmented_reduction::segmented_reduce_generic(uint *d_packed_array, int count,
                                                        uint *d_results, uint *d_seg_sizes, uint *d_seg_count, uint num_segments,
                                                        reduce_type_t inclusive)
{
  KernelParams reduce_kernel_params = {
    PACKED_NUM_THREADS,
    PACKED_VALUES_PER_THREAD,
    PACKED_BLOCKS_PER_SM,
    "SegScanUpsweepPacked",
    "SegScanReduction",
    "SegScanDownsweepPacked"
  };

  SetBlockRanges(reduce_kernel_params, count);
  copy_ranges_to_device();
  CUDA_EXEC(cudaMemset(blockScanMem, 0, blockScanMem_size * sizeof(uint)), *logger);

  if(ranges_vec.size() > 1) {
    SegScanUpsweepPacked<<< ranges_vec.size() - 1,  PACKED_NUM_THREADS >>>(d_packed_array, blockScanMem, headFlagsMem, ranges);

    CUDA_EXEC(cudaDeviceSynchronize(), *logger);

    SegScanReduction<<< 1, REDUCTION_NUM_THREADS >>>(headFlagsMem, blockScanMem, ranges_vec.size());

    CUDA_EXEC(cudaDeviceSynchronize(), *logger);
  }

  SegReduceDownsweepPacked_faster<<< ranges_vec.size(), PACKED_NUM_THREADS >>>(d_packed_array, d_results,
                                                       blockScanMem, ranges, count,
                                                       inclusive,
                                                       d_seg_sizes, d_seg_count, num_segments);

  CUDA_EXEC(cudaDeviceSynchronize(), *logger);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "error while executing kernel ; error: " << err << "; string: " << cudaGetErrorString(err));
    abort();
  }

}


void cuda_segmented_reduction::reduce(uint *d_array_packed,
                                      int count,
                                      std::vector<uint> segment_count,
                                      std::vector<uint> segment_sizes,
                                      std::vector<uint> &results,
                                      reduce_type_t inclusive)
{
  copy_vector_to_device(d_segmented_count, d_segmented_count_size, segment_count);
  copy_vector_to_device(d_segmented_sizes, d_segmented_sizes_size, segment_sizes);

  int seg_count = 0;
  for(int i = 0; i < segment_count.size(); i++) seg_count += segment_count[i];
  resize_d_array(d_reduce_result,   d_reduce_result_size,   seg_count);

  resize_d_array(d_segmented_sizes, d_segmented_sizes_size, segment_sizes.size() + 1);
  resize_d_array(d_segmented_count, d_segmented_count_size, segment_count.size() + 1);

  assert(segment_sizes.size() == segment_count.size());
  CUDA_EXEC(cudaMemset(d_reduce_result, 0, sizeof(uint) * seg_count), *logger);

  segmented_reduce_generic(d_array_packed, count, d_reduce_result, d_segmented_sizes, d_segmented_count, segment_sizes.size(), inclusive);



  unsigned int *h_results = 0;
  results.clear();
  copy_d_array_to_h(d_reduce_result, seg_count, h_results);
  for(int i = 0; i < seg_count; i++) {
    results.push_back(h_results[i]);
  }

}



void cuda_segmented_reduction::reduce_inclusive(uint *d_array_packed,
                                                int count,
                                                std::vector<uint> segment_count,
                                                std::vector<uint> segment_sizes,
                                                std::vector<uint> &results)
{
  reduce(d_array_packed, count, segment_count, segment_sizes, results, INCLUSIVE);
}



void cuda_segmented_reduction::reduce_exclusive(uint *d_array_packed,
                                                int count,
                                                std::vector<uint> segment_count,
                                                std::vector<uint> segment_sizes,
                                                std::vector<uint> &results)
{
  reduce(d_array_packed, count, segment_count, segment_sizes, results, EXCLUSIVE);

}






















cuda_segmented_scan::cuda_segmented_scan()
{
}


cuda_segmented_scan::~cuda_segmented_scan()
{
}


void cuda_segmented_scan::scan(uint *d_packed_array_in, uint *d_packed_array_out, int count, reduce_type_t inclusive)
{
  KernelParams reduce_kernel_params = {
    PACKED_NUM_THREADS,
    PACKED_VALUES_PER_THREAD,
    PACKED_BLOCKS_PER_SM,
    "SegScanUpsweepPacked",
    "SegScanReduction",
    "SegScanDownsweepPacked"
  };

  SetBlockRanges(reduce_kernel_params, count);
  copy_ranges_to_device();
  CUDA_EXEC(cudaMemset(blockScanMem, 0, blockScanMem_size * sizeof(uint)), *logger);

  if(ranges_vec.size() > 1) {
    SegScanUpsweepPacked<<< ranges_vec.size() - 1,  PACKED_NUM_THREADS >>>(d_packed_array_in, blockScanMem, headFlagsMem, ranges);

    CUDA_EXEC(cudaDeviceSynchronize(), *logger);
    SegScanReduction<<< 1, REDUCTION_NUM_THREADS >>>(headFlagsMem, blockScanMem, ranges_vec.size());

    CUDA_EXEC(cudaDeviceSynchronize(), *logger);
  }

  SegScanDownsweepPacked<<< ranges_vec.size(), PACKED_NUM_THREADS >>>(d_packed_array_in, d_packed_array_out, blockScanMem, ranges, count,  inclusive);

  CUDA_EXEC(cudaDeviceSynchronize(), *logger);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "error while executing kernel ; error: " << err << "; string: " << cudaGetErrorString(err));
    abort();
  }

}



void cuda_segmented_scan::global_scan(uint *d_array_in, uint *d_array_out, int count, reduce_type_t inclusive)
{
  KernelParams reduce_kernel_params = {
    SCAN_NUM_THREADS,
    SCAN_VALUES_PER_THREAD,
    SCAN_BLOCKS_PER_SM,
    "GlobalScanUpsweep",
    "GlobalScanReduction",
    "GlobalScanDownsweep"
  };

  SetBlockRanges(reduce_kernel_params, count);
  copy_ranges_to_device();
  CUDA_EXEC(cudaMemset(blockScanMem, 0, blockScanMem_size * sizeof(uint)), *logger);

  if(ranges_vec.size() > 1) {
    GlobalScanUpsweep<<< ranges_vec.size() - 1,  PACKED_NUM_THREADS >>>(d_array_in, blockScanMem, ranges);

    CUDA_EXEC(cudaDeviceSynchronize(), *logger);
    GlobalScanReduction<<< 1, REDUCTION_NUM_THREADS >>>(blockScanMem, (uint)ranges_vec.size());

    CUDA_EXEC(cudaDeviceSynchronize(), *logger);
  }

  GlobalScanDownsweep<<< ranges_vec.size(), PACKED_NUM_THREADS >>>(d_array_in, d_array_out, blockScanMem, ranges, count,  inclusive);

  CUDA_EXEC(cudaDeviceSynchronize(), *logger);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "error while executing kernel ; error: " << err << "; string: " << cudaGetErrorString(err));
    abort();
  }

}

