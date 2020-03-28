#include <cuda_segmented_scan.hpp>


#include <scangen.cu>
//#include <segscanpacked.cu>


cuda_segmented_scan::cuda_segmented_scan()
{
}


cuda_segmented_scan::~cuda_segmented_scan()
{
}


void cuda_segmented_scan::scan(uint *d_packed_array, int count, reduce_type_t inclusive)
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

  SegScanDownsweepPacked<<< ranges_vec.size(), PACKED_NUM_THREADS >>>(d_packed_array, d_packed_array, blockScanMem, ranges, count,  inclusive);

  CUDA_EXEC(cudaDeviceSynchronize(), *logger);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "error while executing kernel ; error: " << err << "; string: " << cudaGetErrorString(err));
    abort();
  }

}



