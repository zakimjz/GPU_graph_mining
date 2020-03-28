#include <cuda_configurator.hpp>
#include <kernel_execution.hpp>
#include <cuda_code.hpp>
#include <cuda_tools.hpp>
//#include <copyif.cu>

template<class T, class predicate>
cuda_copy<T, predicate>::cuda_copy()
{
  remap_array = 0;
  remap_array_length = 0;
  logger = Logger::get_logger("CC");
  //scanner = new cuda_segmented_scan();
}

template<class T, class predicate>
cuda_copy<T, predicate>::~cuda_copy()
{
  CUDAFREE(remap_array, *logger);
  remap_array = 0;
  remap_array_length = 0;

  //delete scanner;
}


template<class T, class predicate>
void cuda_copy<T, predicate>::remap_input_array(uint *d_remap_index_array, int count, reduce_type_t inclusive, predicate pred)
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
    CopyIfUpsweep<<< ranges_vec.size() - 1,  SCAN_NUM_THREADS >>>(blockScanMem, ranges, pred);
    CUDA_EXEC(cudaDeviceSynchronize(), *logger);

    GlobalScanReduction<<< 1, REDUCTION_NUM_THREADS >>>(blockScanMem, ranges_vec.size());
    CUDA_EXEC(cudaDeviceSynchronize(), *logger);
  }

  CopyIfDownsweep<<< ranges_vec.size(), SCAN_NUM_THREADS >>>(d_remap_index_array, blockScanMem, ranges, count,  inclusive, pred);

  CUDA_EXEC(cudaDeviceSynchronize(), *logger);

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_PP"), "error while executing kernel ; error: " << err << "; string: " << cudaGetErrorString(err));
    abort();
  }

}


template<class T, class predicate>
int cuda_copy<T, predicate>::copy_if(T *array_begin, int length, T *to_array_begin, predicate p)
{
  resize_d_array(remap_array, remap_array_length, (length + 1));
  TRACE(*logger, "remapping input array, length: " << length << "; remap_array_length: " << remap_array_length);
  remap_input_array(remap_array, length, EXCLUSIVE, p);

  TRACE(*logger, "getting configuration");
  cudapp::cuda_computation_parameters params = cudapp::cuda_configurator::get_computation_parameters(length, 128);
  TRACE(*logger, "configuration: " << params.to_string());


  TRACE(*logger, "copying output size");
  int copied_count = 0;
  CUDA_EXEC(cudaMemcpy(&copied_count, remap_array+length, sizeof(int), cudaMemcpyDeviceToHost), *logger);

  TRACE(*logger, "performing actual copying, copied_count: " << copied_count);
  copy_kernel cp(array_begin, to_array_begin, remap_array, p);
  cudapp::for_each<copy_kernel>(0, length, cp, params);


  return copied_count;
}





