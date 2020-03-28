#include <cuda_segmented_op.hpp>

cuda_segmented_op::cuda_segmented_op()
{
  logger = Logger::get_logger("CUDA_SEG_REDUCE");
  blockScanMem = 0;
  blockScanMem_size = 0;
  headFlagsMem = 0;
  headFlagsMem_size = 0;
  ranges = 0;
  ranges_size = 0;

  blockScanMem = 0;
  blockScanMem_size = 0;
  headFlagsMem = 0;
  headFlagsMem_size = 0;



  allocate_block_memory(10);
}

cuda_segmented_op::~cuda_segmented_op()
{
  CUDAFREE(blockScanMem, *logger);
  CUDAFREE(headFlagsMem, *logger);
  CUDAFREE(ranges, *logger);


}


void cuda_segmented_op::SetBlockRanges(KernelParams params, int count)
{
  int numBlocks = params.blocksPerSM; //engine->numBlocks[kind];
  int blockSize = params.numThreads * params.valuesPerThread; //engine->blockSize[kind];
  int numBricks = DivUp(count, blockSize);
  if(numBlocks > numBricks) numBlocks = numBricks;

  ranges_vec.resize(numBlocks);

  // Distribute the work along complete bricks.
  div_t brickDiv = div(numBricks, numBlocks);

  // Distribute the work along complete bricks.
  for(int i(0); i < numBlocks; ++i) {
    int2 range;
    range.x = i ? ranges_vec[i - 1].y : 0;
    int bricks = (i < brickDiv.rem) ? (brickDiv.quot + 1) : brickDiv.quot;
    range.y = std::min(range.x + bricks * blockSize, count);
    ranges_vec[i] = range;
  }
}




void cuda_segmented_op::copy_ranges_to_device()
{
  if(ranges == 0) {
    CUDAMALLOC(&ranges, sizeof(uint2) * ranges_vec.size(), *logger);
    ranges_size = ranges_vec.size();
  } else if(ranges_size < ranges_vec.size()) {
    CUDAFREE(ranges, *logger);
    CUDAMALLOC(&ranges, sizeof(uint2) * ranges_vec.size(), *logger);
    ranges_size = ranges_vec.size();
  }

  CUDA_EXEC(cudaMemcpy(ranges, ranges_vec.data(), sizeof(uint2) * ranges_vec.size(), cudaMemcpyHostToDevice), *logger);
}


void cuda_segmented_op::allocate_block_memory(uint numBlocks)
{
  //if(numBlocks > 1025) abort();
  if(blockScanMem == 0) {
    resize_d_array(blockScanMem, blockScanMem_size, 1025);
    resize_d_array(headFlagsMem, headFlagsMem_size, 1025);
    return;
  }

  resize_d_array(blockScanMem, blockScanMem_size, numBlocks);
  resize_d_array(headFlagsMem, headFlagsMem_size, numBlocks);
}



