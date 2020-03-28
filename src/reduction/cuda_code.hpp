#ifndef __CUDA_CODE_HPP__
#define __CUDA_CODE_HPP__

extern "C" __global__
void SegReduceDownsweepPacked_faster(const uint* packedIn_global, uint* valuesOut_global,
                                     const uint* start_global, const int2* rangePairs_global, int count,
                                     int inclusive,
                                     uint *segment_sizes, uint *segment_count, uint num_segments);


extern "C" __global__
void SegReduceDownsweepPacked(const uint* packedIn_global, uint* valuesOut_global,
                              const uint* start_global, const int2* rangePairs_global, int count,
                              int inclusive,
                              uint *segment_sizes, uint *segment_count, uint num_segments);



extern "C" __global__
void GlobalScanReduction(uint* blockTotals_global, uint numBlocks);


//extern "C" __global__
//void GlobalScanReduction(uint* blockTotals_global, int numBlocks);

extern "C"  __global__
void GlobalScanDownsweep(const uint* valuesIn_global, uint* valuesOut_global,
                         const uint* blockScan_global, const int2* range_global, int count,
                         int inclusive);



extern "C" __global__
void SegScanDownsweepPacked(const uint* packedIn_global, uint* valuesOut_global,
                            const uint* start_global, const int2* rangePairs_global, int count,
                            int inclusive);


extern "C" __global__
void SegScanUpsweepPacked(const uint* packedIn_global, uint* blockLast_global,
                          uint* headFlagPos_global, const int2* rangePairs_global);


extern "C" __global__
void SegReduceDownsweepPacked(const uint* packedIn_global, uint* valuesOut_global,
                              const uint* start_global, const int2* rangePairs_global, int count,
                              int inclusive,
                              uint *segment_sizes, uint *segment_count, uint num_segments);


extern "C" __global__
void SegScanReduction(const uint* headFlags_global, uint* blockLast_global,
                      uint numBlocks);


extern "C" __global__
void GlobalScanUpsweep(const uint* valuesIn_global, uint* blockTotals_global, const int2* range_global);

extern "C" __global__
void GlobalScanReduction(uint* blockTotals_global, uint numBlocks);

#endif


