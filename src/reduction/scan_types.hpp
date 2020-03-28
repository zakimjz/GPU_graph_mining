#ifndef __SCAN_TYPES_HPP__
#define __SCAN_TYPES_HPP__

struct KernelParams {
  int numThreads;
  int valuesPerThread;
  int blocksPerSM;
  const char* pass1;
  const char* pass2;
  const char* pass3;
};



enum engine_kind_t {
  GLOBAL_SCAN = 0,
  SEGSCAN_PACKED = 1,
  SEGSCAN_FLAGS = 2,
  SEGSCAN_KEYS = 3,
  SEGREDUCTION = 4
};



struct scanEngine_d {
  ContextPtr context;

  ModulePtr module;

  FunctionPtr funcs[4][3];
  int numBlocks[4];
  int blockSize[4];

  DeviceMemPtr blockScanMem;
  DeviceMemPtr headFlagsMem;
  DeviceMemPtr rangeMem;

  std::vector<int2> ranges;
};



#endif


