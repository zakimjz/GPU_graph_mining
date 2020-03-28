#include <cuda_segmented_scan.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <cuda_tools.hpp>
#include <memory_checker.hpp>

static Logger *logger = Logger::get_logger("TST");


static void do_single_segment_type_test(int segment_size, int segment_count, reduce_type_t reduction_type)
{
  DEBUG(*logger, "segment_size: " << segment_size << "; " << segment_count);
  int size = segment_size * segment_count;

  unsigned int *h_array = new unsigned int[size];
  unsigned int *d_array = 0;

  for(int i = 0; i < size; i++) {
    h_array[i] = 1;
    if(i % segment_size == 0 && i != 0) {
      h_array[i] = h_array[i] | 0x80000000;
    }
  }

  copy_h_array_to_d(d_array, h_array, size);

  cuda_segmented_scan *scanner = new cuda_segmented_scan();

  TRACE(*logger, "calling cuda_segmented_scan::scan");
  scanner->scan(d_array, size, reduction_type);

  unsigned int *copied_result = new unsigned int[size];
  memset(copied_result, 0, sizeof(unsigned int) * size);
  CUDA_EXEC(cudaMemcpy(copied_result, d_array, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost), *Logger::get_logger("UTILS"));


  unsigned int part_sum = 0;
  for(int i = 0; i < size; i++) {
    unsigned int tmp = h_array[i];
    if(tmp & 0x80000000) {
      part_sum = tmp & 0x7fffffff;
    }  else {
      part_sum += tmp;
    }
    tmp = tmp & 0x7fffffff;

    if(reduction_type == INCLUSIVE) {
      tmp = part_sum;
    } else {
      tmp = part_sum - tmp;
    }

    if(tmp != copied_result[i]) {
      CRITICAL_ERROR(*logger, "values differs at position " << i << "; host computed value: " << part_sum << "; cuda computed value: " << copied_result[i]);
    }
    ASSERT_EQ(tmp, copied_result[i]);

  }

  CUDAFREE(d_array, *logger);
  delete scanner;
}



static void scan_basic_test_single_segment(reduce_type_t reduction_type)
{
  do_single_segment_type_test(1024, 32, reduction_type);
  do_single_segment_type_test(1000, 10, reduction_type);
  for(int i = 10; i < 25; i++) {
    do_single_segment_type_test(505, i, reduction_type);
  }

  for(int i = 10; i < 250; i++) {
    do_single_segment_type_test(16, i, reduction_type);
  }

  for(int i = 10; i < 250; i++) {
    do_single_segment_type_test(15, i, reduction_type);
  }


  for(int i = 10; i < 250; i++) {
    do_single_segment_type_test(17, i, reduction_type);
  }
}

TEST(segscan, basic_test)
{
  scan_basic_test_single_segment(INCLUSIVE);
  scan_basic_test_single_segment(EXCLUSIVE);

  memory_checker::detect_memory_leaks();
}


