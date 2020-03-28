#include <cuda_segmented_reduction.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <memory_checker.hpp>
#include <cuda_tools.hpp>

static Logger *logger = Logger::get_logger("TST");

TEST(segreduce, gspan_bug1) {

  uint array[60] = {1,          0, 1, 1, 1, 1, 1, 1, 1, 1,
                    2147483648, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    2147483648, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    2147483648, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                    2147483648, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                    2147483649, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  uint *d_array = 0;
  int array_length = 60;

  copy_h_array_to_d(d_array, array, array_length);


  int block_size = 10;
  std::vector<uint> seg_sizes;
  seg_sizes.push_back(10);
  std::vector<uint> seg_count;
  seg_count.push_back(6);

  std::vector<uint> result;

  cuda_segmented_reduction *r = new cuda_segmented_reduction();

  TRACE(*logger, "calling cuda_segmented_reduction::reduce");
  r->reduce_inclusive(d_array, array_length, seg_count, seg_sizes, result);
  ASSERT_EQ(result[0], 9);
  ASSERT_EQ(result[1], 1);
  ASSERT_EQ(result[2], 1);
  ASSERT_EQ(result[3], 4);
  ASSERT_EQ(result[4], 2);
  ASSERT_EQ(result[5], 10);

  delete r;

  CUDAFREE(d_array, *logger);

  memory_checker::detect_memory_leaks();
}


