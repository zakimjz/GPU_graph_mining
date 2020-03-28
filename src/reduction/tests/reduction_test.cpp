#include <gtest/gtest.h>

#include <logger.hpp>
#include <memory_checker.hpp>
#include <cuda_tools.hpp>


#include <cuda_segmented_reduction.hpp>

static Logger *logger = Logger::get_logger("TST");


static void do_single_segment_type_test(int segment_size, int segment_count, reduce_type_t reduction_type)
{
  DEBUG(*logger, "segment_size: " << segment_size << "; " << segment_count);
  //const int segment_size = 1024;
  int size = segment_size * segment_count;

  unsigned int *h_array = new unsigned int[size];
  unsigned int *d_array = 0;

  for(int i = 0; i < size; i++) {
    h_array[i] = 1;
    if(i % segment_size == 0 && i != 0) {
      h_array[i] = h_array[i] | 0x80000000;
    }
  }

  std::vector<unsigned int> segment_counts_vec;
  segment_counts_vec.push_back(size / segment_size);
  std::vector<unsigned int> segment_sizes_vec;
  segment_sizes_vec.push_back(segment_size);
  std::vector<unsigned int> result;

  copy_h_array_to_d(d_array, h_array, size);

  cuda_segmented_reduction *r = new cuda_segmented_reduction();

  TRACE(*logger, "calling cuda_segmented_reduction::reduce");
  r->reduce(d_array, size, segment_counts_vec, segment_sizes_vec, result, reduction_type);

  for(int i = 0; i < result.size(); i++) {
    if(reduction_type == INCLUSIVE) {
      if(result[i] != segment_size) {
        CRITICAL_ERROR(*logger, "i: " << i << "; value: " << result[i] << " should be " << segment_size);
      }
      ASSERT_EQ(result[i], segment_size);
    } else {
      if(result[i] != segment_size - 1) {
        CRITICAL_ERROR(*logger, "i: " << i << "; value: " << result[i] << " should be " << (segment_size - 1));
      }
      ASSERT_EQ(result[i], segment_size - 1);
    }
  }

  CUDAFREE(d_array, *logger);
  delete r;
}

static void segreduce_basic_test_single_segment(reduce_type_t reduction_type)
{
  // important test:
  //do_single_segment_type_test(17, 17);


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

TEST(segreduce, basic_test_single_segment)
{
  //cout << "segreduce" << endl;
  //do_single_segment_type_test(1024, 32, INCLUSIVE);
  //do_single_segment_type_test(17, 17, INCLUSIVE);


  DEBUG(*logger, "segreduce_basic_test_single_segment, exclusive");
  segreduce_basic_test_single_segment(EXCLUSIVE);

  DEBUG(*logger, "segreduce_basic_test_single_segment, inclusive");
  segreduce_basic_test_single_segment(INCLUSIVE);
  memory_checker::detect_memory_leaks();

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








static void do_multi_segment_type_test(std::vector<uint> segment_sizes, std::vector<uint> segment_count, reduce_type_t reduction_type)
{
  if(segment_count.size() != segment_sizes.size())
    throw std::runtime_error("Cannot create segments: segment size array and segment count array does not match.");


  int size = 0;
  std::stringstream ss;
  for(int i = 0; i < segment_sizes.size(); i++) {
    ss << "[" << segment_sizes[i] << "," <<  segment_count[i] << "]; ";
    size += segment_sizes[i] * segment_count[i];
  }
  INFO(*logger, "segments: " << ss.str());

  unsigned int *h_array = new unsigned int[size];
  unsigned int *d_array = 0;
  unsigned int seg_idx = 0;
  unsigned int seg_count = 0;
  unsigned int seg_start = 0;
  for(int i = 0; i < size; i++) {
    h_array[i] = 1;
    if((i - seg_start) % segment_sizes[seg_idx] == 0 && i != 0) {
      h_array[i] = h_array[i] | 0x80000000;
      seg_count++;
      if(seg_count == segment_count[seg_idx]) {
        seg_count = 0;
        seg_idx++;
        seg_start = i;
      }
    }
  }

  std::vector<unsigned int> result;

  copy_h_array_to_d(d_array, h_array, size);

  cuda_segmented_reduction *r = new cuda_segmented_reduction();

  TRACE(*logger, "calling cuda_segmented_reduction::reduce");
  r->reduce(d_array, size, segment_count, segment_sizes, result, reduction_type);

  seg_idx = 0;
  seg_count = 0;
  seg_start = 0;
  std::stringstream ss_result;
  for(int i = 0; i < result.size(); i++) {
    ss_result << result[i] << "(" << seg_idx << ", " << i << ")  ";

    if(reduction_type == INCLUSIVE) {
      if(result[i] != segment_sizes[seg_idx]) {
        CRITICAL_ERROR(*logger, "i: " << i << "; value: " << result[i] << " should be " << segment_sizes[seg_idx]);
      }
      ASSERT_EQ(result[i], segment_sizes[seg_idx]);
    } else {
      if(result[i] != segment_sizes[seg_idx] - 1) {
        CRITICAL_ERROR(*logger, "i: " << i << "; value: " << result[i] << " should be " << (segment_sizes[seg_idx] - 1));
      }
      ASSERT_EQ(result[i], segment_sizes[seg_idx] - 1);
    }

    seg_count++;
    if(seg_count == segment_count[seg_idx]) {
      seg_idx++;
      seg_count = 0;
    }
  }
  TRACE(*logger, "result array: " << ss_result.str());

  CUDAFREE(d_array, *logger);
  delete r;
}


static void do_multi_segment_type_test(int *seg_sizes, int *seg_count, reduce_type_t reduction_type)
{
  int i = 0;
  std::vector<uint> seg_sizes_vec;
  std::vector<uint> seg_count_vec;
  while(seg_sizes[i] != -1) {
    seg_sizes_vec.push_back(seg_sizes[i]);
    seg_count_vec.push_back(seg_count[i]);
    i++;
  }
  do_multi_segment_type_test(seg_sizes_vec, seg_count_vec, reduction_type);
}


static void segreduce_basic_test_multi_segment(reduce_type_t reduction_type)
{
  // important test:
  //do_single_segment_type_test(17, 17);

  std::vector<uint> segment_sizes;
  std::vector<uint> segment_count;

  segment_sizes.push_back(30);
  segment_count.push_back(10);

  segment_sizes.push_back(50);
  segment_count.push_back(15);
  do_multi_segment_type_test(segment_sizes, segment_count, reduction_type);


  segment_sizes.push_back(32);
  segment_count.push_back(10);

  segment_sizes.push_back(52);
  segment_count.push_back(15);


  segment_sizes.push_back(42);
  segment_count.push_back(15);
  do_multi_segment_type_test(segment_sizes, segment_count, reduction_type);


  const int seg_sizes_array[][6] = {
    {14,    16,  18,   50, 100, -1},
    {15,    16,  32,   50,  32, -1},
    {100, 1000, 100, 2000,  10, -1},
    {-1,    -1,  -1,   -1,  -1, -1}
  };

  const int seg_count_array[][6] = {
    { 5,    40,   5,   40,  50, -1},
    { 8,     8,  16,   32,  33, -1},
    {20,    20,  20,   20,  20, -1},
    {-1,    -1,  -1,   -1,  -1, -1}
  };

  do_multi_segment_type_test((int*) seg_sizes_array[0], (int*) seg_count_array[0], reduction_type);
  do_multi_segment_type_test((int*) seg_sizes_array[1], (int*) seg_count_array[1], reduction_type);
  do_multi_segment_type_test((int*) seg_sizes_array[2], (int*) seg_count_array[2], reduction_type);
}


/*
   TEST(segreduce, basic_test_multi_segment)
   {

   DEBUG(*logger, "segreduce_basic_test_multi_segment, exclusive");
   segreduce_basic_test_multi_segment(EXCLUSIVE);

   DEBUG(*logger, "segreduce_basic_test_multi_segment, inclusive");
   segreduce_basic_test_multi_segment(INCLUSIVE);
   memory_checker::detect_memory_leaks();
   }
 */







