#include <cuda_segmented_reduction.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <memory_checker.hpp>
#include <cuda_tools.hpp>

static Logger *logger = Logger::get_logger("TST");

static void make_array(uint size, uint block_size, uint *&correct_block_count, uint *&data_array)
{
  data_array = new uint[size];
  correct_block_count = new uint[size/block_size];

  //int max_block_ones = 3;
  int current_block_ones = rand() % block_size;

  int block_id = 0;

  correct_block_count[block_id] = current_block_ones;
  block_id++;

  for(int i = 0; i < size; i++) {
    data_array[i] = 0;

    if(i > 0 && (i % block_size) == 0) {
      data_array[i] += 2147483648;
      current_block_ones = rand() % block_size;
      correct_block_count[block_id] = current_block_ones;
      block_id++;
    } // if

    if(current_block_ones > 0) {
      data_array[i] += 1;
      current_block_ones--;
    } // if

    
  } // for i
  
} // make_array



TEST(segreduce, gspan_bug2)
{
  uint *d_array = 0;
  uint *h_array = 0;
  uint *correct_result = 0;

  int block_size = 12;
  int block_count = 276;
  int array_length = block_size * block_count;
  make_array(array_length, block_size, correct_result, h_array);

  copy_h_array_to_d(d_array, h_array, array_length);
  cout << print_d_array(d_array, array_length) << endl;
  cout << endl;
  cout << print_h_array(correct_result, block_count, "correct result") << endl;
  cout << endl;

  //int block_size = 10;
  std::vector<uint> seg_sizes;
  seg_sizes.push_back(block_size);
  std::vector<uint> seg_count;
  seg_count.push_back(block_count);

  std::vector<uint> result;

  cuda_segmented_reduction *r = new cuda_segmented_reduction();

  TRACE(*logger, "calling cuda_segmented_reduction::reduce");
  r->reduce_inclusive(d_array, array_length, seg_count, seg_sizes, result);

  cout << endl;
  cout << print_h_array(result.data(), result.size(), "CUDA result   ") << endl;

  ASSERT_EQ(result.size(), block_count);
  for(int i = 0; i < result.size(); i++) {
    if(result[i] != correct_result[i]) {
      cout << "error at position: " << i << endl;
    }
    EXPECT_EQ(result[i], correct_result[i]);
  }

  delete r;

  CUDAFREE(d_array, *logger);
  delete [] h_array;
  delete [] correct_result;
  memory_checker::detect_memory_leaks();
}


