#include <cuda_copy.hpp>
#include <gtest/gtest.h>


static Logger *logger = Logger::get_logger("TST");


struct is_odd {
  int *d_array;
  is_odd(int *array) {
    d_array = array;
  }
  __host__ __device__
  bool operator()(int index) {
    return (d_array[index] % 2) == 1;
  }
};

void copy_basic_test(int array_size)
{
  int *h_array = new int[array_size];
  for(int i = 0; i < array_size; i++) {
    h_array[i] = i;
  }
  int *d_array = 0;
  copy_h_array_to_d(d_array, h_array, array_size);
  int *d_array_out = 0;
  uint d_array_out_length = 0;
  resize_d_array(d_array_out, d_array_out_length, array_size);

  cuda_copy<int, is_odd> copy;
  int result_size = copy.copy_if(d_array, array_size, d_array_out, is_odd(d_array));

  cout << "result_size: " << result_size << endl;
  int *h_array_out = new int[result_size];
  copy_d_array_to_h(d_array_out, result_size, h_array_out);
  for(int i = 0; i < result_size; i++) {
    if( (2 * i + 1) != h_array_out[i]) {
      CRITICAL_ERROR(*logger, "error at position: " << i << "; h_array_out[i]: " << h_array_out[i]);
    } // if
  } // for i

  CUDAFREE(d_array_out, *logger);
  CUDAFREE(d_array, *logger);
} // copy_basic_test

TEST(copy, basic_test)
{
  copy_basic_test(1000);
  memory_checker::detect_memory_leaks();
}


