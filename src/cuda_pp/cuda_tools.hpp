#ifndef __CUDA_TOOLS_HPP__
#define __CUDA_TOOLS_HPP__


#include <memory_checker.hpp>
#include <logger.hpp>
#include <vector>
#include <cassert>

#define CUDA_EXEC(__call__, __log__)        \
  { \
    __call__;                              \
    cudaError_t err = cudaGetLastError();    \
    if(err != cudaSuccess) {   \
      CRITICAL_ERROR(__log__, "could not execute " #__call__ "; error: " << err << "; string: " << cudaGetErrorString(err)); \
      throw std::runtime_error("Error in some cuda call"); \
    } \
  }



#define CUDAMALLOC(__ptr__, __size__, __logger__) {                 \
    cudaError_t cuda_err = cudaMalloc(__ptr__, __size__);  \
    if(cuda_err != cudaSuccess) { \
      CRITICAL_ERROR(__logger__, "cudaMalloc error at " << __FILE__ << "@" << __LINE__ << "; err string: " << cudaGetErrorString(cuda_err)); \
      throw std::runtime_error("Error in cuda memory allocation"); \
    } \
    CUDA_REGISTER_MBLOCK2(*__ptr__, __size__);          \
}



#define CUDAFREE(__ptr__, __logger__) {             \
    cudaError_t cuda_err = cudaFree(__ptr__);  \
    if(cuda_err != cudaSuccess) { \
      CRITICAL_ERROR(__logger__, "cudaFree error at " << __FILE__ << "@" << __LINE__ << "; err string: " << cudaGetErrorString(cuda_err)); \
      throw std::runtime_error("Error in cuda memory free"); \
    } \
    CUDA_UNREGISTER_MBLOCK(__ptr__); \
}



template<typename T>
std::string print_d_array2(T* d_array, int len){
  T* h_array = new T[len];
  CUDA_EXEC(cudaMemcpy(h_array, d_array, len * sizeof(T), cudaMemcpyDeviceToHost), *Logger::get_logger("UTILS"));

  std::stringstream ss;

  ss << "len: " << len << "; arr: ";
  for(int i = 0; i<len; i++) {
    ss << h_array[i].to_string() << "(" << i << ") ";
  }

  delete [] h_array;
  return ss.str();
}



template<class T>
std::string print_d_array(T *darr, int l)
{
  std::stringstream ss;
  T *harr = new T[l];
  CUDA_EXEC(cudaMemcpy(harr, darr, sizeof(T) * l, cudaMemcpyDeviceToHost), *Logger::get_logger("UTILS"));

  ss << "len: " << l << "; arr: ";
  for(int i = 0; i < l; i++) {
    ss << harr[i] << "(" << i << "), ";
  } // i

  delete [] harr;
  return ss.str();
} // print_d_array



template<class T>
std::string print_h_array(T *harr, int l, const string &label = "array")
{
  Logger *logger = Logger::get_logger("UTILS");
  std::stringstream ss;
  ss << label << "; ";
  ss << "len: " << l << "; arr: ";
  for(int i = 0; i < l; i++) {
    ss << harr[i] << "(" << i << "), ";
  } // i

  return ss.str();
} // print_d_array


template<class T>
void copy_d_array_to_h(T *da, int l, T *&ha)
{
  delete [] ha;
  ha = new T[l];

  CUDA_EXEC(cudaMemcpy(ha, da, sizeof(T) * l, cudaMemcpyDeviceToHost), *Logger::get_logger("UTILS"));
}



template<class T>
void copy_h_array_to_d(T *&da, T *ha, int l)
{
  assert(da == 0);

  CUDAMALLOC(&da, sizeof(T) * l, *Logger::get_logger("UTILS"));
  CUDA_EXEC(cudaMemcpy(da, ha, sizeof(T) * l, cudaMemcpyHostToDevice), *Logger::get_logger("UTILS"));
}




template<class T>
void copy_vector_to_device(T *&where, uint &where_size, const std::vector<T> &vec)
{
  if(where_size < vec.size()) {
    CUDAFREE(where, *Logger::get_logger("UTILS"));
    CUDAMALLOC(&where, sizeof(T) * vec.size(), *Logger::get_logger("UTILS"));
    where_size = vec.size();
  } // if

  CUDA_EXEC(cudaMemcpy(where, vec.data(), sizeof(T) * vec.size(), cudaMemcpyHostToDevice), *Logger::get_logger("UTILS"));
}


template<class T>
void resize_d_array(T *&da, uint &size, uint newsize) {
  if(da == 0) {
    CUDAMALLOC(&da, newsize * sizeof(T), *Logger::get_logger("UTILS"));
    size = newsize;
    return;
  }
  if(size < newsize) {
    CUDAFREE(da, *Logger::get_logger("UTILS"));
    CUDAMALLOC(&da, sizeof(T) * newsize, *Logger::get_logger("UTILS"));
    size = newsize;
  }
}


#endif

