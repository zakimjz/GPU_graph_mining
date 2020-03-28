#ifndef __CUDA_MEMORY_CHECKER_HPP__
#define __CUDA_MEMORY_CHECKER_HPP__

#include <string>
#include <sstream>

#define CUDA_REGISTER_MBLOCK(__ptr__)    memory_checker::register_memory_block(__ptr__, __LINE__, __FILE__)
#define CUDA_UNREGISTER_MBLOCK(__ptr__)  memory_checker::unregister_memory_block(__ptr__)

#define CUDA_REGISTER_MBLOCK2(__ptr__, __size__)    memory_checker::register_memory_block(__ptr__, __size__, __LINE__, __FILE__)

namespace memory_checker {

struct allocated_block_descriptor {
  allocated_block_descriptor(void *ptr, int line, std::string file) {
    this->line = line;
    this->file = file;
    this->ptr = ptr;
    size = 0;
  }
  allocated_block_descriptor(void *ptr, unsigned int size, int line, std::string file) {
    this->line = line;
    this->file = file;
    this->ptr = ptr;
    this->size = size;
  }
  int line;
  std::string file;
  void *ptr;
  unsigned int size;

  std::string to_string() const {
    std::stringstream ss;
    ss << file << "@" << line << "; ptr: " << ptr << "; size: " << size;
    return ss.str();
  }
};
void register_memory_block(void *ptr, int line, const std::string &file);
void register_memory_block(void *ptr, unsigned int size, int line, const std::string &file);
void unregister_memory_block(void *ptr);
void print_unallocated_blocks(bool report_leaks = true);
void detect_memory_leaks(bool report_leaks = true);
bool is_block_allocated(void *ptr);
std::string print_memory_usage();
double get_memory_usage();
double get_max_memory_usage_mb();


} // namespace memory_checker

#endif


