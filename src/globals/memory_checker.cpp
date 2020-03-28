#include <memory_checker.hpp>
#include <map>
#include <logger.hpp>
#include <stdexcept>

namespace memory_checker {

typedef std::map<void *, allocated_block_descriptor> allocated_memory_blocks_t;
static allocated_memory_blocks_t mem_blocks;

static unsigned int current_allocated_size = 0;
static unsigned int max_allocated_size = 0;

void register_memory_block(void *ptr, unsigned int size, int line, const std::string &file)
{
  if(ptr == 0 || size == 0) {
    return;
  }
  if(mem_blocks.find(ptr) != mem_blocks.end()) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_MEMCHECK"),  "registering already registered memory block with ptr: " << ptr << " from " << file << "@" << line);
    abort();
  } // if
  mem_blocks.insert(std::make_pair(ptr, allocated_block_descriptor(ptr, size, line, file)));
  current_allocated_size += size;
  if(max_allocated_size < current_allocated_size) max_allocated_size = current_allocated_size;
}

void register_memory_block(void *ptr, int line, const string &file)
{
  if(ptr == 0) {
    return;
  }
  if(mem_blocks.find(ptr) != mem_blocks.end()) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_MEMCHECK"), "registering already registered memory block with ptr: " << ptr << " from " << file << "@" << line);
    CRITICAL_ERROR(*Logger::get_logger("CUDA_MEMCHECK"), "already registered from: " << mem_blocks.find(ptr)->second.to_string());
    abort();
  } // if
  mem_blocks.insert(std::make_pair(ptr, allocated_block_descriptor(ptr, line, file)));
} // register_memory_block


void unregister_memory_block(void *ptr)
{
  if(ptr == 0) {
    return;
  }
  std::map<void *, allocated_block_descriptor>::iterator it = mem_blocks.find(ptr);
  if(it == mem_blocks.end()) {
    CRITICAL_ERROR(*Logger::get_logger("CUDA_MEMCHECK"),  "un-registering memory block that was not registered with ptr: " << ptr);
    abort();
  } // if

  current_allocated_size -= it->second.size;

  mem_blocks.erase(ptr);
}

std::ostream &operator<<(ostream &out, const allocated_block_descriptor &block) {
  out << "memory block, ptr: " << block.ptr << "; allocated at " << block.file << "@" << block.line << " of size: " << block.size;
  return out;
}


void print_unallocated_blocks(bool report_leaks)
{
  for(allocated_memory_blocks_t::iterator it = mem_blocks.begin(); it != mem_blocks.end(); it++) {
    cout << it->second << endl;
  } // for it
} // print_unallocated_blocks


void detect_memory_leaks(bool report_leaks)
{
  if(mem_blocks.empty()) {
    INFO(*Logger::get_logger("CUDA_MEMCHECK"), "all memory blocks were freed !");
    return;
  }

  for(allocated_memory_blocks_t::iterator it = mem_blocks.begin(); it != mem_blocks.end(); it++) {
    cout << it->second << endl;
  } // for it
  throw std::runtime_error("Detected memory leaks in CUDA device memory !!!");
} // print_unallocated_blocks


bool is_block_allocated(void *ptr)
{
  return mem_blocks.find(ptr) != mem_blocks.end();
}


double get_memory_usage()
{
  int total_size = 0;
  for(allocated_memory_blocks_t::iterator it = mem_blocks.begin(); it != mem_blocks.end(); it++) {
    //cout << it->second << endl;
    total_size += it->second.size;
  } // for it
  return double(total_size);
}

double get_max_memory_usage_mb()
{
  return double(max_allocated_size) / (1024.0L * 1024.0L);
}

std::string print_memory_usage()
{
  std::stringstream ss;
  ss << "cuda memory usage: " << double(get_memory_usage()) / (1024.0L * 1024.0L);
  return ss.str();
}


} // namespace memory_checker




