#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <string>
#include <vector>

namespace types {

  class Graph;
  class Edge;

  typedef int integer_t;
  typedef integer_t int_t;
  typedef unsigned int unsigned_integer_t;
  typedef unsigned_integer_t uint_t;
  typedef uint_t symbol_t;
  typedef double double_t;
  typedef char char_t;
  typedef char * charp_t;
  typedef float float_t;
  typedef long  long_t;
  typedef unsigned long unsigned_long_t;
  typedef unsigned long ulong_t;
  typedef bool bool_t;
  typedef std::string string_t;

  typedef void * void_ptr_t;


  typedef std::vector<Graph> graph_database_t;
  typedef std::vector<int>   RMPath;
  typedef std::vector<Edge *> EdgeList;
} // namespace types

#endif

