#ifndef __GRAPH_OUTPUT_HPP__
#define __GRAPH_OUTPUT_HPP__


#include <graph_types.hpp>
#include <fstream>


class graph_output {
public:
  virtual void output_graph(const types::Graph &g, size_t support = 0) = 0;
  virtual void output_graph(const types::DFSCode &code, size_t support = 0) = 0;
  virtual double get_count() = 0;
  virtual std::string to_string() = 0;
  virtual ~graph_output();
};


class graph_counting_output : public graph_output {
  double counter;
public:
  graph_counting_output();
  virtual void output_graph(const types::Graph &g, size_t support = 0);
  virtual void output_graph(const types::DFSCode &code, size_t support = 0);
  virtual double get_count();
  virtual std::string to_string();
};


class graph_database_output : public graph_output {
  types::graph_database_t gdb;
public:
  graph_database_output();
  virtual void output_graph(const types::Graph &g, size_t support = 0);
  virtual void output_graph(const types::DFSCode &code, size_t support = 0);
  virtual double get_count();
  virtual std::string to_string();
};



class graph_file_output : public graph_output {
  std::ostream *out;
  std::string filename;
  double count;
public:
  graph_file_output(const std::string &fname);
  virtual void output_graph(const types::Graph &g, size_t support = 0);
  virtual void output_graph(const types::DFSCode &code, size_t support = 0);
  virtual double get_count();
  virtual std::string to_string();
};

#endif

