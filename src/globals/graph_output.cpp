#include <graph_output.hpp>
#include <sstream>
#include <fstream>

using std::stringstream;



graph_output::~graph_output()
{
}



void graph_counting_output::output_graph(const types::Graph &g, size_t support) {
  counter += 1.0L;
} // output_sequence


void graph_counting_output::output_graph(const types::DFSCode &code, size_t support)
{
  counter += 1.0L;
}

graph_counting_output::graph_counting_output() {
  counter = 0;
}

std::string graph_counting_output::to_string() {
  std::stringstream ss;
  ss << counter;
  return ss.str();
} // to_string


double graph_counting_output::get_count()
{
  return counter;
}





graph_database_output::graph_database_output()
{
}

void graph_database_output::output_graph(const types::Graph &g, size_t support)
{
  gdb.push_back(g);
}

void graph_database_output::output_graph(const types::DFSCode &code, size_t support)
{
  abort();
}


std::string graph_database_output::to_string()
{
  stringstream ss;
  ss << "graph_database_output, size: " << gdb.size();
  return ss.str();
}

double graph_database_output::get_count()
{
  return double(gdb.size());
}




graph_file_output::graph_file_output(const std::string &fname) {
  filename = fname;
  out = new std::ofstream();
  ((std::ofstream*)out)->open(filename.c_str(), std::ios::out | std::ios::trunc);
}


void graph_file_output::output_graph(const types::Graph &g, size_t support)
{
  types::DFSCode dfs = g.get_min_dfs_code();
  output_graph(dfs);
  //(*out) << g.to_string().c_str() << std::endl;
  //count += 1.0L;
}

void graph_file_output::output_graph(const types::DFSCode &code, size_t support)
{
  //abort();
  (*out) << code.to_string(false) << std::endl;
  count += 1.0L;
}


double graph_file_output::get_count()
{
  return count;
}

std::string graph_file_output::to_string()
{
  stringstream ss;
  ss << "graph_file_output, filename: " << filename;
  return ss.str();
}


