#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

#include <graph_output.hpp>
#include <graph_types.hpp>
#include <logger.hpp>
#include <utils.hpp>
#include <dbio.hpp>


#include <test_support.hpp>

#include <graph_repair_tools.hpp>

using std::string;
using namespace types;
using namespace dbio;
using std::cerr;

using std::fixed;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");


typedef std::vector<types::DFSCode> dfs_vec_t;


dfs_vec_t read_file(const string &dfs_file)
{
  std::ifstream in;
  in.open(dfs_file.c_str(), std::ios::in);
  if(!in.is_open() || !in.good()) {
    CRITICAL_ERROR(*logger, "Could not open file: " << dfs_file);
    throw std::runtime_error("Error while opening some file.");
  }


  dfs_vec_t result;

  while(!in.eof()) {
    std::string line;
    std::getline(in, line);

    types::DFSCode dfs_code = types::DFSCode::read_from_str(line);
    result.push_back(dfs_code);
  } // while

  return result;
} // read_file


int bwd_edge_num(const types::DFSCode &dfs)
{
  int count = 0;
  for(int i = 0; i < dfs.size(); i++) {
    if(dfs[i].is_backward()) count++;
  } // for i

  return count;
} // bwd_edge_num


int main(int argc, char **argv)
{
  if(argc != 2) {
    std::cerr << "usage: \n" << argv[0] << " <filename>" << std::endl;
    return 1;
  }
  string filename = argv[1];

  INFO(*logger, "reading dfs codes");
  dfs_vec_t dfs_codes = read_file(filename);
  INFO(*logger, "reading dfs codes ... done");
  int max_bwd_edges = 0;
  double avg_bwd_edges = 0.0L;
  int max_pat_size = 0;
  double avg_pat_size = 0.0L;


  INFO(*logger, "computing statistics");
  for(int i = 0; i < dfs_codes.size(); i++) {
    int bwd_edges = bwd_edge_num(dfs_codes[i]);
    avg_bwd_edges += double(bwd_edges);
    if(max_bwd_edges < bwd_edges) max_bwd_edges = bwd_edges;

    if(max_pat_size < dfs_codes[i].size()) max_pat_size = dfs_codes[i].size();
    avg_pat_size += double(dfs_codes[i].size());
  } // for i

  avg_bwd_edges = avg_bwd_edges / double(dfs_codes.size());
  avg_pat_size = avg_pat_size / double(dfs_codes.size());

  INFO(*logger, "max_bwd_edges: " << max_bwd_edges);
  INFO(*logger, "avg_bwd_edges: " << avg_bwd_edges);
  INFO(*logger, "max_pat_size: " << max_pat_size);
  INFO(*logger, "avg_pat_size: " << avg_pat_size);
  INFO(*logger, "pattern count: " << dfs_codes.size());
  return 0;
}


