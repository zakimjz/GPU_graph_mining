#include <utils.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <iostream>
#include <logger.hpp>
#include <graph_types.hpp>

using utils::split;
using std::string;
using std::runtime_error;
using types::DFS;
using types::DFSCode;
using types::Graph;

static Logger *grphtst_log = Logger::get_logger("GLOBALSTEST");


TEST(graph, get_min_dfs_basic_test)
{
  string graph_not_min_dfs_code_str = "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 1 -1 0 -1)"; // is not min
  string graph_min_dfs_code_str = "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 0 -1 0 -1)"; // is min

  DFSCode graph_not_min_dfs_code = DFSCode::read_from_str(graph_not_min_dfs_code_str);
  DFSCode graph_min_dfs_code = DFSCode::read_from_str(graph_min_dfs_code_str);


  Graph g;
  graph_not_min_dfs_code.toGraph(g);

  DFSCode min_dfs_code_g = g.get_min_dfs_code();

  DEBUG(*grphtst_log, "min_dfs_code_g: " << min_dfs_code_g.to_string());

} // TEST


