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


/*
 * What should be tested:
 * a) single node graph is a subgraph of another graph
 * b) random subgraph with single component test ... (i.e., implement DFS/BFS search in a graph and mark nodes. If all nodes are marked the do the test)
 * c) Let have a graph G. Add a random edge/node into the graph G, producing G', and test wheter G is subgraph of G'
 *
 */


/*
 * this test implements just a basic test, i.e., is-subgraph test on a hand-made graphs.
 */
TEST(graph, is_subgraph_basic_test)
{
  std::string triangle_str = "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0)";
  std::string square_with_diagonal_str = "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 0 -1 0 -1)";
  std::string line_str = "(0 1 0 0 0)";
  std::string double_line_str = "(0 1 0 0 0);(1 2 0 0 0)";
  std::string square_str = "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0)";

  types::DFSCode triangle_dfs = types::DFSCode::read_from_str(triangle_str);
  types::DFSCode square_with_diagonal_dfs = types::DFSCode::read_from_str(square_with_diagonal_str);
  types::DFSCode line_dfs = types::DFSCode::read_from_str(line_str);
  types::DFSCode double_line_dfs = types::DFSCode::read_from_str(double_line_str);
  types::DFSCode square_dfs = types::DFSCode::read_from_str(square_str);

  types::Graph triangle_grph;
  types::Graph square_with_diagonal_grph;
  types::Graph line_grph;
  types::Graph double_line_grph;
  types::Graph square_grph;

  triangle_dfs.toGraph(triangle_grph);
  square_with_diagonal_dfs.toGraph(square_with_diagonal_grph);
  line_dfs.toGraph(line_grph);
  double_line_dfs.toGraph(double_line_grph);
  square_dfs.toGraph(square_grph);


  // a) triangle     should be a subgraph of      square_with_diagonal.
  TRACE(*grphtst_log, "================================================================================");
  //cout << "square_with_diagonal.is_subgraph(triangle_grph): " << square_with_diagonal_grph.is_subgraph(triangle_dfs) << endl;
  //cout << "=============================================================================================================================" << endl;
  // a)
  EXPECT_TRUE(square_with_diagonal_grph.is_subgraph(triangle_dfs));

  //cout << "=============================================================================================================================" << endl;
  //cout << "triangle_grph.is_subgraph(line_dfs): " << triangle_grph.is_subgraph(line_dfs) << endl;
  //cout << "=============================================================================================================================" << endl;
  // b) line         should be a subgraph of      square, square_with_diagonal, triangle, double_line.
  TRACE(*grphtst_log, "triangle_grph.is_subgraph(line_dfs) meaning that triangle_grph has as a subgraph line_dfs");
  EXPECT_TRUE(triangle_grph.is_subgraph(line_dfs));
  TRACE(*grphtst_log, "square_with_diagonal_grph.is_subgraph(line_dfs)");
  EXPECT_TRUE(square_with_diagonal_grph.is_subgraph(line_dfs));
  TRACE(*grphtst_log, "triangle_grph.is_subgraph(line_dfs)");
  EXPECT_TRUE(triangle_grph.is_subgraph(line_dfs));
  TRACE(*grphtst_log, "double_line_grph.is_subgraph(line_dfs)");
  EXPECT_TRUE(double_line_grph.is_subgraph(line_dfs));


  // c) square       should be a subgraph of      square_with_diagonal.
  TRACE(*grphtst_log, "================================================================================");
  TRACE(*grphtst_log, "square_with_diagonal_grph.is_subgraph(square_dfs)");
  EXPECT_TRUE(square_with_diagonal_grph.is_subgraph(square_dfs));

  // d) double_line                               triangle, square, square_with_diagonal
  TRACE(*grphtst_log, "================================================================================");
  TRACE(*grphtst_log, "square_with_diagonal_grph.is_subgraph(square_dfs)");
  EXPECT_TRUE(square_with_diagonal_grph.is_subgraph(square_dfs));

  TRACE(*grphtst_log, "================================================================================");
  // triangle should not be a subgraph of  square, double_line.
  TRACE(*grphtst_log, "triangle_grph.is_subgraph(square_dfs): " << triangle_grph.is_subgraph(square_dfs));
  EXPECT_FALSE(triangle_grph.is_subgraph(square_dfs));
}


