#include <graph_types.hpp>
#include <string>
#include <gtest/gtest.h>
#include <stdexcept>
#include <graph_repair_tools.hpp>

using std::string;
using std::runtime_error;


types::Graph get_grph_with_selfloop()
{
  types::Graph grph;

  types::Vertex v0;
  v0.label = 0;

  types::Vertex v1;
  v1.label = 0;


  types::Vertex v2;
  v2.label = 0;

  grph.push_back(v0);
  grph.push_back(v1);
  grph.push_back(v2);

  // 0-1
  grph[0].push(0, 1, 0);
  grph[1].push(1, 0, 0);

  // 1-2
  grph[1].push(1, 2, 0);
  grph[2].push(2, 1, 0);

  // self-loop 2-2
  grph[2].push(2, 2, 0);
  grph[2].push(2, 2, 0);

  return grph;
}


types::Graph get_grph_with_parallel_edges()
{
  types::Graph grph;

  types::Vertex v0;
  v0.label = 0;

  types::Vertex v1;
  v1.label = 0;


  types::Vertex v2;
  v2.label = 0;

  grph.push_back(v0);
  grph.push_back(v1);
  grph.push_back(v2);

  //0-1
  grph[0].push(0, 1, 0);
  grph[1].push(1, 0, 0);

  // 1-2
  grph[1].push(1, 2, 0);
  grph[2].push(2, 1, 0);
  // dup 1-2
  grph[1].push(1, 2, 0);
  grph[2].push(2, 1, 0);

  return grph;
}




types::Graph get_grph_with_parallel_edges_diff_label()
{
  types::Graph grph;

  types::Vertex v0;
  v0.label = 0;

  types::Vertex v1;
  v1.label = 0;


  types::Vertex v2;
  v2.label = 0;

  grph.push_back(v0);
  grph.push_back(v1);
  grph.push_back(v2);

  //0-1
  grph[0].push(0, 1, 0);
  grph[1].push(1, 0, 0);

  // 1-2
  grph[1].push(1, 2, 0);
  grph[2].push(2, 1, 0);
  // dup 1-2
  grph[1].push(1, 2, 1);
  grph[2].push(2, 1, 1);

  return grph;
}


TEST(graph, check_graph)
{
  types::Graph self_loop_grph = get_grph_with_selfloop();
  EXPECT_ANY_THROW(check_graph(self_loop_grph));

  types::Graph parallel_edge_grph = get_grph_with_parallel_edges();
  EXPECT_ANY_THROW(check_graph(parallel_edge_grph));


  types::Graph parallel_edge_grph_diff_label = get_grph_with_parallel_edges_diff_label();
  EXPECT_ANY_THROW(check_graph(parallel_edge_grph_diff_label));
}


TEST(graph, fix_graph)
{
  types::Graph self_loop_grph = get_grph_with_selfloop();
  types::Graph parallel_edge_grph = get_grph_with_parallel_edges();
  types::Graph parallel_edge_grph_diff_label = get_grph_with_parallel_edges_diff_label();


  EXPECT_NO_THROW(fix_graph(self_loop_grph));
  EXPECT_NO_THROW(check_graph(self_loop_grph));


  EXPECT_NO_THROW(fix_graph(parallel_edge_grph));
  EXPECT_NO_THROW(check_graph(parallel_edge_grph));


  EXPECT_ANY_THROW(fix_graph(parallel_edge_grph_diff_label));
  EXPECT_ANY_THROW(check_graph(parallel_edge_grph_diff_label));

}


