#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>

using namespace types;
using std::string;

namespace gspan_cuda {

types::graph_database_t get_test_database()
{
  std::string pbec_prefixes[] = {
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(0 4 0 0 0)", // prefix 0
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(1 4 0 0 0)", // prefix 0
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 1
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 1
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 2 0 0 0)", // prefix 1
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 0 0 0 0)", // prefix 2
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 1 0 0 0)", // prefix 2
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(1 4 0 0 0);(4 5 0 0 0)", // prefix 3
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(1 4 0 0 0);(4 0 0 0 0)", // prefix 3
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0)", // prefix 4
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 4
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 4
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 5
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 5
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0)", // prefix 5
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 5 0 0 0)", // prefix 6
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 0 0 0 0)", // prefix 6
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 1 0 0 0)", // prefix 6
    ""
  };

  types::graph_database_t result;

  int idx = 0;
  int total_edge_count = 0;
  while(pbec_prefixes[idx].size() != 0) {
    types::DFSCode dfs = types::DFSCode::read_from_str(pbec_prefixes[idx]);
    total_edge_count += dfs.size();
    types::Graph grph;
    dfs.toGraph(grph);

    unsigned int tmp = grph.edge_size();
    unsigned int tmp2 = dfs.size();
    //EXPECT_EQ(grph.edge_size(), dfs.size());
    assert(grph.edge_size() == dfs.size());

    result.push_back(grph);
    idx++;
  } // while

  return result;
} // get_test_database





types::graph_database_t get_labeled_test_database()
{
  std::string pbec_prefixes[] = {
    "(0 1 1 0 0);(1 2 0 0 0);(2 0 0 0 1);(2 3 1 0 0);(3 0 0 2 0);(0 4 0 0 0)", // prefix 0   0
    "(0 1 1 0 0);(1 2 0 0 0);(2 0 0 0 1);(2 3 1 0 0);(3 0 0 2 0);(1 4 0 0 0)", // prefix 0   1
    "(0 1 1 0 1);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 1   2
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 1 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 1   3
    "(0 1 0 0 0);(1 2 1 0 1);(2 0 0 0 0);(2 3 0 1 0);(3 4 0 0 0);(4 2 0 0 0)", // prefix 1   4
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 0 0 0 0)", // prefix 2   5
    "(0 1 0 0 0);(1 2 0 1 0);(2 0 0 2 0);(2 3 0 0 0);(2 4 0 0 0);(4 1 0 0 0)", // prefix 2   6
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 2 0);(2 3 0 2 0);(1 4 0 0 0);(4 5 0 0 0)", // prefix 3   7
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(1 4 0 0 0);(4 0 0 0 0)", // prefix 3   8
    "(0 1 0 0 1);(1 2 1 1 0);(2 3 0 1 1);(3 0 1 0 0);(3 4 0 0 0)", // prefix 4               9
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 4   10
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 4   11
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 5               12
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 5               13
    "(0 1 1 1 1);(1 2 1 0 0);(2 3 0 0 1);(3 4 1 0 0)", // prefix 5                           14
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 5 0 0 0)", // prefix 6               15
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 2 0);(4 0 0 0 0)", // prefix 6               16
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(4 1 0 0 0)", // prefix 6               17
    ""
  };

  types::graph_database_t result;

  int idx = 0;
  int total_edge_count = 0;
  while(pbec_prefixes[idx].size() != 0) {
    types::DFSCode dfs = types::DFSCode::read_from_str(pbec_prefixes[idx]);
    total_edge_count += dfs.size();
    types::Graph grph;
    dfs.toGraph(grph);

    unsigned int tmp = grph.edge_size();
    unsigned int tmp2 = dfs.size();
    //EXPECT_EQ(grph.edge_size(), dfs.size());
    assert(grph.edge_size() == dfs.size());

    result.push_back(grph);
    idx++;
  } // while

  return result;
} // get_test_database




types::graph_database_t get_labeled_test_database2()
{
  std::string pbec_prefixes[] = {
    "(0 1 1 0 0);(1 2 0 0 0);(2 0 0 0 1);(2 3 1 0 0);(3 0 0 2 0);(0 4 0 0 10)", // prefix 0   0
    "(0 1 1 0 0);(1 2 0 0 0);(2 0 0 0 1);(2 3 1 0 0);(3 0 0 2 0);(1 4 0 0 20)", // prefix 0   1
    "(0 1 1 0 1);(1 2 0 170 0);(2 0 0 180 0);(2 3 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 1   2
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 1 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 1   3
    "(0 1 0 0 0);(1 2 1 0 1);(2 0 0 0 0);(2 3 0 1 0);(3 4 0 0 0);(4 2 0 0 0)", // prefix 1   4
    "(0 1 0 0 0);(1 2 0 0 30);(2 0 30 0 0);(2 3 30 0 0);(2 4 30 0 0);(4 0 0 0 0)", // prefix 2   5
    "(0 1 0 0 0);(1 2 0 1 0);(2 0 0 2 0);(2 3 0 0 0);(2 4 0 0 0);(4 1 0 0 0)", // prefix 2   6
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 2 0);(2 3 0 2 0);(1 4 0 0 0);(4 5 0 0 0)", // prefix 3   7
    "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(1 4 0 0 0);(4 0 0 0 0)", // prefix 3   8
    "(0 1 0 200 100);(1 2 100 1 40);(2 3 40 1 1);(3 0 1 0 0);(3 4 0 0 0)", // prefix 4               9
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 4   10
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 0 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 4   11
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 0 0 0 0)", // prefix 5               12
    "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 4 0 0 0);(4 1 0 0 0)", // prefix 5               13
    "(0 1 1 1 1);(1 2 1 0 0);(2 3 0 0 1);(3 4 1 0 0)", // prefix 5                           14
    "(0 1 0 120 0);(1 2 0 100 0);(2 3 0 0 0);(2 4 0 0 0);(4 5 0 0 0)", // prefix 6               15
    "(0 1 0 130 0);(1 2 0 110 0);(2 3 0 0 0);(2 4 0 2 0);(4 0 0 0 0)", // prefix 6               16
    "(0 1 0 160 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 120 0);(4 1 0 0 0)", // prefix 6               17
    ""
  };

  types::graph_database_t result;

  int idx = 0;
  int total_edge_count = 0;
  while(pbec_prefixes[idx].size() != 0) {
    types::DFSCode dfs = types::DFSCode::read_from_str(pbec_prefixes[idx]);
    total_edge_count += dfs.size();
    types::Graph grph;
    dfs.toGraph(grph);

    unsigned int tmp = grph.edge_size();
    unsigned int tmp2 = dfs.size();
    //EXPECT_EQ(grph.edge_size(), dfs.size());
    assert(grph.edge_size() == dfs.size());

    result.push_back(grph);
    idx++;
  } // while

  return result;
} // get_test_database


} // namespace gspan_cuda


