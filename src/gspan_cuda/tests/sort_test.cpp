#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <test_support.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>
#include <cuda_datastructures.hpp>

#include <cuda_gspan_ops.hpp>
#include <cuda_tools.hpp>
#include <algorithm>

#include <thrust/sort.h>

using namespace types;
using std::string;

using gspan_cuda::create_first_embeddings;
using gspan_cuda::get_all_extensions;
using gspan_cuda::extension_element_t;
using namespace gspan_cuda;

static Logger *logger = Logger::get_logger("CUDA_FE_TEST");

struct embedding_extension_compare_less_then_t_kesslr {
  types::DFS_less_then comp;
  bool operator()(const embedding_extension_t &ee1, const embedding_extension_t &ee2) const {
    bool less = comp(ee1.dfs_elem, ee2.dfs_elem);
    return less;
  } // operator()
};


struct embedding_extension_compare_less_then_t {
  dfs_comparator<types::DFS> comp;
  bool operator()(const embedding_extension_t &ee1, const embedding_extension_t &ee2) const {
    bool less = comp(ee1.dfs_elem, ee2.dfs_elem) < 0 ? true : false;
    return less;
  } // operator()
};


static bool get_comparison(const string &dfs1, const string &dfs2)
{
  embedding_extension_t ee1(true);
  ee1.dfs_elem = types::DFS::parse_from_string(dfs1.c_str());
  embedding_extension_t ee2(true);
  ee2.dfs_elem = types::DFS::parse_from_string(dfs2.c_str());

  //embedding_extension_compare_less_then_t comp;
  embedding_extension_compare_less_then_t_kesslr comp;
  return comp(ee1, ee2);
}

static void basic_test1()
{
  string dfs_elems_str[] = {
    "(14 6 0 3 0)",
    "(14 15 0 0 1)",
    "(5 15 9 0 9)",
    "(4 15 0 3 0)",
    "(0 15 0 3 0)",
    ""
  };

  int max = 5;
  for(int i = 0; i < max; i++) {
    for(int j = 0; j < max; j++) {
      if(i < j) {
        DEBUG(*logger, dfs_elems_str[i] << " < " << dfs_elems_str[j]);
        ASSERT_TRUE(get_comparison(dfs_elems_str[i], dfs_elems_str[j]));
      } else if(i > j) {
        DEBUG(*logger, dfs_elems_str[j] << " < " << dfs_elems_str[i]);
        ASSERT_FALSE(get_comparison(dfs_elems_str[i], dfs_elems_str[j]));
      }
    } // for j
  } // for i
}

static void basic_test2()
{
  string dfs_elems_str[] = {
    "(2 3 0 0 1)",
    "(2 3 0 0 2)",
    "(2 3 0 0 4)",
    "(2 3 0 0 7)",
    "(2 3 0 0 8)",
    "(2 3 0 0 9)",
    "(2 3 0 0 13)",
    "(2 3 0 0 16)",
    "(2 3 0 3 0)",
    "(0 3 0 3 0)",
    ""
  };

  int max = 10;
  for(int i = 0; i < max; i++) {
    for(int j = 0; j < max; j++) {
      if(i < j) {
        DEBUG(*logger, dfs_elems_str[i] << " < " << dfs_elems_str[j]);
        ASSERT_TRUE(get_comparison(dfs_elems_str[i], dfs_elems_str[j]));
      } else if(i > j) {
        DEBUG(*logger, dfs_elems_str[j] << " < " << dfs_elems_str[i]);
        ASSERT_FALSE(get_comparison(dfs_elems_str[i], dfs_elems_str[j]));
      }
    } // for j
  } // for i

}

TEST(sort_test, basic_test)
{
  basic_test1();
  basic_test2();
}


