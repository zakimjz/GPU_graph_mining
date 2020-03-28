#include <dfs_code.hpp>
#include <string>
#include <gtest/gtest.h>
#include <stdexcept>
#include <graph_types.hpp>

using std::string;
using std::runtime_error;
using types::DFS;
using types::DFSCode;
using types::Graph;


TEST(graph, basic_serialization)
{
  string grph_dfscode_str = "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 0 -1 0 -1)"; // is min
  DFSCode grph_dfs = DFSCode::read_from_str(grph_dfscode_str);
  Graph grph_orig;
  grph_dfs.toGraph(grph_orig);

  char *buffer = 0;
  int buff_size = 0;


  int min_buff_size = Graph::get_serialized_size(grph_orig);

  EXPECT_THROW(Graph::serialize(grph_orig, buffer, buff_size), runtime_error);
  buffer = new char[min_buff_size];
  buff_size = min_buff_size - 1;
  EXPECT_THROW(Graph::serialize(grph_orig, buffer, buff_size), runtime_error);


  buff_size = min_buff_size;
  int written = Graph::serialize(grph_orig, buffer, buff_size);
  EXPECT_EQ(written, buff_size);


  Graph grph_deser;
  int read = Graph::deserialize(grph_deser, buffer, buff_size);
  EXPECT_EQ(read, written);
  DFSCode grph_deser_dfs = grph_deser.get_min_dfs_code();
  EXPECT_EQ(grph_deser_dfs, grph_dfs);

  delete [] buffer;
}


extern types::graph_database_t get_test_database();
TEST(graph_database, basic_serialization_test)
{
  types::graph_database_t gdb = get_test_database();

  size_t gdb_ser_size = Graph::get_serialized_size(gdb);
  char *buffer = new char[gdb_ser_size];
  size_t written_bytes = Graph::serialize(gdb, buffer, gdb_ser_size);

  ASSERT_EQ(written_bytes, gdb_ser_size);

  types::graph_database_t deser_gdb;
  size_t read_bytes = Graph::deserialize(deser_gdb, buffer, written_bytes);

  ASSERT_EQ(read_bytes, written_bytes);
  ASSERT_EQ(gdb.size(), deser_gdb.size());

  for(int i = 0; i < gdb.size(); i++) {
    std::string gdb_dfs = gdb[i].to_string();
    std::string deser_gdb_dfs = deser_gdb[i].to_string();
    ASSERT_EQ(gdb_dfs, deser_gdb_dfs);
  } // for i
  delete [] buffer;
} // TEST(graph_database, basic_serialization_test)



