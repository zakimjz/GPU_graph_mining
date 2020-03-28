#include <dfs_code.hpp>
#include <string>
#include <gtest/gtest.h>
#include <stdexcept>
#include <dbio.hpp>
#include <logger.hpp>
#include <cuda_graph_types.hpp>

using std::string;
using std::runtime_error;
using types::DFS;

using namespace dbio;

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
  while(pbec_prefixes[idx].size() != 0) {
    types::DFSCode dfs = types::DFSCode::read_from_str(pbec_prefixes[idx]);
    types::Graph grph;
    dfs.toGraph(grph);
    result.push_back(grph);
    idx++;
  } // while

  return result;
} // get_test_database

TEST(dbio, basic_io_bin_test)
{
  types::graph_database_t db = get_test_database();
  types::graph_database_t read_db;

  TRACE5(*Logger::get_logger("GLOBALSTEST"), "test database size (in # of graphs): " << db.size());

  write_database_bin("test.dat", db);
  read_database_bin("test.dat", read_db);

  ASSERT_EQ(db.size(), read_db.size());

  for(int i = 0; i < read_db.size(); i++) {
    std::string read_db_dfs_code = read_db[i].get_min_dfs_code().to_string();
    std::string orig_db_dfs_code = db[i].get_min_dfs_code().to_string();
    ASSERT_EQ(read_db_dfs_code, orig_db_dfs_code);
  } // for i
} // TEST dbio.basic_io_bin_test


static void perform_basic_io_bin_part_test(const types::graph_database_t &db, int total_parts)
{
  types::graph_database_t *read_db = new types::graph_database_t[total_parts];
  for(int i = 0; i < total_parts; i++) {
    read_database_bin("test.dat", read_db[i], total_parts, i);
  }

  int db_idx = 0;
  for(int i = 0; i < total_parts; i++) {
    for(int j = 0; j < read_db[i].size(); j++) {
      //std::cout << "comparing; read: " << read_db[i][j].to_string() << "; stored: " << db[db_idx].to_string() << std::endl;
      ASSERT_EQ(read_db[i][j].to_string(), db[db_idx].to_string());
      db_idx++;
    }  // for j
  }   // for i
}



TEST(dbio, basic_io_bin_part_test)
{
  types::graph_database_t db = get_test_database();

  TRACE5(*Logger::get_logger("GLOBALSTEST"), "test database size (in # of graphs): " << db.size());

  write_database_bin("test.dat", db);
  for(int parts = 1; parts < db.size(); parts++) {
    perform_basic_io_bin_part_test(db, parts);
  }
} // TEST dbio.basic_io_bin_test





TEST(dbio, basic_io_cuda)
{
  types::graph_database_t db = get_test_database();

  TRACE(*Logger::get_logger("GLOBALSTEST"), "test database size (in # of graphs): " << db.size());

  types::graph_database_cuda cuda_gdb = types::graph_database_cuda::create_from_host_representation(db);
  types::graph_database_cuda cuda_gdb_read(true);

  write_database_cudabin("test.dat", cuda_gdb);
  read_database_cudabin("test.dat", cuda_gdb_read);


  types::graph_database_t db_read;
  cuda_gdb_read.convert_to_host_representation(db_read);

  TRACE(*Logger::get_logger("GLOBALSTEST"), "read database size (in # of graphs): " << db_read.size());

  ASSERT_EQ(db_read.size(), db.size());
  for(int i = 0; i < db_read.size(); i++) {
    if(db_read[i].get_min_dfs_code() == db[i].get_min_dfs_code()) {
      ASSERT_TRUE(true);
    } else {
      ASSERT_TRUE(false);
    }
  } // for i

  cuda_gdb_read.delete_from_host();
  cuda_gdb.delete_from_host();
} // TEST dbio.basic_io_bin_test




