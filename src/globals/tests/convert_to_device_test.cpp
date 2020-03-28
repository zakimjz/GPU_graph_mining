#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <test_support.hpp>

using namespace types;
using std::string;

Logger *logger = Logger::get_logger("CUDA_GDB_TEST");


void perform_test(const types::graph_database_t &gdb, bool copy_roundtrip = false)
{
  graph_database_cuda h_gdb = graph_database_cuda::create_from_host_representation(gdb);

  graph_database_cuda h_new_gdb(true);
  // maybe copy it to device and back
  if(copy_roundtrip) {
    graph_database_cuda d_gdb(false);
    h_gdb.copy_to_device(&d_gdb);
    h_gdb.delete_from_host();

    h_new_gdb.copy_from_device(&d_gdb);
    d_gdb.delete_from_device();
  } else {
    h_new_gdb = h_gdb;
  }

  // reconstruct the graph database back from the array representation
  types::graph_database_t gdb_from_cuda;
  h_new_gdb.convert_to_host_representation(gdb_from_cuda);


  ASSERT_EQ(gdb.size(), h_new_gdb.db_size);

  for(int i = 0; i < gdb.size(); i++) {
    string gdb_string;
    string gdb_from_cuda_string;

    gdb_string = gdb[i].to_string();
    gdb_from_cuda_string = gdb_from_cuda[i].to_string();
    if(gdb_string != gdb_from_cuda_string) {
      INFO(*logger, "graphs at position  " << i << " are not isomorphic");
    }
    ASSERT_EQ(gdb_string, gdb_from_cuda_string);
  } // for i
} // perform_test


TEST(cuda_gdb, convert_simple_test) {
  types::graph_database_t gdb = gspan_cuda::get_test_database();
  perform_test(gdb);
  perform_test(gdb, true);


  types::graph_database_t gdb_labeled = gspan_cuda::get_labeled_test_database();
  perform_test(gdb_labeled);
  perform_test(gdb_labeled, true);
}


