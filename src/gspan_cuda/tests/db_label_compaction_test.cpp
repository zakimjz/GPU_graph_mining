#include <string>
#include <iomanip>
#include <iostream>

#include <logger.hpp>
#include <utils.hpp>

#include <test_support.hpp>
#include <sys/time.h>

#include <cuda_utils.hpp>
#include <gtest/gtest.h>

using std::string;
using namespace types;
using std::cerr;

using std::fixed;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");


TEST(cuda_utils, db_label_compaction)
{
  types::graph_database_t gdb = gspan_cuda::get_labeled_test_database2();
  types::graph_database_cuda h_gdb = types::graph_database_cuda::create_from_host_representation(gdb);
  types::graph_database_cuda h_gdb_compacted = types::graph_database_cuda::create_from_host_representation(gdb);

  std::cout << "before: " << h_gdb.to_string() << std::endl;
  std::set<int> edge_label_set;
  std::set<int> vertex_label_set;
  gspan_cuda::compact_labels(h_gdb_compacted, vertex_label_set, edge_label_set);
  std::cout << "after: " << h_gdb_compacted.to_string() << std::endl;

  std::set<int> vertex_labels;
  std::set<int> edge_labels;
  // test whether the resulting sequence of labels is without gaps
  for(int i = 0; i < h_gdb_compacted.edges_sizes; i++) {
    if(h_gdb_compacted.edges_labels[i] == -1) continue;
    edge_labels.insert(h_gdb_compacted.edges_labels[i]);
  }


  for(int i = 0; i < h_gdb_compacted.db_size * h_gdb_compacted.max_graph_vertex_count; i++) {
    if(h_gdb_compacted.vertex_labels[i] == -1) continue;
    vertex_labels.insert(h_gdb_compacted.vertex_labels[i]);
  }

  // check whether the compaction maps every label to some other label using bijective mapping
  std::map<int, int> edge_map;
  for(int i = 0; i < h_gdb_compacted.edges_sizes; i++) {
    if(h_gdb_compacted.edges_labels[i] == -1) continue;
    if(edge_map.find(h_gdb_compacted.edges_labels[i]) == edge_map.end()) {
      edge_map.insert(std::make_pair(h_gdb_compacted.edges_labels[i], h_gdb.edges_labels[i]));
    }

    ASSERT_EQ(edge_map[h_gdb_compacted.edges_labels[i]], h_gdb.edges_labels[i]);
  }


  std::map<int, int> vertex_map;
  for(int i = 0; i < h_gdb_compacted.db_size * h_gdb_compacted.max_graph_vertex_count; i++) {
    if(h_gdb_compacted.vertex_labels[i] == -1) continue;

    if(vertex_map.find(h_gdb_compacted.vertex_labels[i]) == vertex_map.end()) {
      vertex_map.insert(std::make_pair(h_gdb_compacted.vertex_labels[i], h_gdb.vertex_labels[i]));
    }

    ASSERT_EQ(vertex_map[h_gdb_compacted.vertex_labels[i]], h_gdb.vertex_labels[i]);
  }


}


