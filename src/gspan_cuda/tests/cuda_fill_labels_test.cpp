#include <logger.hpp>
#include <dbio.hpp>
#include <gspan_cuda_no_sort.hpp>
#include <gspan_cuda.hpp>
#include <test_support.hpp>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <cuda_graph_types.hpp>
#include <cuda_utils.hpp>

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");


using namespace types;

class fill_labels_test_class : public gspan_cuda::gspan_cuda_no_sort {
  Logger *logger;
public:
  fill_labels_test_class() {
    logger = Logger::get_logger("FILL_TESTS");
  }


  void do_fill_labels_test();
  void do_fill_root_test();

  void contains_labels(const types::edge_gid_list3_t &root,
                       const types::edge_gid_list3_t &check_sizes_root,
                       int from_label, int elabel, int to_label);

  void print_root(const types::edge_gid_list3_t &root);

};



void fill_labels_test_class::print_root(const types::edge_gid_list3_t &root)
{
  DEBUG(*logger, "printing edges in root: ");
  for(types::edge_gid_list3_t::const_iterator fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
    for(types::edge_gid_list2_t::const_iterator elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
      for(types::edge_gid_list1_t::const_iterator tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
        DEBUG(*logger, "edge: (" << fromlabel->first << ", " << elabel->first << ", " << tolabel->first << ")");
      } // for tolabel
    } // for elabel
  } // for fromlabel
} // fill_labels_test_class::print_root



void fill_labels_test_class::contains_labels(const types::edge_gid_list3_t &root,
                                             const types::edge_gid_list3_t &check_sizes_root,
                                             int from_label, int elabel, int to_label)
{
  types::edge_gid_list3_t::const_iterator fromlabel_it = root.find(from_label);
  ASSERT_FALSE(fromlabel_it == root.end());

  types::edge_gid_list3_t::const_iterator cs_fromlabel_it = check_sizes_root.find(from_label);
  ASSERT_FALSE(cs_fromlabel_it == check_sizes_root.end());
  if(cs_fromlabel_it->second.size() != fromlabel_it->second.size()) {
    DEBUG(*logger, "error for edge: (" << from_label << ", " << elabel << ", " << to_label << ")\n\n");
  }
  ASSERT_EQ(cs_fromlabel_it->second.size(), fromlabel_it->second.size());

  /////////////////////////////////////////////////////////////////////////////////////
  types::edge_gid_list2_t::const_iterator elabel_it = fromlabel_it->second.find(elabel);
  ASSERT_FALSE(elabel_it == fromlabel_it->second.end());

  types::edge_gid_list2_t::const_iterator cs_elabel_it = cs_fromlabel_it->second.find(elabel);
  ASSERT_FALSE(cs_elabel_it == cs_fromlabel_it->second.end());

  ASSERT_EQ(cs_elabel_it->second.size(), elabel_it->second.size());

  /////////////////////////////////////////////////////////////////////////////////////
  types::edge_gid_list1_t::const_iterator tolabel_it = elabel_it->second.find(to_label);
  ASSERT_FALSE(tolabel_it == elabel_it->second.end());

  types::edge_gid_list1_t::const_iterator cs_tolabel_it = cs_elabel_it->second.find(to_label);
  ASSERT_FALSE(cs_tolabel_it == cs_elabel_it->second.end());

  //ASSERT_EQ(cs_tolabel_it->second.size(), tolabel_it->second.size());
}



void fill_labels_test_class::do_fill_root_test()
{
  types::edge_gid_list3_t cuda_root;
  types::edge_gid_list3_t host_root;
  //fill_root_cuda(cuda_root);
  prepare_run(cuda_root);
  fill_root(host_root);


  print_root(cuda_root);
  print_root(host_root);

  types::edge_gid_list3_t::iterator it;

  ASSERT_EQ(cuda_root.size(), host_root.size());

  for(types::edge_gid_list3_t::iterator fromlabel_it = host_root.begin(); fromlabel_it != host_root.end(); ++fromlabel_it) {
    for(types::edge_gid_list2_t::iterator elabel_it = fromlabel_it->second.begin(); elabel_it != fromlabel_it->second.end(); ++elabel_it) {
      for(types::edge_gid_list1_t::iterator tolabel_it = elabel_it->second.begin(); tolabel_it != elabel_it->second.end(); ++tolabel_it) {
        int from_label = fromlabel_it->first;
        int elabel = elabel_it->first;
        int tolabel = tolabel_it->first;
        contains_labels(host_root, cuda_root, from_label, elabel, tolabel);

        unsigned int host_supp = 0;
        int last_id = -1;
        for(int i = 0; i < tolabel_it->second.size(); i++) {
          if(tolabel_it->second[i] != last_id) {
            last_id = tolabel_it->second[i];
            host_supp++;
          }
        }
        unsigned int cuda_supp = compute_support(from_label, elabel, tolabel);
        ASSERT_EQ(cuda_supp, host_supp);
      } // for tolabel
    } // for elabel
  } // for fromlabel
} // fill_labels_test_class::do_fill_root_test



void fill_labels_test_class::do_fill_labels_test()
{
  fill_labels();
  std::set<int> host_edges = edge_label_set;
  std::set<int> host_vertices = vertex_label_set;
  edge_label_set.clear();
  vertex_label_set.clear();

  fill_labels_cuda();
  std::set<int> cuda_edges = edge_label_set;
  std::set<int> cuda_vertices = vertex_label_set;

  DEBUG(*logger, "counts; host edges labels: " << host_edges.size() << "; cuda edges labels: " << cuda_edges.size());
  DEBUG(*logger, "counts; host vertex labels: " << host_vertices.size() << "; cuda edges labels: " << cuda_vertices.size());

  if(host_edges != cuda_edges) {
    CRITICAL_ERROR(*logger, "edges are not the same");
    ASSERT_TRUE(false);
  } else {
    ASSERT_TRUE(true);
  }

  if(host_vertices != cuda_vertices) {
    CRITICAL_ERROR(*logger, "vertices are not the same");
    ASSERT_TRUE(false);
  } else {
    ASSERT_TRUE(true);
  }

  //ASSERT_EQ(host_edges, cuda_edges);
  //ASSERT_EQ(host_vertices, cuda_vertices);
} // fill_labels_test_class::do_fill_labels_test



static void fill_labels_test(const types::graph_database_t &database, int absolute_min_support, cudapp::cuda_configurator exec_config)
{
  timeval start_time, stop_time;
  gettimeofday(&start_time, 0);

  graph_counting_output *count_out = new graph_counting_output();
  graph_output *current_out = count_out;
  fill_labels_test_class gspan;

  types::graph_database_cuda h_graph_db = types::graph_database_cuda::create_from_host_representation(database);
  std::set<int> vertex_label_set;
  std::set<int> edge_label_set;

  gspan_cuda::compact_labels(h_graph_db, vertex_label_set, edge_label_set);
  gspan.set_database(h_graph_db);

  gspan.set_min_support(absolute_min_support);
  gspan.set_graph_output(current_out);
  gspan.set_exec_configurator(&exec_config);
  gspan.do_fill_labels_test();
  gspan.do_fill_root_test();

  gettimeofday(&stop_time, 0);

  LOG_VALUE(*value_logger, "total_time", std::fixed << utils::get_time_diff(start_time, stop_time));
  LOG_VALUE(*value_logger, "total_frequent_graphs", current_out->get_count());

  gspan.delete_database_from_device();
  delete count_out;
}




TEST(cuda_fill_labels, basic_test)
{
  cudaDeviceReset();

  cudapp::cuda_configurator exec_config;
  exec_config.set_parameter("create_first_embeddings.valid_vertex", 100);
  exec_config.set_parameter("create_first_embeddings.store_map", 100);
  exec_config.set_parameter("create_first_embeddings.find_valid_edge", 100);
  exec_config.set_parameter("create_first_embeddings.edge_store", 100);
  exec_config.set_parameter("compute_extensions.store_validity", 100);
  exec_config.set_parameter("compute_extensions.exec_extension_op", 100);
  exec_config.set_parameter("compute_support.boundaries", 100);
  exec_config.set_parameter("filter_backward_embeddings.store_col_indices", 100);
  exec_config.set_parameter("filter_backward_embeddings.copy_rows", 100);
  exec_config.set_parameter("fwd_fwd.compute_extension", 100);


  types::graph_database_t db = gspan_cuda::get_labeled_test_database2();
  fill_labels_test(db, 1, exec_config);

  db = gspan_cuda::get_labeled_test_database();
  fill_labels_test(db, 1, exec_config);
  memory_checker::detect_memory_leaks();
  Logger::free_loggers();
}


