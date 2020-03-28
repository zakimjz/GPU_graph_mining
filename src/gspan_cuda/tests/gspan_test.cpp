#include <string>
#include <iomanip>
#include <iostream>

#include <graph_output.hpp>
#include <graph_types.hpp>
#include <logger.hpp>
#include <utils.hpp>
#include <dbio.hpp>

#include <memory_checker.hpp>
#include <gspan_cuda.hpp>
#include <test_support.hpp>
#include <gspan.hpp>

#include <gspan_cuda_lists.hpp>
#include <gspan_cuda_no_sort.hpp>
#include <gspan_cuda_no_sort_block.hpp>
#include <gspan_cuda_mult_block.hpp>

#include <cuda_utils.hpp>

#include <sys/time.h>


using std::string;
using namespace types;
using namespace dbio;
using std::cerr;

using std::fixed;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");

int execute_host(graph_database_t &database, int absolute_min_support)
{
  INFO(*logger, "execute_host");
  if(absolute_min_support >= database.size()) {
    cerr << "error: user specified support " << absolute_min_support << " is greater then the database size " << database.size() << endl;
    return 2;
  }


  if(absolute_min_support < 1) {
    cerr << "error: absolute_min_support < 1" << endl;
    return 3;
  }

  graph_counting_output *count_out = new graph_counting_output();
  graph_output *current_out = count_out;

  timeval start_time, stop_time;
  gettimeofday(&start_time, 0);

  //LOG_VAL(*value_logger, absolute_min_support);
  //LOG_VALUE(*value_logger, "output", current_out->to_string());

  GSPAN::gSpan gspan;

  gspan.set_database(database);
  gspan.set_min_support(absolute_min_support);
  gspan.set_graph_output(current_out);
  gspan.run();

  gettimeofday(&stop_time, 0);


  LOG_VALUE(*value_logger, "total_time", std::fixed << utils::get_time_diff(start_time, stop_time));
  LOG_VALUE(*value_logger, "total_frequent_graphs", current_out->get_count());
  int total_frequent_graphs = current_out->get_count();

  INFO(*logger, "clean-up...");

  delete count_out;
  return total_frequent_graphs;
}


int execute_cuda(gspan_cuda::gspan_cuda *gspan, graph_database_t &database, int absolute_min_support)
{
  if(absolute_min_support >= database.size()) {
    cerr << "error: user specified support " << absolute_min_support << " is greater then the database size " << database.size() << endl;
    return 2;
  }

  if(absolute_min_support < 1) {
    cerr << "error: absolute_min_support < 1" << endl;
    return 3;
  }

  graph_counting_output *count_out = new graph_counting_output();
  graph_output *current_out = count_out;

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

  try {
    timeval start_time, stop_time;
    gettimeofday(&start_time, 0);

    //LOG_VAL(*value_logger, absolute_min_support);
    //LOG_VALUE(*value_logger, "output", current_out->to_string());

    //gspan_cuda::gspan_cuda gspan;

    graph_database_cuda h_graph_db = graph_database_cuda::create_from_host_representation(database);
    std::set<int> vertex_label_set;
    std::set<int> edge_label_set;
    gspan_cuda::compact_labels(h_graph_db, vertex_label_set, edge_label_set);
    gspan->set_database(h_graph_db);

    gspan->set_min_support(absolute_min_support);
    gspan->set_graph_output(current_out);
    gspan->set_exec_configurator(&exec_config);
    gspan->run();

    gettimeofday(&stop_time, 0);


    LOG_VALUE(*value_logger, "total_time", std::fixed << utils::get_time_diff(start_time, stop_time));
    LOG_VALUE(*value_logger, "total_frequent_graphs", current_out->get_count());

    gspan->delete_database_from_device();
  } catch(std::exception &e) {
    CRITICAL_ERROR(*logger, "caught exception: " << e.what());
    CRITICAL_ERROR(*logger, "maximal device memory usage (MB): " << memory_checker::get_max_memory_usage_mb());
    throw;
  } catch(...) {
    CRITICAL_ERROR(*logger, "caught unknown exception");
    CRITICAL_ERROR(*logger, "maximal device memory usage (MB): " << memory_checker::get_max_memory_usage_mb());
    throw;
  }
  INFO(*logger, "maximal device memory usage (MB): " << memory_checker::get_max_memory_usage_mb());
  INFO(*logger, "clean-up...");

  int total_frequent_graphs = current_out->get_count();

  delete count_out;
  return total_frequent_graphs;
}



int execute_cuda(graph_database_t &database, int absolute_min_support) {
  gspan_cuda::gspan_cuda gspan;
  INFO(*logger, "execute gspan_cuda");
  return execute_cuda(&gspan, database, absolute_min_support);
}


int execute_cuda_lists(graph_database_t &database, int absolute_min_support) {
  gspan_cuda::gspan_cuda_lists gspan;
  INFO(*logger, "execute gspan_cuda_lists");
  return execute_cuda(&gspan, database, absolute_min_support);
}


int execute_cuda_no_sort(graph_database_t &database, int absolute_min_support) {
  gspan_cuda::gspan_cuda_no_sort gspan;
  INFO(*logger, "execute gspan_cuda_no_sort");
  return execute_cuda(&gspan, database, absolute_min_support);
}


int execute_cuda_no_sort_block(graph_database_t &database, int absolute_min_support) {
  gspan_cuda::gspan_cuda_no_sort_block gspan;
  INFO(*logger, "execute gspan_cuda_no_sort_block");
  return execute_cuda(&gspan, database, absolute_min_support);
}


int execute_cuda_mult_block(graph_database_t &database, int absolute_min_support) {
  gspan_cuda::gspan_cuda_mult_block gspan;
  INFO(*logger, "execute gspan_cuda_mult_block");
  return execute_cuda(&gspan, database, absolute_min_support);
}



int main(int argc, char **argv)
{

  if(argc != 6) {
    std::cerr << "usage: \n" << argv[0] << " <filetype> <filename> <support-int-abs> <config-file> <config-id>" << std::endl;
    return 1;
  }
  string filetype = argv[1];
  string filename = argv[2];
  int absolute_min_support = atoi(argv[3]);


  graph_database_t database;

  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str(string(argv[1]));
  dbio::read_database(file_format, filename, database);

  if(absolute_min_support >= database.size()) {
    cerr << "error: user specified support " << absolute_min_support << " is greater then the database size " << database.size() << endl;
    return 2;
  }

  // 48, 52
  for(int i = 0; i < database.size(); i++) {
    INFO(*logger, "==========================================================");
    INFO(*logger, "window starts at transaction: " << i);
    graph_database_t actual_database = database;
    int db_size = 20;
    if(actual_database.size() - i < db_size) {
      INFO(*logger, "sucessfully finising with the test: sliding window reached end of graph database.");
      break;
    }
    actual_database.erase(actual_database.begin(), actual_database.begin() + i);
    actual_database.erase(actual_database.begin() + db_size, actual_database.end());
    INFO(*logger, "database size: " << actual_database.size() << "; support: " << absolute_min_support);
    int cuda_lists_count = execute_cuda_lists(actual_database, absolute_min_support);
    int cuda_no_sort_count = execute_cuda_no_sort(actual_database, absolute_min_support);
    int cuda_no_sort_block_count = execute_cuda_no_sort_block(actual_database, absolute_min_support);
    int cuda_count = execute_cuda(actual_database, absolute_min_support);
    int cuda_mult_block = execute_cuda_mult_block(actual_database, absolute_min_support);
    int host_count = execute_host(actual_database, absolute_min_support);
    if(cuda_count != host_count ||
       cuda_lists_count != host_count ||
       cuda_no_sort_count != cuda_count ||
       cuda_count != cuda_no_sort_block_count ||
       cuda_mult_block != cuda_count) {
      cout << "cuda_count: " << cuda_count
      << "; host_count: " << host_count
      << "; cuda_lists_count: " << cuda_lists_count
      << "; cuda_no_sort_block_count: " << cuda_no_sort_block_count
      << "; cuda_mult_block: " << cuda_mult_block << endl;
      cout << "for i: " << i << "; meaning transactions from: " << 10 * i << endl;
      throw std::runtime_error("error while testing cuda vs. host implementation");
    } // if
  }



  memory_checker::detect_memory_leaks();


  return 0;
}
