#include <string>
#include <iomanip>
#include <iostream>

#include <graph_output.hpp>
#include <graph_types.hpp>
#include <logger.hpp>
#include <utils.hpp>
#include <dbio.hpp>

#include <memory_checker.hpp>
#include <gspan_cuda_lists.hpp>
#include <test_support.hpp>

#include <sys/time.h>


using std::string;
using namespace types;
using namespace dbio;
using std::cerr;

using std::fixed;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");

int main(int argc, char **argv)
{
  if(argc != 4) {
    std::cerr << "usage: \n" << argv[0] << " <filetype> <filename> <support-int-abs> " << std::endl;
    return 1;
  }
  string filetype = argv[1];
  string filename = argv[2];
  int absolute_min_support = atoi(argv[3]);

  LOG_VAL(*value_logger, absolute_min_support);
  LOG_VAL(*value_logger, filename);
  LOG_VAL(*value_logger, filetype);


  graph_database_t database;
  types::graph_database_cuda cuda_gdb(true);

  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str(string(argv[1]));

  if(file_format == CUDABIN) {
    dbio::read_database_cudabin(filename, cuda_gdb);
    cuda_gdb.convert_to_host_representation(database);
  } else {
    dbio::read_database(file_format, filename, database);
  }

  if(absolute_min_support >= database.size()) {
    cerr << "error: user specified support " << absolute_min_support << " is greater then the database size " << database.size() << endl;
    return 2;
  }

  //absolute_min_support = 5;
  //database.erase(database.begin(), database.begin() + 5);
  //database.erase(database.begin() + 10, database.end());
  //graph_database_t database = gspan_cuda::get_labeled_test_database();
  //int absolute_min_support = 3;

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
    gspan_cuda::gspan_cuda_lists gspan;

    if(file_format == CUDABIN) {
      gspan.set_database(cuda_gdb);
    } else {
      gspan.set_database(database, file_format == CUDABIN ? false : true);
    }

    gspan.set_min_support(absolute_min_support);
    gspan.set_graph_output(current_out);
    gspan.set_exec_configurator(&exec_config);


    timeval start_time, stop_time;
    gettimeofday(&start_time, 0);
    LOG_VAL(*value_logger, absolute_min_support);
    gspan.run();
    gettimeofday(&stop_time, 0);


    LOG_VALUE(*value_logger, "total_time", std::fixed << utils::get_time_diff(start_time, stop_time));
    LOG_VALUE(*value_logger, "total_frequent_graphs", current_out->get_count());

    gspan.delete_database_from_device();
  } catch(std::exception &e) {
    CRITICAL_ERROR(*logger, "caught exception: " << e.what());
    CRITICAL_ERROR(*logger, "maximal device memory usage (MB): " << memory_checker::get_max_memory_usage_mb());
  } catch(...) {
    CRITICAL_ERROR(*logger, "caught unknown exception");
    CRITICAL_ERROR(*logger, "maximal device memory usage (MB): " << memory_checker::get_max_memory_usage_mb());
  }
  INFO(*logger, "maximal device memory usage (MB): " << memory_checker::get_max_memory_usage_mb());
  INFO(*logger, "clean-up...");

  delete count_out;


  memory_checker::detect_memory_leaks();


  return 0;
}



