#include <iostream>

#include <logger.hpp>
#include <graph_output.hpp>
#include <graph_types.hpp>
#include <utils.hpp>
#include <dbio.hpp>

#include <gspan.hpp>

#include <sys/time.h>

using std::cout;
using std::endl;
using std::cerr;
using types::graph_database_t;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");

int main(int argc, char **argv)
{
  if(argc != 5 && argc != 4) {
    std::cerr << "usage: \n" << argv[0] << " <filetype> <filename> <support-int-abs> [<output-filename>]" << std::endl;
    return 1;
  }


  string filetype = argv[1];
  string filename = argv[2];
  int absolute_min_support = atoi(argv[3]);
  string output_filename;
  if(argc == 5) output_filename = argv[4];

  graph_database_t database;

  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str(string(argv[1]));
  if(file_format == dbio::CUDABIN) {
    INFO(*logger, "loading the database in cuda format");
    types::graph_database_cuda cuda_gdb(true);
    dbio::read_database_cudabin(filename, cuda_gdb);
    cuda_gdb.convert_to_host_representation(database);
  } else {
    dbio::read_database(file_format, filename, database);
  }

  if(absolute_min_support >= database.size()) {
    cerr << "error: user specified support " << absolute_min_support << " is greater then the database size " << database.size() << endl;
    return 2;
  }


  if(absolute_min_support < 1) {
    cerr << "error: absolute_min_support < 1" << endl;
    return 3;
  }

  graph_counting_output *count_out = new graph_counting_output();
  graph_file_output *file_out = 0;

  graph_output *current_out = count_out;

  if(output_filename.size() > 0 && output_filename != "-") {
    file_out = new graph_file_output(output_filename);
    current_out = file_out;
  }



  timeval start_time, stop_time;
  gettimeofday(&start_time, 0);

  LOG_VAL(*value_logger, absolute_min_support);
  LOG_VALUE(*value_logger, "output", current_out->to_string());

  GSPAN::gSpan gspan;

  gspan.set_database(database);
  gspan.set_min_support(absolute_min_support);
  gspan.set_graph_output(current_out);
  gspan.run();

  gettimeofday(&stop_time, 0);


  LOG_VALUE(*value_logger, "total_time", std::fixed << utils::get_time_diff(start_time, stop_time));
  LOG_VALUE(*value_logger, "total_frequent_graphs", current_out->get_count());

  INFO(*logger, "clean-up...");

  delete count_out;
  delete file_out;

  return 0;
}

