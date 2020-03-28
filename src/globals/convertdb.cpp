#include <string>
#include <iomanip>
#include <iostream>

#include <graph_output.hpp>
#include <graph_types.hpp>
#include <logger.hpp>
#include <utils.hpp>
#include <dbio.hpp>

#include <test_support.hpp>

#include <graph_repair_tools.hpp>

using std::string;
using namespace types;
using namespace dbio;
using std::cerr;

using std::fixed;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");


extern void check_database(const types::graph_database_t &db);

int main(int argc, char **argv)
{
  if(argc != 5) {
    std::cerr << "usage: \n" << argv[0] << " <filetype-in> <filename-in>  <filetype-out> <filename-out>" << std::endl;
    return 1;
  }
  string filetype_in = argv[1];
  string filename_in = argv[2];
  string filetype_out = argv[3];
  string filename_out = argv[4];


  graph_database_t database;
  //database = gspan_cuda::get_labeled_test_database();
  //database = gspan_cuda::get_test_database();

  std::cout << "reading database" << endl;
  dbio::FILE_TYPE file_format_in;
  file_format_in = dbio::ftype2str(filetype_in);
  dbio::read_database(file_format_in, filename_in, database);

  fix_database(database);
  print_database_statistics(database);

  std::cout << "writing database" << endl;
  dbio::FILE_TYPE file_format_out;
  file_format_out = dbio::ftype2str(filetype_out);
  if(file_format_out == CUDABIN) {
    types::graph_database_cuda cuda_gdb = types::graph_database_cuda::create_from_host_representation(database);
    dbio::write_database_cudabin(filename_out, cuda_gdb);
    cuda_gdb.delete_from_host();
  } else {
    dbio::write_database(database,
                         file_format_out,
                         filename_out);
  }

  return 0;
}
