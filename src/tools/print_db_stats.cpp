#include <iostream>

#include <logger.hpp>
#include <graph_output.hpp>
#include <graph_types.hpp>
#include <utils.hpp>
#include <dbio.hpp>

#include <set>

#include <gspan.hpp>

#include <sys/time.h>

using std::cout;
using std::endl;
using std::cerr;
using std::set;
using std::fixed;
using types::graph_database_t;

Logger *value_logger = Logger::get_logger("VAL");
Logger *logger = Logger::get_logger("MAIN");

int main(int argc, char **argv)
{
  if(argc != 3 ) {
    std::cerr << "usage: \n" << argv[0] << " <filetype> <filename>" << std::endl;
    return 1;
  }


  string filetype = argv[1];
  string filename = argv[2];


  types::graph_database_cuda cuda_gdb(true);


  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str(string(argv[1]));
  if(file_format == dbio::CUDABIN) {
    INFO(*logger, "loading the database in cuda format");
    dbio::read_database_cudabin(filename, cuda_gdb);
    //cuda_gdb.convert_to_host_representation(database);
  } else {
    graph_database_t host_database;
    dbio::read_database(file_format, filename, host_database);
    cuda_gdb = types::graph_database_cuda::create_from_host_representation(host_database);
  } // if-else


  set<int> vertex_labels;
  set<int> edge_labels;
  int vrtx_array_size = cuda_gdb.max_graph_vertex_count * cuda_gdb.db_size;
  int total_vertex_count = 0;
  for(int i = 0; i < vrtx_array_size; i++) {
    vertex_labels.insert(cuda_gdb.vertex_labels[i]);
    //if(cuda_gdb.vertex_labels[i] != -1) total_vertex_count++;
  } // for i

  for(int i = 0;i < cuda_gdb.edges_sizes; i++) {
    edge_labels.insert(cuda_gdb.edges_labels[i]);
  }

  int total_vertex_degree = 0;
  for(int i = 0; i < cuda_gdb.db_size; i++) {
    int start_vid = i * cuda_gdb.max_graph_vertex_count;
    for(int i = start_vid; i < start_vid + cuda_gdb.max_graph_vertex_count; i++) {
      if(cuda_gdb.vertex_is_valid(i)) {
        total_vertex_degree += cuda_gdb.get_vertex_degree(i);
      } // if
    } // for
  } // for i

  cout << "edge count: " << cuda_gdb.edges_sizes/2 << endl;
  cout << "avg edge count: " << fixed << double(cuda_gdb.edges_sizes) / double(cuda_gdb.db_size*2) << endl;
  cout << "db size(graph count): " << cuda_gdb.db_size << endl;
  cout << "average vertex count: " << fixed << double(cuda_gdb.vertex_count)/double(cuda_gdb.db_size) << endl;
  cout << "edges label count: " << edge_labels.size() << endl;
  cout << "vertex label count: " << vertex_labels.size() << endl;
  cout << "avg vertex degree: " << double(total_vertex_degree) / double(cuda_gdb.vertex_count) << endl;
  //cout <<  << endl;
} // main


