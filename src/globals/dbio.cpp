#include <dbio.hpp>

#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <numeric>
#include <iostream>

#include <logger.hpp>


using std::string;
using std::stringstream;
using std::runtime_error;
using std::find;
using std::make_pair;

using std::ifstream;
using std::ofstream;
using std::ios;
using std::endl;
using std::flush;
using std::streampos;

using namespace std;

namespace dbio {


FILE_TYPE ftype2str(const std::string &ftype)
{
  if(ftype == "-txt" || ftype == "txt") return TXT;
  if(ftype == "-bin" || ftype == "bin") return BIN;
  if(ftype == "-fsg" || ftype == "fsg") return FSG;
  if(ftype == "-cudabin" || ftype == "cudabin") return CUDABIN;

  throw runtime_error("unknown filetype");
}


void read_database(const FILE_TYPE &ftype, const std::string &filename, types::graph_database_t &gdb)
{
  switch(ftype) {
  case TXT:
    read_database_txt(filename, gdb);
    break;
  case BIN:
    read_database_bin(filename, gdb);
    break;
  case FSG:
    read_database_fsg(filename, gdb);
    break;
  case CUDABIN:
  default:
    throw runtime_error("invalid file type for read_database");
  } // switch
} // read_database

void write_database(types::graph_database_t &gdb, const FILE_TYPE &ftype, const string &filename)
{
  switch(ftype) {
  case BIN:
    write_database_bin(filename, gdb);
    break;
  case CUDABIN:
  case TXT:
  default:
    throw runtime_error("invalid or unsupported file type for write_database");
  } // switch
    //    write_database_txt(filename, gdb) is missing
} // write_database




void read_database_txt(const string &filename, types::graph_database_t &gdb)
{
  ifstream in;
  in.open(filename.c_str(), ios::in);
  read_database_txt(in, gdb);
} // read_database

void read_database_txt(std::istream &is, types::graph_database_t &gdb)
{
  types::Graph g;
  int idx = 0;
  while(true) {
    try {
      g.read(is);
    } catch(std::runtime_error &e) {
      throw;
    }
    if(g.empty()) break;
    gdb.push_back(g);
    idx++;
  } // while
} // read_database



void read_database_fsg(const string &filename, types::graph_database_t &gdb)
{
  ifstream in;
  in.open(filename.c_str(), ios::in);
  read_database_fsg(in, gdb);
} // read_database

void read_database_fsg(std::istream &is, types::graph_database_t &gdb)
{
  types::Graph g;
  int idx = 0;
  while(true) {
    try {
      g.read_fsg(is);
    } catch(std::runtime_error &e) {
      throw;
    }
    if(g.empty()) break;
    gdb.push_back(g);
    idx++;
  } // while
} // read_database






void read_database_bin(std::istream &is, types::graph_database_t &gdb)
{
  is.seekg(0, std::ios_base::end);
  size_t file_size = is.tellg();
  is.seekg(0, std::ios_base::beg);


  std::vector<size_t> index;
  size_t index_size;
  is.read((char*) &index_size, sizeof(size_t));

  char *buf = 0;
  char *buf_pos = 0;
  size_t buf_size = 0;

  buf = (char *) new size_t[index_size];
  buf_size = sizeof(size_t) * index_size;
  buf_pos = buf;

  is.read(buf_pos, buf_size);

  for(size_t i = 0; i < index_size; i++) {
    index.push_back(*((size_t*)buf_pos));
    buf_pos += sizeof(size_t);
  } // for i


  index.push_back(file_size);

  for(size_t i = 0; i < index_size; i++) {
    std::streampos curr_off = is.tellg();
    if(index[i] != curr_off) {
      ERROR(*Logger::get_logger("DBIO"), "while reading graph no " << i << "; expected offset: " << index[i] << "; current offset: " << curr_off);
      throw std::runtime_error("Error: reading a graph from different offset then the graph was stored.");
    } // if

    if(buf_size < index[i + 1] - index[i]) {
      delete [] buf;
      buf = new char[index[i + 1] - index[i]];
      buf_pos = buf;
      buf_size = index[i + 1] - index[i];
    } // if

    size_t stored_graph_size = 0;
    is.read((char*)&stored_graph_size, sizeof(size_t));
    is.read(buf, stored_graph_size * sizeof(char));

    size_t grph_size = types::Graph::get_serialized_size(buf, stored_graph_size);
    if(grph_size != stored_graph_size) {
      ERROR(*Logger::get_logger("DBIO"), "error; size of graph in file: " << stored_graph_size << "; serialized size: " << grph_size);
      abort();
    }
    assert(grph_size == stored_graph_size);
    types::Graph g;
    size_t read = types::Graph::deserialize(g, buf, stored_graph_size);
    gdb.push_back(g);
  } // for i

  size_t is_off = is.tellg();
  size_t is_size = index.back();
  if(is_off != is_size) {
    throw std::runtime_error("read_database_bin did not read the whole file, strange.");
  } // if
  delete [] buf;
} // read_database_bin


void read_database_bin(const std::string &filename, types::graph_database_t &gdb)
{
  std::ifstream is;
  is.open(filename.c_str(), ios::in);
  read_database_bin(is, gdb);
} // read_database_bin




void write_database_bin(std::ostream &os, const types::graph_database_t &gdb)
{
  if(gdb.size() == 0) throw runtime_error("Cannot save empty database.");

  std::vector<size_t> index;
  std::vector<size_t> index_offsets;
  size_t db_start_offset = (gdb.size() + 1) * sizeof(size_t);

  char *buf = 0;
  size_t buf_size = 0;
  char *buf_pos = 0;

  index.push_back(db_start_offset);
  for(int i = 0; i < gdb.size() - 1; i++) {
    size_t s = types::Graph::get_serialized_size(gdb[i]) + sizeof(size_t);
    index.push_back(s);
  } // for i

  std::partial_sum(index.begin(), index.end(), std::back_insert_iterator<std::vector<size_t> >(index_offsets));

  buf = (char *) new size_t[index.size() + 1];
  buf_size = sizeof(size_t) * (index.size() + 1);
  buf_pos = buf;

  *((size_t*)buf_pos) = (size_t) gdb.size();
  buf_pos += sizeof(size_t);
  for(int i = 0; i < index_offsets.size(); i++) {
    *((size_t*)buf_pos) = (size_t) index_offsets[i];
    buf_pos += sizeof(size_t);
  } // for i

  os.write(buf, buf_pos - buf);
  assert((buf_pos - buf) == db_start_offset);
  buf_pos = buf;

  for(int i = 0; i < gdb.size(); i++) {
    size_t s = types::Graph::get_serialized_size(gdb[i]);
    if(buf_size < s) {
      delete [] buf;
      buf_size = s + sizeof(size_t);
      buf = new char[buf_size];
      buf_pos = buf;
    } // if

    *((size_t*)buf) = s;
    size_t ser_s = types::Graph::serialize(gdb[i], buf + sizeof(size_t), s + sizeof(size_t));
    assert(ser_s <= s);
    os.write(buf, s + sizeof(size_t));
  } // for i

  delete [] buf;
} // write_database_bin

void write_database_bin(const std::string &filename, types::graph_database_t &gdb)
{
  std::ofstream os;
  os.open(filename.c_str(), std::ios::trunc | std::ios::out);
  os.seekp(0, std::ios::beg);
  write_database_bin(os, gdb);
} // write_database_bin






void read_database_bin(std::istream &is, types::graph_database_t &gdb, int total_parts, int part)
{
  DEBUG(*Logger::get_logger("DBIO"), "part: " << part);
  DEBUG(*Logger::get_logger("DBIO"), "total_parts: " << total_parts);
  if(part < 0 || total_parts <= part) throw runtime_error("incorrect parameter value: part should be from [0, total_parts-1]");
  int db_size;
  is.read((char*) &db_size, sizeof(int));

  TRACE4(*Logger::get_logger("DBIO"), "db size: " << db_size);

  int start_transaction_no = db_size / total_parts * part;
  int end_transaction_no = db_size / total_parts * (part + 1) - 1;
  if(part == (total_parts - 1)) end_transaction_no = db_size - 1;

  TRACE4(*Logger::get_logger("DBIO"), "start transaction: " << start_transaction_no << "; end transaction no: " << end_transaction_no);

  is.seekg(sizeof(size_t) + start_transaction_no * sizeof(size_t));
  size_t db_beg;
  is.read((char *) &db_beg, sizeof(size_t));
  is.seekg(db_beg);


  char * buffer = 0;
  size_t buffer_size = 0;
  for(int i = start_transaction_no; i <= end_transaction_no; i++) {
    streampos cpos = is.tellg();
    size_t stored_graph_size = 0;
    is.read((char*) &stored_graph_size, sizeof(size_t));
    if(buffer_size < stored_graph_size) {
      delete [] buffer;
      buffer_size = stored_graph_size;
      buffer = new char[buffer_size];
    } // if

    is.read((char*) buffer, stored_graph_size * sizeof(char));
    types::Graph g;
    int deser_size = types::Graph::deserialize(g, buffer, buffer_size);
    gdb.push_back(g);
    if(deser_size != types::Graph::get_serialized_size(g)) {
      stringstream ss;
      ss << "could not read sequence number: " << i;
      throw runtime_error(ss.str());
    } // if
  } // for i

  delete [] buffer;
} // read_database_bin


void read_database_bin(const std::string &filename, types::graph_database_t &gdb, int total_parts, int part)
{
  std::ifstream is;
  is.open(filename.c_str(), ios::in);
  read_database_bin(is, gdb, total_parts, part);
} // read_database_bin













/*******************************************************************************************
* CUDABIN database
*******************************************************************************************/
void write_database_cudabin(const string &filename, types::graph_database_cuda &gdb)
{
  std::ofstream os;
  os.open(filename.c_str(), std::ios::trunc | std::ios::out);
  os.seekp(0, std::ios::beg);
  write_database_cudabin(os, gdb);
}

void write_database_cudabin(std::ostream &os, const types::graph_database_cuda &gdb)
{
  if(gdb.located_on_host == false) {
    throw std::runtime_error("cannot write a database from device directly to device");
  }
  os.write((const char *)&gdb.db_size, sizeof(int));
  os.write((const char *)&gdb.edges_sizes, sizeof(int));
  os.write((const char *)&gdb.max_graph_vertex_count, sizeof(int));
  os.write((const char *)&gdb.vertex_count, sizeof(int));

  os.write((const char *)gdb.edges, sizeof(int) * gdb.edges_sizes);
  os.write((const char *)gdb.edges_labels, sizeof(int) * gdb.edges_sizes);

  os.write((const char *)gdb.vertex_labels, sizeof(int) * gdb.max_graph_vertex_count * gdb.db_size);
  os.write((const char *)gdb.vertex_offsets, sizeof(int) * (gdb.max_graph_vertex_count * gdb.db_size + 1));

}


void read_database_cudabin(const string &filename, types::graph_database_cuda &gdb)
{
  std::ifstream is;
  is.open(filename.c_str(), ios::in);
  read_database_cudabin(is, gdb);
}


void read_database_cudabin(std::istream &is, types::graph_database_cuda &gdb)
{
  if(gdb.located_on_host == false) {
    throw std::runtime_error("cannot read a database from file and write it directly to device");
  }
  gdb.delete_from_host();

  is.read((char *)&gdb.db_size, sizeof(int));
  is.read((char *)&gdb.edges_sizes, sizeof(int));
  is.read((char *)&gdb.max_graph_vertex_count, sizeof(int));
  is.read((char *)&gdb.vertex_count, sizeof(int));

  gdb.edges = new int[gdb.edges_sizes];
  is.read((char *)gdb.edges, sizeof(int) * gdb.edges_sizes);

  gdb.edges_labels = new int[gdb.edges_sizes];
  is.read((char *)gdb.edges_labels, sizeof(int) * gdb.edges_sizes);

  gdb.vertex_labels = new int[gdb.max_graph_vertex_count * gdb.db_size];
  is.read((char *)gdb.vertex_labels, sizeof(int) * gdb.max_graph_vertex_count * gdb.db_size);

  gdb.vertex_offsets = new int[gdb.max_graph_vertex_count * gdb.db_size + 1];
  is.read((char *)gdb.vertex_offsets, sizeof(int) * (gdb.max_graph_vertex_count * gdb.db_size + 1));
}

} // namespace dbio



