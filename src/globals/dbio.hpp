#ifndef __DBIO_HPP__
#define __DBIO_HPP__

#include <fstream>
#include <graph_types.hpp>

namespace dbio {
typedef enum { TXT, BIN, FSG, CUDABIN } FILE_TYPE;   //QUEST_BIN, QUEST_ASCII, BMS_ASCII,
FILE_TYPE ftype2str(const std::string &ftype);

// reads an arbitrary support fileformat
void read_database(const FILE_TYPE &ftype, const std::string &filename, types::graph_database_t &gdb);
void write_database(types::graph_database_t &gdb, const FILE_TYPE &ftype, const std::string &filename);

// reading
void read_database_txt(std::istream &is, types::graph_database_t &gdb);
void read_database_txt(const std::string &filename, types::graph_database_t &gdb);
void read_database_fsg(std::istream &is, types::graph_database_t &gdb);
void read_database_fsg(const std::string &filename, types::graph_database_t &gdb);

void read_database_cudabin(std::istream &is, types::graph_database_cuda &gdb);
void read_database_cudabin(const string &filename, types::graph_database_cuda &gdb);

void read_database_bin(std::istream &is, types::graph_database_t &gdb);
void read_database_bin(const std::string &filename, types::graph_database_t &gdb);
void read_database_bin(std::istream &is, types::graph_database_t &gdb, int total_parts, int part);
void read_database_bin(const std::string &filename, types::graph_database_t &gdb, int total_parts, int part);

void write_database_cudabin(std::ostream &os, const types::graph_database_cuda &gdb);
void write_database_cudabin(const string &filename, types::graph_database_cuda &gdb);


// writing
void write_database_bin(std::ostream &os, const types::graph_database_t &gdb);
void write_database_bin(const std::string &filename, types::graph_database_t &gdb);

} // namespace dbio

#endif

