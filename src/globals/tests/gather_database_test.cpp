#include <dfs_code.hpp>
#include <string>
#include <gtest/gtest.h>
#include <stdexcept>

using std::string;
using std::runtime_error;
using types::DFS;
using std::cerr;
using std::endl;

void test_gather_database(const string &ftype, const string &filename)
{
  if(dbio::ftype2str(ftype) != dbio::BIN) {
    cerr << "test possible only on binary database." << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int processor_rank;
  int processor_number;
  MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank); /* find out process rank */
  MPI_Comm_size(MPI_COMM_WORLD, &processor_number); /* find out number of processes */

  types::graph_database_t local_gdb;
  read_database_bin(filename, processor_number, processor_rank);

  types::graph_database_t whole_gdb;
  read_database(dbio::BIN, filename, whole_gdb);

  types::graph_database_t received_whole_gdb;
  gather_database(local_gdb, received_whole_gdb, 0);
  if(processor_rank == 0) {

  }
} // test_gather_database

TEST(mpi, gather_database)
{


} // TEST


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  ::testing::InitGoogleTest(&argc, argv);


  bool r = RUN_ALL_TESTS();
  //perform_mpi_comm_test();

  /* shut down MPI */
  MPI_Finalize();

  return 0;
}

