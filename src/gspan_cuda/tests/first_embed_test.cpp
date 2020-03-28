#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <test_support.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>

#include <dbio.hpp>

#include <cuda_gspan_ops.hpp>

using namespace types;
using std::string;

using gspan_cuda::create_first_embeddings;


Logger *logger = Logger::get_logger("CUDA_FE_TEST");

void check_host_db_agains_embedding(const types::graph_database_t &gdb, types::DFS dfs, const types::embedding_list_columns &h_embed);
void check_embedding_against_host_db(const types::graph_database_t &gdb, types::DFS dfs, const types::embedding_list_columns &h_embed);

/*
   static int get_max_vertex_count(const types::graph_database_t &gdb)
   {
   int max = 0;
   for(int i = 0; i < gdb.size(); i++) {
    if(max < gdb[i].vertex_size()) max = gdb[i].vertex_size();
   } // for i
   return max;
   } // get_max_vertex_count
 */
/*
   static bool edge_exist(const types::embedding_list_columns &h_embed, int from, int to)
   {
   for(int i = 0; i < h_embed.columns_lengths[0]; i++) {
   if(h_embed.columns[0][i].vertex_id == from &&
      h_embed.columns[1][i].vertex_id == to)
   {
     return true;
   }

   } // for i

   return false;
   }
 */

static void perform_test(const types::graph_database_t &gdb, types::DFS dfs)
{
  types::graph_database_cuda h_gdb = types::graph_database_cuda::create_from_host_representation(gdb);
  types::graph_database_cuda d_gdb(false);
  h_gdb.copy_to_device(&d_gdb);

  INFO(*logger, "cuda gdb size: " << d_gdb.size());
  INFO(*logger, "host gdb size: " << h_gdb.size());
  TRACE(*logger, "cuda gdb: " << h_gdb.to_string());



  types::embedding_list_columns d_embeddings(false);
  types::embedding_list_columns h_embeddings(true);
  cudapp::cuda_configurator exec_config;
  exec_config.set_parameter("create_first_embeddings.valid_vertex", 100);
  exec_config.set_parameter("create_first_embeddings.store_map", 100);
  exec_config.set_parameter("create_first_embeddings.find_valid_edge", 100);
  exec_config.set_parameter("create_first_embeddings.edge_store", 100);


  create_first_embeddings(dfs, &d_gdb, &d_embeddings, &exec_config);

  h_embeddings.copy_from_device(&d_embeddings);

  TRACE5(*logger, h_embeddings.to_string());

  ASSERT_NO_THROW(check_host_db_agains_embedding(gdb, dfs, h_embeddings));
  ASSERT_NO_THROW(check_embedding_against_host_db(gdb, dfs, h_embeddings));

  d_gdb.delete_from_device();
  h_gdb.delete_from_host();
  d_embeddings.delete_from_device();
  h_embeddings.delete_from_host();
}


TEST(cuda_first_embedding, basic_test)
{
  cudaDeviceReset();

  types::graph_database_t gdb = gspan_cuda::get_test_database();
  types::DFS dfs;
  dfs.from = 0;
  dfs.to = 1;
  dfs.fromlabel = 0;
  dfs.elabel = 0;
  dfs.tolabel = 0;
  perform_test(gdb, dfs);




  types::graph_database_t gdb_labeled = gspan_cuda::get_labeled_test_database();
  dfs.from = 0;
  dfs.to = 1;
  dfs.fromlabel = 0;
  dfs.elabel = 2;
  dfs.tolabel = 0;
  perform_test(gdb_labeled, dfs);
}


TEST(cuda_first_embedding, bugtest)
{
  cudaDeviceReset();

  graph_database_t database;

  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str("-txt");
  dbio::read_database(file_format, "../../testdata/Chemical_340", database);
  database.erase(database.begin() + 4, database.end());

  types::DFS dfs(0, 1, 3, 1, 4);
  perform_test(database, dfs);

}


TEST(cuda_first_embedding, bugtest2)
{
  cudaDeviceReset();

  graph_database_t database;

  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str("-txt");
  dbio::read_database(file_format, "../../testdata/Chemical_340", database);
  database.erase(database.begin() + 4, database.end());

  types::DFS dfs(0, 1, 8, 0, 9);
  perform_test(database, dfs);

}





TEST(cuda_first_embedding, bugtest3)
{
  cudaDeviceReset();

  graph_database_t database;

  dbio::FILE_TYPE file_format;
  file_format = dbio::ftype2str("-txt");
  dbio::read_database(file_format, "../../testdata/Chemical_340", database);
  database.erase(database.begin() + 4, database.end());

  types::DFS dfs(0, 1, 0, 0, 1);
  perform_test(database, dfs);

}




