#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <test_support.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>
#include <cuda_datastructures.hpp>
#include <cuda_gspan_ops.hpp>
#include <cuda_tools.hpp>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

using namespace types;
using std::string;

using gspan_cuda::create_first_embeddings;
using gspan_cuda::get_all_extensions;
using gspan_cuda::extension_element_t;
using gspan_cuda::cuda_allocs_for_get_all_extensions;

static Logger *logger = Logger::get_logger("CUDA_FE_TEST");

void store_valid_backward_extension(std::set<int> &host_embedding_extensions,
                                    types::Graph &h_grph,
                                    types::graph_database_cuda &cuda_gdb,
                                    int h_grph_from,
                                    int d_grph_from,
                                    std::set<int> &rmpath_embedding_vids,
                                    std::vector<extension_element_t> &curr_embedding_extensions);

void store_valid_forward_extension(std::set<int> &host_embedding_extensions,
                                   types::Graph &h_grph,
                                   types::graph_database_cuda &cuda_gdb,
                                   int h_grph_from,
                                   int d_grph_from,
                                   std::set<int> &embedding_vids,
                                   std::vector<extension_element_t> &curr_embedding_extensions);

std::vector<extension_element_t>
get_graph_extensions(extension_element_t *exts, int exts_length, int grph_id,
                     types::graph_database_cuda &cuda_gdb, types::embedding_list_columns h_embeddings, int row);




bool check_one_embedding(types::graph_database_t &gdb,
                         types::graph_database_cuda &cuda_gdb,
                         types::RMPath rmpath,
                         types::embedding_list_columns h_embeddings,
                         int row,
                         extension_element_t *extensions,
                         int extensions_length);


void extension_test_1()
{
  types::graph_database_t gdb = gspan_cuda::get_test_database();
  types::graph_database_cuda h_gdb = types::graph_database_cuda::create_from_host_representation(gdb);
  types::graph_database_cuda d_gdb(false);
  h_gdb.copy_to_device(&d_gdb);
  TRACE(*logger, "host gdb: " << h_gdb.to_string());

  DEBUG(*logger, "cuda gdb size: " << d_gdb.size());
  DEBUG(*logger, "host gdb size: " << h_gdb.size());

  types::embedding_list_columns d_embeddings(false);
  types::embedding_list_columns h_embeddings(true);

  types::DFS dfs;
  dfs.from = 0;
  dfs.to = 1;
  dfs.fromlabel = 0;
  dfs.elabel = 0;
  dfs.tolabel = 0;


  DFSCode dfscode = types::DFSCode::read_from_str("(0 1 0 0 0)");
  cudapp::cuda_configurator exec_config;
  exec_config.set_parameter("create_first_embeddings.valid_vertex", 100);
  exec_config.set_parameter("create_first_embeddings.store_map", 100);
  exec_config.set_parameter("create_first_embeddings.find_valid_edge", 100);
  exec_config.set_parameter("create_first_embeddings.edge_store", 100);
  exec_config.set_parameter("compute_extensions.store_validity", 100);
  exec_config.set_parameter("compute_extensions.exec_extension_op", 100);

  create_first_embeddings(dfs, &d_gdb, &d_embeddings, &exec_config);

  h_embeddings.copy_from_device(&d_embeddings);

  types::RMPath cuda_rmpath;
  cuda_rmpath.push_back(0);
  cuda_rmpath.push_back(1);
  types::RMPath host_rmpath;
  host_rmpath.push_back(0);
  extension_element_t *d_extensions = 0;
  extension_element_t *h_extensions = 0;
  int extensions_length = 0;
  cuda_allocs_for_get_all_extensions alloc;
  alloc.init();

  TRACE(*logger, "embeddings: " << h_embeddings.to_string());
  INFO(*logger, "executing get_all_extensions");
  get_all_extensions(&d_gdb, &alloc, &d_embeddings, cuda_rmpath, host_rmpath, dfscode, d_extensions, extensions_length, -1, &exec_config, cuda_rmpath);


  h_extensions = new extension_element_t[extensions_length];
  CUDA_EXEC(cudaMemcpy(h_extensions, d_extensions, sizeof(extension_element_t) * extensions_length, cudaMemcpyDeviceToHost), *logger);
  CUDAFREE(d_extensions, *logger);

  INFO(*logger, "executing test");
  int col_count = h_embeddings.columns_count;
  for(int i = 0; i < h_embeddings.columns_lengths[col_count - 1]; i++) {
    TRACE5(*logger, "checking extensions of embedding #" << i);
    bool ok = check_one_embedding(gdb, h_gdb, cuda_rmpath, h_embeddings, i, h_extensions, extensions_length);
    ASSERT_TRUE(ok);
  } // for i

  alloc.device_free();
  d_gdb.delete_from_device();
  d_embeddings.delete_from_device();
} // extension_test_1()




void extension_test_2()
{
  types::graph_database_t gdb = gspan_cuda::get_test_database();
  types::graph_database_cuda h_gdb = types::graph_database_cuda::create_from_host_representation(gdb);
  types::graph_database_cuda d_gdb(false);
  h_gdb.copy_to_device(&d_gdb);
  TRACE(*logger, "host gdb: " << h_gdb.to_string());

  DEBUG(*logger, "cuda gdb size: " << d_gdb.size());
  DEBUG(*logger, "host gdb size: " << h_gdb.size());

  types::embedding_list_columns d_embeddings(false);
  types::embedding_list_columns h_embeddings(true);
  //int nblocks = 1;
  //cudapp::cuda_computation_parameters fe_exec_config(nblocks, d_gdb.edges_sizes / nblocks);

  types::DFS dfs;
  dfs.from = 0;
  dfs.to = 1;
  dfs.fromlabel = 0;
  dfs.elabel = 0;
  dfs.tolabel = 0;


  DFSCode dfscode = types::DFSCode::read_from_str("(0 1 0 0 0)");
  cudapp::cuda_configurator exec_config;
  exec_config.set_parameter("create_first_embeddings.valid_vertex", 100);
  exec_config.set_parameter("create_first_embeddings.store_map", 100);
  exec_config.set_parameter("create_first_embeddings.find_valid_edge", 100);
  exec_config.set_parameter("create_first_embeddings.edge_store", 100);
  exec_config.set_parameter("compute_extensions.store_validity", 100);
  exec_config.set_parameter("compute_extensions.exec_extension_op", 100);
  create_first_embeddings(dfs, &d_gdb, &d_embeddings, &exec_config);

  h_embeddings.copy_from_device(&d_embeddings);

  types::RMPath cuda_rmpath;
  cuda_rmpath.push_back(0);
  cuda_rmpath.push_back(1);
  types::RMPath host_rmpath;
  host_rmpath.push_back(0);

  extension_element_t *d_extensions = 0;
  extension_element_t *h_extensions = 0;
  int extensions_length = 0;
  cuda_allocs_for_get_all_extensions alloc;
  alloc.init();

  TRACE(*logger, "embeddings: " << h_embeddings.to_string());
  INFO(*logger, "executing get_all_extensions");
  get_all_extensions(&d_gdb, &alloc, &d_embeddings, cuda_rmpath, host_rmpath, dfscode, d_extensions, extensions_length, -1, &exec_config, cuda_rmpath);

  h_extensions = new extension_element_t[extensions_length];
  CUDA_EXEC(cudaMemcpy(h_extensions, d_extensions, sizeof(extension_element_t) * extensions_length, cudaMemcpyDeviceToHost), *logger);
  CUDAFREE(d_extensions, *logger);




  int extension_count = 0;
  for(int i = 0; i < extensions_length; i++) {
    if(h_extensions[i].from == 0 && h_extensions[i].to == 2) {
      extension_count++;
    }
  } // for i


  embedding_element *embed = new embedding_element[extension_count];
  int idx = 0;
  for(int i = 0; i < extensions_length; i++) {
    if(h_extensions[i].from == 0 && h_extensions[i].to == 2) {
      embed[idx].vertex_id = h_extensions[i].to_grph;
      embed[idx].back_link = h_extensions[i].row;
      //cout << __FILE__ << "@" << __LINE__ << ": " << h_extensions[i].to_string() << endl;
      idx++;
    } // if
  } // for i


  DFS d_elem = DFS::parse_from_string("(0 2 0 0 0)");
  h_embeddings.h_extend_by_one_column(d_elem, embed, extension_count);


  delete [] h_extensions;
  types::embedding_list_columns d_extended_embeddings(false);
  h_embeddings.copy_to_device(&d_extended_embeddings);
  cuda_rmpath.clear();
  cuda_rmpath.push_back(0);
  cuda_rmpath.push_back(2);

  TRACE(*logger, "current embeddings: " << h_embeddings.to_string());


  //get_all_extensions(&d_gdb, &d_extended_embeddings, rmpath, d_extensions, extensions_length, -1, &exec_config);
  get_all_extensions(&d_gdb, &alloc, &d_embeddings, cuda_rmpath, host_rmpath, dfscode, d_extensions, extensions_length, -1, &exec_config, cuda_rmpath);
  h_extensions = new extension_element_t[extensions_length];
  CUDA_EXEC(cudaMemcpy(h_extensions, d_extensions, sizeof(extension_element_t) * extensions_length, cudaMemcpyDeviceToHost), *logger);
  CUDAFREE(d_extensions, *logger);

  INFO(*logger, "extensions_length: " << extensions_length);

  INFO(*logger, "executing test, extensions_length: " << extensions_length);
  int col_count = h_embeddings.columns_count;


  for(int i = 0; i < h_embeddings.columns_lengths[col_count - 1]; i++) {
    TRACE5(*logger, "checking extensions of embedding #" << i);
    bool ok = check_one_embedding(gdb, h_gdb, cuda_rmpath, h_embeddings, i, h_extensions, extensions_length);
    if(ok == false) {
      DEBUG(*logger, "failed to check embedding #" << i << "; embedding: " << h_embeddings.embedding_to_string(i));
    }
    ASSERT_TRUE(ok);
  } // for i

  alloc.device_free();
  d_gdb.delete_from_device();
  d_embeddings.delete_from_device();
  d_extended_embeddings.delete_from_device();
  h_embeddings.delete_from_host();
} // extension_test_2




TEST(cuda_get_all_extensions, basic_test)
{
  cudaDeviceReset();
  extension_test_1();
}



TEST(cuda_get_all_extensions, expanding_dfs)
{
  cudaDeviceReset();
  INFO(*logger, "extension_test_2");
  extension_test_2();
}


struct int_compare {
  bool operator()(const int &v1, const int &v2) {
    return rand() % 2;
    //return v1 < v2;
  }
};

TEST(cuda_get_all_extensions, sort_performance_test2)
{
  int *array = new int[20000];
  std::sort(array, array + 20000, int_compare());
}









struct extension_element_comparator_host {

  int max_node;

  extension_element_comparator_host(const extension_element_comparator_host &arg) {
    max_node = arg.max_node;
  }


  extension_element_comparator_host(int max_node) {
    this->max_node = max_node;
  }


  bool operator()(const extension_element_t &elem1, const extension_element_t &elem2) const {
    //bool x = elem1.from != elem2.from || elem1.to != elem2.to;
    //return rand()%2 ^ x;
    //cout << elem1 << " < " << elem2 << endl;
    /* (i,j) != (x,y) */
    if(elem1.from != elem2.from && elem1.to != elem2.to) { //( i != x || j !=y )

      //if elem1 is a forwward edge
      if(elem1.from < elem1.to) {
        //if elem2 is a forward edge
        if(elem2.from < elem2.to) {
          //(j < y || ( j == y && i > x ) )
          if(elem1.to < elem2.to || ( elem1.to == elem2.to && elem1.from > elem2.from ))
            return true;

        }else{      //if elem2 is a backward edge

          if(elem1.from < elem2.to)          // (i < y)
            return true;
        }

      }else{      //if elem1 is a backward edge

        //if elem2 is a backward edge
        if(elem2.from > elem2.to) {
          //(i < x || ( i == x && j < y ) )
          if(elem1.from < elem2.from || ( elem1.from == elem2.from && elem1.to < elem2.to ))
            return true;

        }else{      //if elem2 is a forward edge

          if(elem1.to <= elem2.from)          // (j <= x)
            return true;
        }

      }
    }else{ // ( i == x && j == y)

      if ( elem1.fromlabel < elem2.fromlabel) return true;
      else if ( elem1.fromlabel > elem2.fromlabel) return false;
      else {   //elem1.tolabel == elem2.tolabel

        if ( elem1.elabel < elem2.elabel) return true;
        else if ( elem1.elabel > elem2.elabel) return false;
        else {

          if ( elem1.tolabel < elem2.tolabel) return true;
          else if ( elem1.tolabel > elem2.tolabel) return false;
          else{   // all the labels are the same (same dfs code)

            /* if the elem1 is from g1 and elem2 is from g2
               if g1 < g2 return  true, if g1 > g2 return false */
            //if( (elem1.from_grph / max_node) < (elem2.from_grph / max_node) )
            //  return true;
            //else if ( (elem1.from_grph / max_node) > (elem2.from_grph / max_node) )
            //  return false;
            if( elem1.from_grph < elem2.from_grph )
              return true;
            else if ( elem1.from_grph > elem2.from_grph )
              return false;
            //else if they are from the same graph (and off course has the same dfs code), it does not matter, will return false
          }
        }
      }
    }

    return false;
  } // operator()
};


struct ext_element_eq {
  bool operator()(const extension_element_t &v1, const extension_element_t &v2) const {
    if(v1.from == v2.from &&
       v1.to == v2.to &&
       v1.fromlabel == v2.fromlabel &&
       v1.elabel == v2.elabel &&
       v1.tolabel == v2.tolabel &&
       v1.from_grph == v2.from_grph &&
       v1.to_grph == v2.to_grph &&
       v1.row == v2.row) {
      return true;
    }
    return false;
  }
};

TEST(cuda_get_all_extensions, extension_comparator_test2)
{
  cudaDeviceReset();

  int exts_array_length = 6420;
  extension_element_t *exts = new extension_element_t[exts_array_length];
  //CUDA_EXEC(cudaMemcpy(h_array, d_exts_array, sizeof(extension_element_t) * exts_array_length, cudaMemcpyDeviceToHost), *logger);
  std::ifstream out;
  out.open("extensions.dat", std::ios::in);
  out.read((char *) exts, sizeof(extension_element_t) * exts_array_length);



  INFO(*logger, "host sorting ...");
  std::sort(exts, exts + exts_array_length, gspan_cuda::extension_element_comparator(214));

  extension_element_t *d_exts = 0;
  CUDAMALLOC(&d_exts, sizeof(extension_element_t) * exts_array_length, *logger);
  CUDA_EXEC(cudaMemcpy(d_exts, exts, sizeof(extension_element_t) * exts_array_length, cudaMemcpyHostToDevice), *logger);
  INFO(*logger, "device sorting ...");
  thrust::sort(thrust::device_pointer_cast(d_exts),
               thrust::device_pointer_cast(d_exts + exts_array_length),
               gspan_cuda::extension_element_comparator(214));

  CUDAFREE(d_exts, *logger);
}

TEST(cuda_get_all_extensions, extension_comparator_test)
{
  extension_element_t bckwrd_extensions[] = {
    extension_element_t(5, 0, 0, 0, 0, 10, 10, 0),
    extension_element_t(5, 0, 0, 1, 0, 10, 10, 0),
    extension_element_t(5, 1, 0, 0, 0, 10, 10, 0),
    extension_element_t(5, 1, 0, 1, 0, 10, 10, 0),
  };

  EXPECT_PRED2(gspan_cuda::extension_element_comparator(50), bckwrd_extensions[0], bckwrd_extensions[1]);
  EXPECT_PRED2(gspan_cuda::extension_element_comparator(50), bckwrd_extensions[1], bckwrd_extensions[2]);
  EXPECT_PRED2(gspan_cuda::extension_element_comparator(50), bckwrd_extensions[2], bckwrd_extensions[3]);


  extension_element_t frwrd_extensions[] = {
    extension_element_t(5, 6, 0, 1, 0, 10, 10, 0),
    extension_element_t(5, 6, 0, 1, 1, 10, 10, 0),
    extension_element_t(5, 6, 0, 2, 0, 10, 10, 0),
    extension_element_t(4, 6, 0, 1, 0, 10, 10, 0),
    extension_element_t(4, 6, 0, 1, 1, 10, 10, 0),
    extension_element_t(3, 6, 0, 1, 0, 10, 10, 0),
    extension_element_t(2, 6, 0, 0, 0, 10, 10, 0),
    extension_element_t(1, 6, 0, 1, 0, 10, 10, 0),
    extension_element_t(0, 6, 0, 0, 0, 10, 10, 0),
    extension_element_t(0, 6, 0, 1, 0, 10, 10, 0),
    extension_element_t(0, 6, 0, 1, 1, 10, 10, 0)
  };

  for(int i = 0; i < 10; i++) {
    EXPECT_PRED2(gspan_cuda::extension_element_comparator(50), frwrd_extensions[i], frwrd_extensions[i + 1]);
  }

  std::vector<extension_element_t> ext_vec;
  for(int i = 0; i < 4; i++) ext_vec.push_back(bckwrd_extensions[i]);
  for(int i = 0; i < 11; i++) ext_vec.push_back(frwrd_extensions[i]);

  std::vector<extension_element_t> shuffled_vec = ext_vec;
  std::random_shuffle(shuffled_vec.begin(), shuffled_vec.end());


  std::sort(shuffled_vec.begin(), shuffled_vec.end(), gspan_cuda::extension_element_comparator_neel(50));
  bool same = equal(ext_vec.begin(), ext_vec.end(),
                    shuffled_vec.begin(), ext_element_eq());

  for(int i = 0; i < shuffled_vec.size(); i++) {
    cout << shuffled_vec[i] << endl;
  }

  EXPECT_TRUE(same);
}

/*
   TEST(cuda_get_all_extensions, sort_test)
   {
   cudaDeviceReset();
   int size = 50000;
   extension_element_t *exts = new extension_element_t[size];
   INFO(*logger, "preparing array for sorting");
   for(int i = 0; i < size; i++) {
    exts[i].from = rand() % 7;
    exts[i].to = rand() % 7;
    if(exts[i].from == exts[i].to) exts[i].from = exts[i].from + 1;
    exts[i].fromlabel = rand() % 3;
    exts[i].elabel = rand() % 3;
    exts[i].tolabel = rand() % 3;
    exts[i].from_grph = rand() % 1000;
    exts[i].to_grph = rand() % 1000;
   } // for i

   INFO(*logger, "host sorting ...");
   std::sort(exts, exts + size, gspan_cuda::extension_element_comparator(50));

   //for(int i = 0; i < size; i++) {
   //std::cout << exts[i].to_string() << std::endl;
   //}

   //return;

   INFO(*logger, "copying to device");
   extension_element_t *d_exts = 0;
   CUDAMALLOC(&d_exts, sizeof(extension_element_t) * size, *logger);
   CUDA_EXEC(cudaMemcpy(d_exts, exts, sizeof(extension_element_t) * size, cudaMemcpyHostToDevice), *logger);


   INFO(*logger, "device sorting ...");
   thrust::sort(thrust::device_pointer_cast(d_exts),
               thrust::device_pointer_cast(d_exts + size),
               gspan_cuda::extension_element_comparator(50));

   } // TEST
 */

