#include <vector>
#include <algorithm>
#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <test_support.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>
#include <thrust/device_ptr.h>
#include <cuda_gspan_ops.hpp>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
using namespace types;
using std::string;

//using gspan_cuda::create_first_embeddings;
using gspan_cuda::get_support_for_extensions;
using gspan_cuda::extension_element_t;
using gspan_cuda::embedding_extension_t;
using gspan_cuda::intersection_fwd_fwd;
using gspan_cuda::exp_intersection_bwd_bwd;

Logger *logger = Logger::get_logger("CUDA_FE_TEST");

template<typename T>
static void print(const T *a, const int len){
  for(int i = 0; i<len; i++)
    cout << a[i] << " ";
  cout << endl;
}

static int get_max_vertex_count(const types::graph_database_t &gdb)
{
  int max = 0;
  for(int i = 0; i < gdb.size(); i++) {
    if(max < gdb[i].vertex_size()) max = gdb[i].vertex_size();
  } // for i
  return max;
} // get_max_vertex_count


/*
   static bool check_equals(int *a, int *b, int len){
   for(int i = 0; i <len; i++)
    if(a[i] != b[i]) return false;

   return true;
   }*/
/*
   static bool check_equals(std::vector<extension_element_t> a,std::vector<extension_element_t> b){


   if (a.size() != b.size() ) return false;

   for(int i = 0; i <a.size(); i++) {
    if(a[i].from != b[i].from || a[i].to != b[i].to || a[i].fromlabel != b[i].fromlabel || a[i].elabel != b[i].elabel || a[i].tolabel != b[i].tolabel || a[i].from_grph != b[i].from_grph || a[i].to_grph != b[i].to_grph )
      return false;
   }

   return true;
   }
 */

static bool check_equals2(embedding_element *a, embedding_element *b, int len_1, int len_2){

  std::cout << len_1 << " " << len_2 << "\n";
  if(len_1 != len_2) return false;
  for(int i = 0; i<len_1; i++) {
    if(a[i].vertex_id != b[i].vertex_id || a[i].back_link != b[i].back_link) {
      std::cout << " Did not match " << a[i].to_string() << " " << b[i].to_string() << "\n";
      return false;
    }
  }

  return true;
}

static bool check_equals_int(int *a, int *b, int len_1, int len_2){

  std::cout << len_1 << " " << len_2 << "\n";
  if (len_1 != len_2) return false;
  for(int i = 0; i<len_1; i++)
    if(a[i] != b[i]) {
      std::cout << " did not match " << a[i] << " " << b[i] << std::endl;
      return false;
    }
  return true;
}

struct dfs_code_comparator {

  __device__ __host__
  dfs_code_comparator() {
  }

  __device__ __host__
  bool operator()(const DFS &elem1, const DFS &elem2) const {

    /* (i,j) != (x,y) */
    if(elem1.from != elem2.from || elem1.to != elem2.to) { //( i != x || j !=y )

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
          //else if they are from the same graph (and off course has the same dfs code), it does not matter, will return false

        }
      }

    }

    return false;

  } // operator()
};



static void perform_test5(){

  embedding_element *d_cl1, *d_cl2;

  int len_1 = 7, len_2 = 6;
  embedding_element h_cl1[] = { embedding_element(10,0),
                                embedding_element(11,0),
                                embedding_element(14,1),
                                embedding_element(17,1),
                                embedding_element(19,1),
                                embedding_element(22,3),
                                embedding_element(23,3)};
  embedding_element h_cl2[] = { embedding_element(14,1),
                                embedding_element(18,1),
                                embedding_element(22,3),
                                embedding_element(23,3),
                                embedding_element(25,3),
                                embedding_element(28,4)};

  int h_result[] = { 2, 5, 6 };
  int h_res_len  = 3;

  cudaMalloc(&d_cl1, len_1 * sizeof(types::embedding_element) );
  cudaMemcpy(d_cl1, h_cl1, len_1 * sizeof(types::embedding_element), cudaMemcpyHostToDevice);
  cudaMalloc(&d_cl2, len_2 * sizeof(types::embedding_element) );
  cudaMemcpy(d_cl2, h_cl2, len_2 * sizeof(types::embedding_element), cudaMemcpyHostToDevice);

  embedding_extension_t d_eo_1(false), d_eo_2(false),d_eo_result(false);
  int *d_result, len_result;
  d_eo_1.embedding_column = d_cl1; d_eo_1.col_length = len_1;
  d_eo_2.embedding_column = d_cl2; d_eo_2.col_length = len_2;

  //intersection_bwd_bwd(d_eo_1, d_eo_2, d_result, len_result);
  exp_intersection_bwd_bwd(d_eo_1, d_eo_2, d_eo_result);


  d_result = d_eo_result.filtered_emb_offsets;
  len_result = d_eo_result.filtered_emb_offsets_length;
  int *h_out = new int[len_result];
  //cudaMemcpy(h_out, d_result , len_result*sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_out, d_result, d_eo_result.filtered_emb_offsets_length * sizeof(int), cudaMemcpyDeviceToHost);

  ASSERT_TRUE(check_equals_int(h_out, h_result, len_result, h_res_len) );
  //free resources
  //free(h_cl1); free(h_cl2);
  delete [] h_out;

  cudaFree(d_cl1); cudaFree(d_cl2);
  cudaFree(d_result);

}

static void perform_test4(){


  embedding_element *d_cl1;
  std::cout << "Self FWD FWD intersection test\n";

  int len_1 = 7;
  embedding_element h_cl1[] = { embedding_element(10,0),
                                embedding_element(11,0),
                                embedding_element(12,1),
                                embedding_element(13,1),
                                embedding_element(14,1),
                                embedding_element(17,3),
                                embedding_element(19,3)};

  embedding_element h_res[] = { embedding_element(11,0),
                                embedding_element(10,1),
                                embedding_element(13,2),
                                embedding_element(14,2),
                                embedding_element(12,3),
                                embedding_element(14,3),
                                embedding_element(12,4),
                                embedding_element(13,4),
                                embedding_element(19,5),
                                embedding_element(17,6)};
  int h_res_len = 10;

  cudaMalloc(&d_cl1, len_1 * sizeof(types::embedding_element) );
  cudaMemcpy(d_cl1, h_cl1, len_1 * sizeof(types::embedding_element), cudaMemcpyHostToDevice);
  //cudaMalloc(&d_cl2, len_2 * sizeof(types::embedding_element) );
  //cudaMemcpy(d_cl2, h_cl2, len_2 * sizeof(types::embedding_element), cudaMemcpyHostToDevice);

  embedding_extension_t d_eo_1(false), d_eo_result(false);
  d_eo_1.embedding_column = d_cl1; d_eo_1.col_length = len_1;
  //d_eo_2.embedding_column = d_cl2; d_eo_2.col_length = len_2;

  //embedding_extension_t *d_eo_1, *d_eo_2, *d_eo_result;
  //cudaMemcpy(d_eo_1, &h_eo_1, sizeof(embedding_extension_t), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_eo_2, &h_eo_2, sizeof(embedding_extension_t), cudaMemcpyHostToDevice);

  cudapp::cuda_configurator exec_config;
  intersection_fwd_fwd(d_eo_1, d_eo_1, d_eo_result, &exec_config);

  embedding_element *h_out = new embedding_element[d_eo_result.col_length];

  cudaMemcpy(h_out, d_eo_result.embedding_column, d_eo_result.col_length * sizeof(embedding_element), cudaMemcpyDeviceToHost);

  std::cout << "support = " << compute_support_for_fwd_ext(d_eo_result,5) << std::endl;
  int * graph_id_list = new int[d_eo_result.support];
  cudaMemcpy(graph_id_list, d_eo_result.graph_id_list, d_eo_result.support * sizeof(int), cudaMemcpyDeviceToHost);
  print(graph_id_list,d_eo_result.support);

  // std::cout << "intersection size = " << get_graph_id_list_intersection_size(d_eo_result,d_eo_result) << std::endl;

  ASSERT_TRUE(check_equals2(h_out, h_res, d_eo_result.col_length, h_res_len) );

  //free resources
  //free(h_cl1); free(h_cl2);
  cudaFree(d_cl1);
  free(h_out);
  free(graph_id_list);
  //cudaFree(d_eo_1); cudaFree(d_eo_2);

}

static void perform_test3(){


  embedding_element *d_cl1, *d_cl2;

  int len_1 = 7, len_2 = 6;
  embedding_element h_cl1[] = { embedding_element(10,0),
                                embedding_element(11,0),
                                embedding_element(12,1),
                                embedding_element(13,1),
                                embedding_element(14,1),
                                embedding_element(17,3),
                                embedding_element(19,3)};
  embedding_element h_cl2[] = { embedding_element(20,1),
                                embedding_element(21,1),
                                embedding_element(22,2),
                                embedding_element(23,2),
                                embedding_element(25,3),
                                embedding_element(28,4)};

  embedding_element h_res[] = { embedding_element(20,2),
                                embedding_element(21,2),
                                embedding_element(20,3),
                                embedding_element(21,3),
                                embedding_element(20,4),
                                embedding_element(21,4),
                                embedding_element(25,5),
                                embedding_element(25,6)};
  int h_res_len = 8;

  cudaMalloc(&d_cl1, len_1 * sizeof(types::embedding_element) );
  cudaMemcpy(d_cl1, h_cl1, len_1 * sizeof(types::embedding_element), cudaMemcpyHostToDevice);
  cudaMalloc(&d_cl2, len_2 * sizeof(types::embedding_element) );
  cudaMemcpy(d_cl2, h_cl2, len_2 * sizeof(types::embedding_element), cudaMemcpyHostToDevice);

  embedding_extension_t d_eo_1(false), d_eo_2(false), d_eo_result(false);
  d_eo_1.embedding_column = d_cl1; d_eo_1.col_length = len_1;
  d_eo_2.embedding_column = d_cl2; d_eo_2.col_length = len_2;

  //embedding_extension_t *d_eo_1, *d_eo_2, *d_eo_result;
  //cudaMemcpy(d_eo_1, &h_eo_1, sizeof(embedding_extension_t), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_eo_2, &h_eo_2, sizeof(embedding_extension_t), cudaMemcpyHostToDevice);

  cudapp::cuda_configurator exec_config;
  intersection_fwd_fwd(d_eo_1, d_eo_2, d_eo_result, &exec_config);

  embedding_element *h_out = new embedding_element[d_eo_result.col_length];

  cudaMemcpy(h_out, d_eo_result.embedding_column, d_eo_result.col_length * sizeof(embedding_element), cudaMemcpyDeviceToHost);

  std::cout << "support = " << compute_support_for_fwd_ext(d_eo_result,5) << std::endl;

  ASSERT_TRUE(check_equals2(h_out, h_res, d_eo_result.col_length, h_res_len) );

  //free resources
  //free(h_cl1); free(h_cl2);
  cudaFree(d_cl1); cudaFree(d_cl2);
  free(h_out);
  //cudaFree(d_eo_1); cudaFree(d_eo_2);

}


static void perform_test2(const types::graph_database_t &gdb)
{
  int max_vertex_count = get_max_vertex_count(gdb);
  types::graph_database_cuda h_gdb = types::graph_database_cuda::create_from_host_representation(gdb);
  types::graph_database_cuda d_gdb(false);
  h_gdb.copy_to_device(&d_gdb);

  INFO(*logger, "cuda gdb size: " << d_gdb.size());
  INFO(*logger, "host gdb size: " << h_gdb.size());

  INFO(*logger, "*** Get Extension Support Test ***");

  cudapp::cuda_computation_parameters fe_exec_config = cudapp::cuda_configurator::get_computation_parameters(d_gdb.edges_sizes, 512);

  int ext_size = 10;

  /* from, to, fromlabel, elabel, tolabel, from_grph, to_grph */
  gspan_cuda::extension_element_t t[] = {
    extension_element_t( 0, 1, 0, 1, 0, 7, 9, 0),
    extension_element_t( 3, 2, 0, 1, 0,21,19, 0),
    extension_element_t( 2, 1, 0, 1, 0,17,15, 0),
    extension_element_t( 0, 2, 0, 1, 0, 7, 4, 0),
    extension_element_t( 2, 3, 0, 1, 0,22,17, 0),
    extension_element_t( 0, 1, 0, 1, 0,26,27, 0),
    extension_element_t( 2, 1, 0, 1, 0,28,29, 0),
    extension_element_t( 3, 2, 0, 1, 0,27,31, 0),
    extension_element_t( 2, 3, 0, 1, 0, 0, 2, 0),
    extension_element_t( 0, 2, 0, 1, 0,32,35, 0)
  };


  types::DFS *dfs_array;


  int *dfs_offsets, dfs_array_len, *support;
  gspan_cuda::extension_element_t* d_exts_array;
  cudaMalloc(&d_exts_array, ext_size * sizeof(gspan_cuda::extension_element_t));
  cudaMemcpy(d_exts_array, t, ext_size * sizeof(gspan_cuda::extension_element_t), cudaMemcpyHostToDevice);
  cudapp::cuda_configurator exec_config;
  exec_config.set_parameter("compute_support.boundaries", 100);

  get_support_for_extensions(13, d_exts_array, ext_size, dfs_array_len, dfs_array, dfs_offsets, support, &exec_config);

  cout << "size of dfs array" << dfs_array_len << endl;
  print(dfs_offsets,dfs_array_len); print(support,dfs_array_len);
  print(dfs_array,dfs_array_len);

  INFO(*logger, "*** Checking the unique DFS codes order ***");
  dfs_code_comparator comp;
  for(int i = 0; i<dfs_array_len - 1; i++)
    ASSERT_TRUE(comp(dfs_array[i], dfs_array[i + 1]) );

}


static void perform_test1(const types::graph_database_t &gdb)
{
  int max_vertex_count = get_max_vertex_count(gdb);
  types::graph_database_cuda h_gdb = types::graph_database_cuda::create_from_host_representation(gdb);
  types::graph_database_cuda d_gdb(false);
  h_gdb.copy_to_device(&d_gdb);

  INFO(*logger,  "*** Sorting of Extension Elements ***" );

  int size = 10;

  /* from, to, fromlabel, elabel, tolabel, from_grph, to_grph */
  gspan_cuda::extension_element_t t[] = {
    extension_element_t( 0, 1, 0, 1, 0, 7, 9, 0),
    extension_element_t( 3, 2, 0, 1, 0,21,19, 0),
    extension_element_t( 2, 1, 0, 1, 0,17,15, 0),
    extension_element_t( 0, 2, 0, 1, 0, 7, 4, 0),
    extension_element_t( 2, 3, 0, 1, 0,22,17, 0),
    extension_element_t( 0, 1, 0, 1, 0,26,27, 0),
    extension_element_t( 2, 1, 0, 1, 0,28,29, 0),
    extension_element_t( 3, 2, 0, 1, 0,27,31, 0),
    extension_element_t( 2, 3, 0, 1, 0, 0, 2, 0),
    extension_element_t( 0, 2, 0, 1, 0,32,35, 0)
  };


  std::vector<gspan_cuda::extension_element_t> v;
  for(int i = 0; i<size; i++)
    v.push_back(t[i]);

  gspan_cuda::extension_element_comparator comp(13);
  sort(v.begin(), v.end(), comp);


  std::vector<gspan_cuda::extension_element_t>::iterator it;

  for(it = v.begin(); it != v.end(); ++it) {
    //INFO(*logger,  (*it).to_string() );
    cout << (*it).to_string();
  }
  cout << endl;

  int exts_array_length = size;
  gspan_cuda::extension_element_t* d_exts_array;
  cudaMalloc(&d_exts_array, exts_array_length * sizeof(gspan_cuda::extension_element_t));
  cudaMemcpy(d_exts_array, t, exts_array_length * sizeof(gspan_cuda::extension_element_t), cudaMemcpyHostToDevice);

  thrust::sort(thrust::device_pointer_cast(d_exts_array), thrust::device_pointer_cast(d_exts_array) + exts_array_length, comp);

  //thrust::host_vector<extension_element_t> h_exts_vec(exts_array_length);
  //thrust::copy(thrust::device_pointer_cast(d_exts_array), thrust::device_pointer_cast(d_exts_array) + exts_array_length, h_exts_vec);

  //INFO(*logger, "*** checking if the sort results from the std::sort and the thrust::sort are equal");
  //ASSERT_TRUE(check_equals(v,h_exts_vec));

  cudaFree(d_exts_array);

}

TEST(cuda_create_extensions, basic_test)
{
  cudaDeviceReset();

  types::graph_database_t gdb = gspan_cuda::get_test_database();

  perform_test1(gdb);
  perform_test2(gdb);

}

TEST(cuda_create_extensions, test_fwd_fwd_intersection)
{

  perform_test3();
  perform_test4();

}

TEST(cuda_create_extensions, test_bwd_bwd_intersection)
{

  perform_test5();

}


