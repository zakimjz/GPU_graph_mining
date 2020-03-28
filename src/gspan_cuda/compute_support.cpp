#include <embedding_lists.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_computation_parameters.hpp>
#include <kernel_execution.hpp>
#include <cuda_gspan_ops.hpp>

#include <cuda_tools.hpp>
#include <cuda.h>

#include <cuda_segmented_scan.hpp>

#include <cuda_copy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#define REUSEMEM 1

using namespace types;

namespace gspan_cuda {

static Logger *logger = Logger::get_logger("GET_EXTS");





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// get_support_for_extensions
// this function uses the thrust::sort and thus is veeeeerry slow ...
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



// BinaryPredicate for the head flag segment representation for the dfs blocks of the extension_element_t
// equivalent to thrust::not2(thrust::project2nd<int,int>()));
template <typename HeadFlagType>
struct head_flag_predicate
  : public thrust::binary_function<HeadFlagType,HeadFlagType,bool>
{
  __device__ __host__
  bool operator()(HeadFlagType left, HeadFlagType right) const
  {
    return !right;
  }
};


// dfs code comparator -  equality checking for two dfs codes in the extension_element_t
template <typename extension_element_t>
struct dfs_equals_predicate
  : public thrust::binary_function<extension_element_t,extension_element_t,bool>
{

  __device__ __host__
  dfs_equals_predicate() {
  }

  __device__ __host__
  bool operator()(extension_element_t &left, extension_element_t &right) const
  {
    return (left.from == right.from && left.to == right.to && left.fromlabel == right.fromlabel && left.elabel == right.elabel && left.tolabel == right.tolabel);
  }
};


/* Computes the dfs and the graph boundaries from the extension_element_t array
   The output for both is a stream of 1's and 0's, where 1 marks the head of the block and the other members of the same block are 0
   The dfs boundaries are computed via the dfs_equals_predicate and the graph boundaries are computed by checking the extension_element_t.from_grph of two adjacent elements
   A dfs block may contain several graph blocks inside it
   Example: dfs   boundaries: 100000 100 100
            graph boundaries: 101010 101 110
 */
struct compute_boundaries {

  extension_element_t *d_exts_array;
  int exts_array_length, max_vertex_count;
  int *d_exts_dfs_boundaries, *d_exts_graph_boundaries;


  __device__ __host__
  compute_boundaries(int max_vertex_count, extension_element_t *d_exts_array, int exts_array_length, int *d_exts_graph_boundaries, int *d_exts_dfs_boundaries){
    this->max_vertex_count = max_vertex_count;
    this->d_exts_array = d_exts_array;
    this->exts_array_length = exts_array_length;
    this->d_exts_graph_boundaries = d_exts_graph_boundaries;
    this->d_exts_dfs_boundaries = d_exts_dfs_boundaries;
  }

  __device__ __host__
  void operator()(int idx)  {
    if(idx > 0 ) {
      /* check dfs boundaries */
      dfs_equals_predicate<extension_element_t> pred;
      // check if two dfs codes are equal
      if ( pred( d_exts_array[idx - 1], d_exts_array[idx])  ) {

        d_exts_dfs_boundaries[idx] = 0;

        /* check graph boundaries */
        if(d_exts_array[idx - 1].from_grph / max_vertex_count ==  d_exts_array[idx].from_grph / max_vertex_count )
          d_exts_graph_boundaries[idx] = 0;
        else
          d_exts_graph_boundaries[idx] = 1;

      }else{
        d_exts_dfs_boundaries[idx] = 1;
        d_exts_graph_boundaries[idx]  = 1;
      }    //if pred
    }else  { // if(idx == 0) {
      d_exts_dfs_boundaries[idx] = 1;
      d_exts_graph_boundaries[idx] = 1;
    } // idx == 0
  } // operator
};

// A predicate used by copy_if for checking if the element is 1
struct is_one
{
  __device__ __host__
  bool operator()(const int x)
  {
    return (x == 1);
  }
};




/**
 * Sorts the extension_element_t array and computes the support of the extensions.
 * This functions is veeeeery slow.
 *
 * TODO: fix the execution configuration, it should be given as the argument.
 *
 *
 */
void get_support_for_extensions(int max_vertex_count,
                                extension_element_t *d_exts_array,
                                int exts_array_length,
                                int &dfs_array_length,
                                DFS*&dfs_array,
                                int*&dfs_offsets,
                                int *&support,
                                cudapp::cuda_configurator *exec_conf)
{


  //cudapp::cuda_computation_parameters params(1,exts_array_length);

  TRACE(*logger, "sorting extension elements on device, array length: " << exts_array_length);

  //for testing purpose we expect the extension_element_array in the host
  //extension_element_t* d_exts_array;
  //CUDA_EXEC(cudaMalloc(&d_exts_array, exts_array_length * sizeof(extension_element_t)), *logger);
  //CUDA_EXEC(cudaMemcpy(d_exts_array, h_exts_array, exts_array_length * sizeof(extension_element_t), cudaMemcpyHostToDevice), *logger);

  thrust::sort(thrust::device_pointer_cast(d_exts_array),
               thrust::device_pointer_cast(d_exts_array + exts_array_length),
               extension_element_comparator_kesslr2(max_vertex_count));

  int *d_exts_dfs_boundaries, *d_exts_graph_boundaries;

  CUDAMALLOC(&d_exts_dfs_boundaries, exts_array_length * sizeof(int), *logger);
  CUDA_EXEC(cudaMemset(d_exts_dfs_boundaries, 0, exts_array_length * sizeof(int)), *logger);
  CUDAMALLOC(&d_exts_graph_boundaries, exts_array_length * sizeof(int), *logger);
  CUDA_EXEC(cudaMemset(d_exts_graph_boundaries, 0, exts_array_length * sizeof(int)), *logger);

  // Now compute the dfs and graph boundaries in two arrays (start of the boundaries are marked 1 and the rest are 0 - 10000100100... )
  compute_boundaries bd(max_vertex_count,d_exts_array,exts_array_length,d_exts_graph_boundaries,d_exts_dfs_boundaries);
  cudapp::cuda_computation_parameters bd_params = exec_conf->get_exec_config("compute_support.boundaries", exts_array_length);
  for_each(0, exts_array_length, bd, bd_params);


  // Now get the total number of dfs extension blocks
  int total_dfs_extensions = thrust::reduce(thrust::device_pointer_cast(d_exts_dfs_boundaries),
                                            thrust::device_pointer_cast(d_exts_dfs_boundaries) + exts_array_length,
                                            (int) 0,
                                            thrust::plus<int>());


  // Get the offsets of the boundaries in the sorted extension element array
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + exts_array_length;

  //offsets array has the same length as the total dfs extensions
  int *d_exts_dfs_offsets;
  CUDAMALLOC(&d_exts_dfs_offsets, total_dfs_extensions * sizeof(int), *logger);
  CUDA_EXEC(cudaMemset(d_exts_dfs_offsets, 0, total_dfs_extensions * sizeof(int)), *logger);
  thrust::copy_if(first, // counting iterator starts at  0
                  last,  // exts_array_length
                  thrust::device_pointer_cast(d_exts_dfs_boundaries), // input : streams of 1 and 0 - 100010001010110
                  thrust::device_pointer_cast(d_exts_dfs_offsets),    // output : offsets of 1's in the array
                  is_one());   // operator ( if 1 return true)



  //support array has the same length as the total dfs extensions
  int * d_exts_dfs_support, *d_keys_output;
  CUDAMALLOC(&d_exts_dfs_support, total_dfs_extensions * sizeof(int), *logger);
  CUDA_EXEC(cudaMemset(d_exts_dfs_support, 0, total_dfs_extensions * sizeof(int)), *logger);
  CUDAMALLOC(&d_keys_output, total_dfs_extensions * sizeof(int), *logger);
  CUDA_EXEC(cudaMemset(d_keys_output, 0, total_dfs_extensions * sizeof(int)), *logger);

  //compute the support of the dfs blocks
  thrust::reduce_by_key(thrust::device_pointer_cast(d_exts_dfs_boundaries), //keys input:       1000001000
                        thrust::device_pointer_cast(d_exts_dfs_boundaries) + exts_array_length,
                        thrust::device_pointer_cast(d_exts_graph_boundaries), //values input  : 1010101011
                        thrust::device_pointer_cast(d_keys_output),  //keys output
                        thrust::device_pointer_cast(d_exts_dfs_support),    //values output
                        head_flag_predicate<int>()); //  pred  = not_one


  //allocate the (host) pointers from the caller and copy the values back
  dfs_array_length = total_dfs_extensions;

  // dfs_offsets (host)  <-- d_exts_dfs_offsets (device);
  dfs_offsets = new int[dfs_array_length];
  CUDA_EXEC(cudaMemcpy(dfs_offsets,  d_exts_dfs_offsets, sizeof(int) * dfs_array_length, cudaMemcpyDeviceToHost), *logger);

  // support (host)  <-- d_exts_dfs_support (device);
  support = new int[dfs_array_length];
  CUDA_EXEC(cudaMemcpy(support, d_exts_dfs_support, sizeof(int) * dfs_array_length, cudaMemcpyDeviceToHost), *logger);

  //  h_exts_array2 (host) <-- d_exts_array (device);
  //  dfs_array (host) <-- h_exts_array2 [dfs_offsets] ( host)
  extension_element_t* h_exts_array2 = new extension_element_t[exts_array_length];
  CUDA_EXEC(cudaMemcpy(h_exts_array2, d_exts_array, exts_array_length * sizeof(extension_element_t), cudaMemcpyDeviceToHost), *logger);
  dfs_array = new types::DFS[total_dfs_extensions];
  for(int i = 0; i<total_dfs_extensions; i++) {
    extension_element_t ext = h_exts_array2[dfs_offsets[i]];
    dfs_array[i].from = ext.from;
    dfs_array[i].to = ext.to;
    dfs_array[i].fromlabel = ext.fromlabel;
    dfs_array[i].elabel = ext.elabel;
    dfs_array[i].tolabel = ext.tolabel;
  }

  // Free resources from the device
  //CUDA_EXEC(cudaFree(d_exts_array), *logger);
  delete [] h_exts_array2;
  CUDAFREE(d_exts_dfs_boundaries, *logger);
  CUDAFREE(d_exts_graph_boundaries, *logger);
  CUDAFREE(d_exts_dfs_support, *logger);
  CUDAFREE(d_exts_dfs_offsets, *logger);
  CUDAFREE(d_keys_output, *logger);


} // compute_supports




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract_extensions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct mark_EV_labels {

  extension_element_t *d_exts_array;
  int *d_fwd_flags;
  int *d_bwd_flags;
  DFS *d_forward_dfs_array;
  DFS *d_backward_dfs_array;
  int num_edge_labels; // number of (edge label, vertex label) combinations
  int embedding_cols; // number of vertices in one embedding, i.e., the number of columns in the embedding lists

  __device__ __host__
  mark_EV_labels(extension_element_t *d_exts_array,
                 int *d_fwd_flags,
                 int *d_bwd_flags,
                 DFS *d_forward_dfs_array,
                 DFS *d_backward_dfs_array,
                 int num_edge_labels,
                 int embedding_cols)
  {
    this->d_exts_array = d_exts_array;
    this->d_fwd_flags = d_fwd_flags;
    this->d_bwd_flags = d_bwd_flags;
    this->d_forward_dfs_array = d_forward_dfs_array;
    this->d_backward_dfs_array = d_backward_dfs_array;
    this->num_edge_labels = num_edge_labels;
    this->embedding_cols = embedding_cols;
  }


  __device__ __host__ void process_forward_edges(int idx) {
    int EV_label_index  = d_exts_array[idx].tolabel * num_edge_labels + d_exts_array[idx].elabel;

    d_fwd_flags[EV_label_index] = 1;

    d_forward_dfs_array[EV_label_index].from = d_exts_array[idx].from;
    d_forward_dfs_array[EV_label_index].to = d_exts_array[idx].to;
    d_forward_dfs_array[EV_label_index].fromlabel = d_exts_array[idx].fromlabel;
    d_forward_dfs_array[EV_label_index].elabel = d_exts_array[idx].elabel;
    d_forward_dfs_array[EV_label_index].tolabel = d_exts_array[idx].tolabel;
  }

  // backward edges the backward edges are mapped using 'to' pattern
  // id and edge label. The question is: is this sufficient !?
  __device__ __host__ void process_backward_edges(int idx) {
    //int EV_label_index  = d_exts_array[idx].to * embedding_cols + d_exts_array[idx].elabel;
    int EV_label_index  = d_exts_array[idx].to * num_edge_labels + d_exts_array[idx].elabel;

    d_bwd_flags[EV_label_index] = 1;

    d_backward_dfs_array[EV_label_index].from = d_exts_array[idx].from;
    d_backward_dfs_array[EV_label_index].to = d_exts_array[idx].to;
    d_backward_dfs_array[EV_label_index].fromlabel = d_exts_array[idx].fromlabel;
    d_backward_dfs_array[EV_label_index].elabel = d_exts_array[idx].elabel;
    d_backward_dfs_array[EV_label_index].tolabel = d_exts_array[idx].tolabel;
  }

  __device__ __host__
  void operator() (int idx) {
    if(d_exts_array[idx].is_forward()) process_forward_edges(idx);
    if(d_exts_array[idx].is_backward()) process_backward_edges(idx);
  }

};


struct set_block_offsets {

  extension_element_t *d_exts_array;
  int len;
  int *d_block_offsets;

  __device__ __host__
  set_block_offsets(extension_element_t *d_exts_array, int len, int *d_block_offsets){
    this->d_exts_array = d_exts_array;
    this->len = len;
    this->d_block_offsets = d_block_offsets;
  }

  __device__ __host__
  void operator() (int idx){
    if(idx == 0) {
      d_block_offsets[d_exts_array[idx].from] = 0;
      return;
    } else if(d_exts_array[idx].from != d_exts_array[idx - 1].from) {
      d_block_offsets[d_exts_array[idx].from] = idx;
      return;
    }
  }

};


/**
 * computes the DFS codes of the extensions contained in an array of
 * extension_element_t structures.  does not use thrust::sort
 * (therefore is much faster). This is a first step for support
 * computation. After obtaining extensions (types::DFS structs), a
 * variant of compute_support must be called.
 *
 */
void extract_extensions(int num_edge_labels,
                        int num_vertex_labels,
                        int num_columns,
                        extension_element_t *d_exts_array,
                        int exts_array_length,
                        int *&d_block_offsets,
                        int &d_block_offsets_length,
                        DFS *&d_dfs_array,
                        int &d_dfs_array_length,
                        int *&d_fwd_flags,
                        int *&d_bwd_flags,
                        DFS *&d_fwd_block_dfs_array,
                        DFS *&d_bwd_block_dfs_array,
                        int *&block_offsets_arg,
                        DFS *&dfs_array,
                        int &dfs_array_length,
                        cuda_copy<types::DFS, is_one_array> *cucpy)
{

#ifdef REUSEMEM
  //get the block offsets from the exts array
  if(d_block_offsets_length < num_columns) {
    if(d_block_offsets !=0)
      CUDAFREE(d_block_offsets, *logger);

    CUDAMALLOC(&d_block_offsets, sizeof(int) * num_columns, *logger);
    d_block_offsets_length = num_columns;
  }
#else
    if(d_block_offsets !=0)
      CUDAFREE(d_block_offsets, *logger);
    CUDAMALLOC(&d_block_offsets, sizeof(int) * num_columns, *logger);
    d_block_offsets_length = num_columns;
#endif 

  CUDA_EXEC(cudaMemset(d_block_offsets, 0xff, sizeof(int) * num_columns), *logger ); //fill the offsets array with -1
  set_block_offsets set_offsets(d_exts_array, exts_array_length, d_block_offsets);
  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e(exts_array_length);
  thrust::for_each(b, e, set_offsets);

  TRACE(*logger, "d_block_offsets: " << d_block_offsets);

  delete [] block_offsets_arg;
  block_offsets_arg = new int[num_columns + 1];
  CUDA_EXEC(cudaMemcpy(block_offsets_arg, d_block_offsets, sizeof(int) * num_columns, cudaMemcpyDeviceToHost), *logger);
  block_offsets_arg[num_columns] = exts_array_length;

  int num_blocks = 0;
  for(int i = 0; i < num_columns; i++)
    if(block_offsets_arg[i] != -1) num_blocks++;

  int *block_offsets = new int[num_blocks + 1];
  for(int i = 0, j = 0; i < num_columns; i++) {
    if(block_offsets_arg[i] != -1)
      block_offsets[j++] = block_offsets_arg[i];
  }
  block_offsets[num_blocks] = exts_array_length;

  // Done obtaining block offsets, now process each block separately

  int num_fwd_combinations = num_edge_labels * num_vertex_labels;
  int num_bwd_combinations = num_columns * num_edge_labels;

#ifdef REUSEMEM
  //rellocate bwd arrays if needed
  if (num_bwd_combinations > num_fwd_combinations) {
    CUDAFREE(d_bwd_flags, *logger);
    CUDAMALLOC(&d_bwd_flags, sizeof(int) * num_bwd_combinations, *logger);
    CUDAFREE(d_bwd_block_dfs_array, *logger);
    CUDAMALLOC(&d_bwd_block_dfs_array, sizeof(DFS) * num_bwd_combinations, *logger);
  }
#else
    CUDAFREE(d_bwd_flags, *logger);
    CUDAMALLOC(&d_bwd_flags, sizeof(int) * num_bwd_combinations, *logger);
    CUDAFREE(d_bwd_block_dfs_array, *logger);
    CUDAMALLOC(&d_bwd_block_dfs_array, sizeof(DFS) * num_bwd_combinations, *logger);
#endif

#ifdef REUSEMEM
  // d_dfs_array is just for holding all unique dfs extensions on device
  // DFS elements can't be more than the exts_array_length, so this is enough to hold all the extensions.
  if(d_dfs_array_length < exts_array_length) {
    if(d_dfs_array != 0)
      CUDAFREE(d_dfs_array, *logger);
    CUDAMALLOC(&d_dfs_array, sizeof(DFS) * exts_array_length, *logger);
    d_dfs_array_length = exts_array_length;
  }
#else
    if(d_dfs_array != 0)
      CUDAFREE(d_dfs_array, *logger);
    CUDAMALLOC(&d_dfs_array, sizeof(DFS) * exts_array_length, *logger);
    d_dfs_array_length = exts_array_length;
#endif

  dfs_array_length = 0;  //store the number of dfs extensions, initially 0

  //TRACE(*logger, "d_exts_array: " << print_d_array(d_exts_array, exts_array_length));
  TRACE(*logger, "num_blocks: " << num_blocks
        << "; num_columns: " << num_columns
        << "; exts_array_length: " << exts_array_length
        << "; num_edge_labels: " << num_edge_labels);

  TRACE(*logger, "num_fwd_combinations: " << num_fwd_combinations << "; num_bwd_combinations: " << num_bwd_combinations);


  //process all the blocks except the last one, these all have only fwd extensions
  for(int i = 0; i < num_blocks; i++) {
    TRACE(*logger, "processing block: " << i);
    int start = block_offsets[i];
    int block_len = block_offsets[i + 1] - block_offsets[i];
    if(i == num_blocks - 1) block_len = exts_array_length - block_offsets[i];
    TRACE(*logger, "block start: " << start << "; block length: " << block_len << "; dfs_array_length: " << dfs_array_length);

    CUDA_EXEC(cudaMemset(d_fwd_flags, 0, sizeof(int) * num_fwd_combinations), *logger);
    CUDA_EXEC(cudaMemset(d_bwd_flags, 0, sizeof(int) * num_bwd_combinations), *logger);
    mark_EV_labels mark_ev(d_exts_array, d_fwd_flags, d_bwd_flags, d_fwd_block_dfs_array, d_bwd_block_dfs_array, num_edge_labels, num_columns);
    thrust::counting_iterator<int> b1(start);
    thrust::counting_iterator<int> e1(start + block_len);
    thrust::for_each(b1, e1, mark_ev);

    /*
       int num_exts = thrust::reduce(thrust::device_pointer_cast(d_fwd_flags),
                                  thrust::device_pointer_cast(d_fwd_flags + num_fwd_combinations),
                                  (int) 0,
                                  thrust::plus<int>());

       //copy_if d_block_dfs_array to d_dfs_array in the correct location
       thrust::copy_if(thrust::device_pointer_cast<DFS>(d_fwd_block_dfs_array),
                    thrust::device_pointer_cast<DFS>(d_fwd_block_dfs_array + num_fwd_combinations),
                    thrust::device_pointer_cast<int>(d_fwd_flags),
                    thrust::device_pointer_cast<DFS>(d_dfs_array + dfs_array_length), //total exts from previously processed block
                    is_one());
     */

    int num_exts = cucpy->copy_if(d_fwd_block_dfs_array, num_fwd_combinations, d_dfs_array + dfs_array_length, is_one_array(d_fwd_flags));

    //Now update dfs_array_length
    dfs_array_length += num_exts;
  }

  TRACE(*logger, "dfs_array_length: " << dfs_array_length);


  int num_exts = cucpy->copy_if(d_bwd_block_dfs_array, num_bwd_combinations, d_dfs_array + dfs_array_length, is_one_array(d_bwd_flags));


  /*
     //TODO: copy the backward extensions
     int num_exts = thrust::reduce(thrust::device_pointer_cast(d_bwd_flags),
                                thrust::device_pointer_cast(d_bwd_flags + num_bwd_combinations),
                                (int) 0,
                                thrust::plus<int>());
     //copy_if d_block_dfs_array to d_dfs_array in the correct location
     thrust::copy_if(thrust::device_pointer_cast<DFS>(d_bwd_block_dfs_array),
                  thrust::device_pointer_cast<DFS>(d_bwd_block_dfs_array + num_bwd_combinations),
                  thrust::device_pointer_cast<int>(d_bwd_flags),
                  thrust::device_pointer_cast<DFS>(d_dfs_array + dfs_array_length), //total exts from previously processed block
                  is_one());
   */

  //Now update dfs_array_length
  dfs_array_length += num_exts;


  //Memcpy the device DFS array  to host DFS array
  dfs_array = new types::DFS[dfs_array_length];
  CUDA_EXEC(cudaMemcpy(dfs_array, d_dfs_array, sizeof(types::DFS) * dfs_array_length, cudaMemcpyDeviceToHost), *logger);

  //Free resources
  delete [] block_offsets;
  //CUDAFREE(d_block_offsets, *logger);
  //CUDAFREE(d_dfs_array, *logger);

} //extract extensions




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// compute_support
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct store_db_flag {
  int *d_graph_flags;
  extension_element_t *d_exts;
  types::DFS dfs_elem;
  int max_graph_vertex_count;

  store_db_flag(int *d_graph_flags, extension_element_t *d_exts, types::DFS dfs_elem, int max_graph_vertex_count) {
    this->d_graph_flags = d_graph_flags;
    this->d_exts = d_exts;
    this->dfs_elem = dfs_elem;
    this->max_graph_vertex_count = max_graph_vertex_count;
  }

  __host__ __device__
  void operator()(int idx) {
    if(d_exts[idx].from == dfs_elem.from &&
       d_exts[idx].to == dfs_elem.to &&
       d_exts[idx].fromlabel == dfs_elem.fromlabel &&
       d_exts[idx].elabel == dfs_elem.elabel &&
       d_exts[idx].tolabel == dfs_elem.tolabel) {
      int gid = d_exts[idx].from_grph / max_graph_vertex_count;
      d_graph_flags[gid] = 1;
    }
  } // operator()
};


/**
 * Hypothesis (that is probably valid): the graph ids forms
 * non-decreasing sequence. if they do not then this must be assured
 * when calling the create_first_embeddings. Then (due to the
 * algorithm in get_all_extensions) it will be valid during the
 * computation.
 *
 * How to shrink the array that needs to be reduce to obtain the
 * support:
 *
 * 1) create an array A of length l(A) of integers of the same size as
 *    is the size of one block in the extension_element_t array. (This
 *    array can be kept during the computation on top of the block).
 *
 *    a) write 1's on the boundaries of the embeddings of different
 *       graphs. Otherwise write 0.
 *
 *    b) prefix-scan array A.
 *
 * - for each support computation the array A can be reused. The array
 *   actually remaps graph ids into a linear sequence of ids.
 *
 * 2) For each support computation create an array S (support) of the
 *    size A[l(A) - 1] + 1
 *
 *    Each thread with thread id T then checks its assigned element of
 *    the extension_element_t array agains the dfs_elem and writes one
 *    on position S[A[T]]. S is then reduced.
 *
 * This technique should significantly reduce the size of the array
 * that is finally reduced.
 *
 *
 */
int compute_support(extension_element_t *d_exts, int d_exts_size, types::DFS dfs_elem, int db_size, int max_graph_vertex_count)
{
  int *d_graph_flags = 0;
  CUDAMALLOC(&d_graph_flags, sizeof(int) * db_size, *logger);
  CUDA_EXEC(cudaMemset(d_graph_flags, 0, sizeof(int) * db_size), *logger);

  store_db_flag sdbf(d_graph_flags, d_exts, dfs_elem, max_graph_vertex_count);
  thrust::for_each(thrust::counting_iterator<int>(0),
                   thrust::counting_iterator<int>(d_exts_size),
                   sdbf);
  int supp = thrust::reduce(thrust::device_pointer_cast<int>(d_graph_flags),
                            thrust::device_pointer_cast<int>(d_graph_flags + db_size));

  CUDAFREE(d_graph_flags, *logger);
  return supp;
}







//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// compute_support_remapped_db
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct get_graph_boundaries_in_exts {

  int exts_size, max_vertex_count, *d_graph_boundaries;
  extension_element_t *d_exts;

  __device__ __host__
  get_graph_boundaries_in_exts(int *d_graph_boundaries, extension_element_t *d_exts,int exts_size,int max_vertex_count){
    this->d_graph_boundaries = d_graph_boundaries;
    this->d_exts = d_exts;
    this->exts_size = exts_size;
    this->max_vertex_count = max_vertex_count;
  }

  __device__ __host__
  void operator() (int idx){
    if(idx < (exts_size - 1) ) {
      if(d_exts[idx].from_grph / max_vertex_count != d_exts[idx + 1].from_grph / max_vertex_count)
        d_graph_boundaries[idx] = 1;
      else
        d_graph_boundaries[idx] = 0;
    }else{ //idx ==  exts_size - 1
      d_graph_boundaries[idx] = 1;
    }
  }

};

struct store_mapped_db_flag {
  int *d_graph_flags;
  int *d_gdb_map;
  extension_element_t *d_exts;
  types::DFS dfs_elem;
  int max_graph_vertex_count;

  __device__ __host__
  store_mapped_db_flag(int *d_graph_flags, extension_element_t *d_exts, types::DFS dfs_elem, int max_graph_vertex_count,int *d_gdb_map) {
    this->d_graph_flags = d_graph_flags;
    this->d_exts = d_exts;
    this->dfs_elem = dfs_elem;
    this->max_graph_vertex_count = max_graph_vertex_count;
    this->d_gdb_map = d_gdb_map;
  }

  __device__ __host__
  void operator()(int idx) {
    if(d_exts[idx].from == dfs_elem.from &&
       d_exts[idx].to == dfs_elem.to &&
       d_exts[idx].fromlabel == dfs_elem.fromlabel &&
       d_exts[idx].elabel == dfs_elem.elabel &&
       d_exts[idx].tolabel == dfs_elem.tolabel) {
      int gid = d_gdb_map[idx];
      d_graph_flags[gid] = 1;
      //printf(" i = %d  (%d %d %d %d %d) (%d %d %d %d %d) MATCHED! gid = %d value %d\n", idx, d_exts[idx].from, d_exts[idx].to, d_exts[idx].fromlabel, d_exts[idx].elabel, d_exts[idx].tolabel, dfs_elem.from, dfs_elem.to, dfs_elem.fromlabel, dfs_elem.elabel, dfs_elem.tolabel, gid, d_graph_flags[gid]);
    }
  } // operator()
};


void remap_database_graph_ids(extension_element_t *d_exts,
                              int exts_size,
                              int max_graph_vertex_count,
                              int *&d_graph_boundaries_scan,
                              int &d_graph_boundaries_scan_length,
                              int &mapped_db_size,
                              cuda_segmented_scan *scanner)
{

  if(d_graph_boundaries_scan_length < exts_size) {
    if(d_graph_boundaries_scan != 0) {
      CUDAFREE(d_graph_boundaries_scan, *logger);
    }
    CUDAMALLOC(&d_graph_boundaries_scan, sizeof(int) * (exts_size), *logger);
    d_graph_boundaries_scan_length = exts_size;
  }

  CUDA_EXEC(cudaMemset(d_graph_boundaries_scan, 0, sizeof(int) * exts_size), *logger);

  get_graph_boundaries_in_exts grb(d_graph_boundaries_scan, d_exts, exts_size, max_graph_vertex_count);

  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e = b + exts_size;
  thrust::for_each(b, e, grb);

  int last_entry;
  CUDA_EXEC(cudaMemcpy(&last_entry, d_graph_boundaries_scan + (exts_size - 1), sizeof(int), cudaMemcpyDeviceToHost), *logger);

  TRACE5(*logger, "Before scan:" << print_d_array(d_graph_boundaries_scan, exts_size));

  //Now get the offsets for the remapped graphs in the db, this will be used in the d_graph_flags array
  thrust::exclusive_scan(thrust::device_pointer_cast<int>(d_graph_boundaries_scan),
                         thrust::device_pointer_cast<int>(d_graph_boundaries_scan + exts_size),
                         thrust::device_pointer_cast<int>(d_graph_boundaries_scan));

  //scanner->scan((uint*)d_graph_boundaries_scan, exts_size, EXCLUSIVE);


  TRACE5(*logger, "After scan:" << print_d_array(d_graph_boundaries_scan, exts_size));

  //int mapped_db_size;
  CUDA_EXEC(cudaMemcpy(&mapped_db_size, d_graph_boundaries_scan + exts_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  mapped_db_size += (last_entry == 1) ? 1 : 0;
}

int compute_support_remapped_db(extension_element_t *d_exts,
                                int exts_size,
                                types::DFS dfs_elem,
                                int max_graph_vertex_count,
                                int *d_graph_flags,
                                int *&d_graph_boundaries_scan,
                                int &graph_boundaries_scan_length,
                                bool &compute_boundaries,
                                int &mapped_db_size,
                                cuda_segmented_scan *scanner)
{
  if(compute_boundaries) {
    TRACE(*logger, dfs_elem);
    remap_database_graph_ids(d_exts, exts_size, max_graph_vertex_count, d_graph_boundaries_scan, graph_boundaries_scan_length, mapped_db_size, scanner);
    compute_boundaries = false;
  }

  TRACE4(*logger, "Mapped db size = " << mapped_db_size);

  CUDA_EXEC(cudaMemset(d_graph_flags, 0, sizeof(int) * mapped_db_size), *logger);
  store_mapped_db_flag dbf(d_graph_flags, d_exts, dfs_elem, max_graph_vertex_count, d_graph_boundaries_scan);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1 = b1 + exts_size;
  thrust::for_each(b1, e1, dbf);

  TRACE4(*logger, "Support flags = " << print_d_array(d_graph_flags, mapped_db_size));

  int support = thrust::reduce(thrust::device_pointer_cast<int>(d_graph_flags),
                               thrust::device_pointer_cast<int>(d_graph_flags + mapped_db_size),
                               (int) 0,
                               thrust::plus<int>());

  return support;
}








//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// compute_support_remapped_db_multiple_dfs
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct store_segment_boundaries {
  int *d_flags;
  int segment_size;

  __device__ __host__
  store_segment_boundaries(int *d_flags, int segment_size) {
    this->d_flags = d_flags;
    this->segment_size = segment_size;
  }


  __device__ __host__
  void operator()(int thread_idx) {
    int idx = (thread_idx + 1) * segment_size;
    d_flags[idx] = d_flags[idx] | 0x80000000;
  } // operator()
};

struct store_mapped_db_flag_multiple_dfs {
  int *d_graph_flags;
  extension_element_t *d_exts;
  types::DFS *dfs_elem;
  int dfs_elem_length;
  int max_graph_vertex_count;
  int *d_gdb_map;
  int db_size;

  __device__ __host__
  store_mapped_db_flag_multiple_dfs(int *d_graph_flags,
                                    extension_element_t *d_exts,
                                    types::DFS *dfs_elem,
                                    int dfs_elem_length,
                                    int max_graph_vertex_count,
                                    int *d_gdb_map,
                                    int db_size)
  {
    this->d_graph_flags = d_graph_flags;
    this->d_exts = d_exts;
    this->dfs_elem = dfs_elem;
    this->max_graph_vertex_count = max_graph_vertex_count;
    this->d_gdb_map = d_gdb_map;
    this->dfs_elem_length = dfs_elem_length;
    this->db_size = db_size;
  }

  __device__ __host__
  void store_graph_flag(int thread_idx, int dfs_idx, int *d_flags) {
    if(d_exts[thread_idx].from == dfs_elem[dfs_idx].from &&
       d_exts[thread_idx].to == dfs_elem[dfs_idx].to &&
       d_exts[thread_idx].fromlabel == dfs_elem[dfs_idx].fromlabel &&
       d_exts[thread_idx].elabel == dfs_elem[dfs_idx].elabel &&
       d_exts[thread_idx].tolabel == dfs_elem[dfs_idx].tolabel)
    {
      int gid = d_gdb_map[thread_idx];
      d_flags[gid] = 1;
    }
  }

  __device__ __host__
  void operator()(int thread_idx) {
    for(int dfs_idx = 0; dfs_idx < dfs_elem_length; dfs_idx++) {
      int *d_flags = d_graph_flags + dfs_idx * db_size;
      store_graph_flag(thread_idx, dfs_idx, d_flags);
    } // for dfs_idx
  } // operator()
};


struct get_key_for_compute_support : public thrust::unary_function<int, int>
{
  int db_size;
  get_key_for_compute_support(int db_size) {
    this->db_size = db_size;
  }

  __host__ __device__
  int operator()(int x) const
  {
    return x / db_size;
  }
};



/**
 * This function takes as the argument extension_element_t array and
 * an array of types::DFS (aka extensions) and computes the support of
 * the extensions in the d_exts using reduce_by_key. This may be
 * faster then compute_support_remapped_db in certain cases:
 *
 * 1) a small support and therefore short d_graph_boundaries_scan arrays.
 *
 *   => insufficient amount of parallelity in the computation and huge
 *      amount of calls to thrust::reduce
 *
 *  TODO: We should try to statically allocate d_dfs_elem_index and d_supports
 *
 */
void compute_support_remapped_db_multiple_dfs(extension_element_t *d_exts,
                                              int exts_size,
                                              types::DFS *dfs_elem,
                                              int dfs_elem_length,
                                              int *&supports,
                                              int max_graph_vertex_count,
                                              int *&d_graph_flags,
                                              int &d_graph_flags_length,
                                              int *&d_graph_boundaries_scan,
                                              int &d_graph_boundaries_scan_length,
                                              int &mapped_db_size,
                                              cuda_segmented_reduction *reduction,
                                              cuda_segmented_scan *scanner)
{
  if(!d_graph_boundaries_scan) {
    remap_database_graph_ids(d_exts, exts_size, max_graph_vertex_count, d_graph_boundaries_scan, d_graph_boundaries_scan_length, mapped_db_size, scanner);
  }


  TRACE4(*logger, "Mapped db size: " << mapped_db_size);

#ifdef REUSEMEM
  //reallocate d_graph_flags array if needed
  if(d_graph_flags_length < mapped_db_size * dfs_elem_length) {
    if(d_graph_flags != 0)
      CUDAFREE(d_graph_flags, *logger);
    CUDAMALLOC(&d_graph_flags, sizeof(int) * mapped_db_size * dfs_elem_length, *logger);
    d_graph_flags_length = mapped_db_size * dfs_elem_length;
  }
#else 
    if(d_graph_flags != 0)
      CUDAFREE(d_graph_flags, *logger);
    CUDAMALLOC(&d_graph_flags, sizeof(int) * mapped_db_size * dfs_elem_length, *logger);
    d_graph_flags_length = mapped_db_size * dfs_elem_length;
#endif


  CUDA_EXEC(cudaMemset(d_graph_flags, 0, sizeof(int) * mapped_db_size * dfs_elem_length), *logger);
  TRACE4(*logger, "storing indices, dfs_elem_length: " << dfs_elem_length);

  store_mapped_db_flag_multiple_dfs store_flag(d_graph_flags, d_exts, dfs_elem, dfs_elem_length, max_graph_vertex_count, d_graph_boundaries_scan, mapped_db_size);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1 = b1 + exts_size;
  thrust::for_each(b1, e1, store_flag);
  TRACE4(*logger, "storing indices - finished");

  int *d_dfs_elem_index = 0;
  int *d_supports = 0;
  CUDAMALLOC(&d_dfs_elem_index, sizeof(int) * dfs_elem_length, *logger);
  CUDAMALLOC(&d_supports, sizeof(int) * dfs_elem_length, *logger);

  TRACE4(*logger, "mapped_db_size: " << mapped_db_size << "; dfs_elem_length: " << dfs_elem_length << "; array size: " << (mapped_db_size * dfs_elem_length));
  //INFO(*logger, "d_graph_flags: " << print_d_array(d_graph_flags, mapped_db_size * dfs_elem_length));


  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e(mapped_db_size * dfs_elem_length);
  thrust::reduce_by_key(make_transform_iterator(b, get_key_for_compute_support(mapped_db_size)), // key begin
                        make_transform_iterator(e, get_key_for_compute_support(mapped_db_size)), // key end
                        thrust::device_pointer_cast<int>(d_graph_flags), // 0/1 array
                        thrust::device_pointer_cast<int>(d_dfs_elem_index), // index array
                        thrust::device_pointer_cast<int>(d_supports)); // result array

  TRACE4(*logger, "reduce_by_key finished");

  delete [] supports;
  supports = new int[dfs_elem_length];

  CUDA_EXEC(cudaMemcpy(supports, d_supports, sizeof(int) * dfs_elem_length, cudaMemcpyDeviceToHost), *logger);


  CUDAFREE(d_dfs_elem_index, *logger);
  CUDAFREE(d_supports, *logger);

} // compute_support_remapped_db


/**
 * This function takes as the argument extension_element_t array and
 * an array of types::DFS (aka extensions) and computes the support of
 * the extensions in the d_exts using reduce_by_key. This may be
 * faster then compute_support_remapped_db in certain cases:
 *
 * 1) a small support and therefore short d_graph_boundaries_scan arrays.
 *
 *   => insufficient amount of parallelity in the computation and huge
 *      amount of calls to thrust::reduce
 *
 *  TODO: We should try to statically allocate d_dfs_elem_index and d_supports
 *
 */
/*
void compute_support_remapped_db_multiple_dfs(extension_element_t *d_exts,
                                              int exts_size,
                                              types::DFS *dfs_elem,
                                              int dfs_elem_length,
                                              int *&supports,
                                              int max_graph_vertex_count,
                                              int *&d_graph_flags,
                                              int &d_graph_flags_length,
                                              int *&d_graph_boundaries_scan,
                                              int &d_graph_boundaries_scan_length,
                                              int &mapped_db_size,
                                              cuda_segmented_reduction *reduction,
                                              cuda_segmented_scan *scanner)
{
  if(!d_graph_boundaries_scan) {
    remap_database_graph_ids(d_exts, exts_size, max_graph_vertex_count, d_graph_boundaries_scan, d_graph_boundaries_scan_length, mapped_db_size, scanner);
  }


  TRACE4(*logger, "Mapped db size: " << mapped_db_size);

  //reallocate d_graph_flags array if needed
  if(d_graph_flags_length < mapped_db_size * dfs_elem_length) {
    if(d_graph_flags != 0)
      CUDAFREE(d_graph_flags, *logger);
    CUDAMALLOC(&d_graph_flags, sizeof(int) * mapped_db_size * dfs_elem_length, *logger);
    d_graph_flags_length = mapped_db_size * dfs_elem_length;
  }

  CUDA_EXEC(cudaMemset(d_graph_flags, 0, sizeof(int) * mapped_db_size * dfs_elem_length), *logger);
  TRACE4(*logger, "storing indices, dfs_elem_length: " << dfs_elem_length);

  store_mapped_db_flag_multiple_dfs store_flag(d_graph_flags, d_exts, dfs_elem, dfs_elem_length, max_graph_vertex_count, d_graph_boundaries_scan, mapped_db_size);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1 = b1 + exts_size;
  thrust::for_each(b1, e1, store_flag);
  store_segment_boundaries store_segment(d_graph_flags, mapped_db_size);
  thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(dfs_elem_length), store_segment);
  TRACE4(*logger, "storing indices - finished");

  int *d_dfs_elem_index = 0;
  int *d_supports = 0;
  CUDAMALLOC(&d_dfs_elem_index, sizeof(int) * dfs_elem_length, *logger);
  CUDAMALLOC(&d_supports, sizeof(int) * dfs_elem_length, *logger);

  TRACE4(*logger, "mapped_db_size: " << mapped_db_size << "; dfs_elem_length: " << dfs_elem_length << "; array size: " << (mapped_db_size * dfs_elem_length));
  //INFO(*logger, "d_graph_flags: " << print_d_array(d_graph_flags, mapped_db_size * dfs_elem_length));


  std::vector<uint> segment_count;
  segment_count.push_back(dfs_elem_length);
  std::vector<uint> segment_sizes;
  segment_sizes.push_back(mapped_db_size);
  std::vector<uint> results;
  TRACE2(*logger, print_d_array((uint*)d_graph_flags, mapped_db_size * dfs_elem_length));
  reduction->reduce_inclusive((uint*)d_graph_flags, mapped_db_size * dfs_elem_length, segment_count, segment_sizes, results);


  TRACE4(*logger, "reduce_by_key finished");
  if(results.size() != dfs_elem_length) {
    CRITICAL_ERROR(*logger, "Error, results.size(): " << results.size() << "; expected length(dfs_elem_length): " << dfs_elem_length);
    throw std::runtime_error("Incorrect result size of segmented reduction.");
  }

  delete [] supports;
  supports = new int[dfs_elem_length];
  memcpy(supports, results.data(), sizeof(uint) * dfs_elem_length);



  CUDAFREE(d_dfs_elem_index, *logger);
  CUDAFREE(d_supports, *logger);
} // compute_support_remapped_db
*/


//////////////////////////////////////////////////////////////////////////////////
//
//  Compute Support function for the multiple dfs in the multiple extension blocks
//
///////////////////////////////////////////////////////////////////////////////////

/* d_exts is of size exts_size and is supposed to contain only
 * extensions from one right-most path vertex.
 *
 *
 */
void remap_database_graph_ids_mult(extension_element_t *d_exts,
                                   int exts_size,
                                   int max_graph_vertex_count,
                                   int *d_graph_boundaries_scan,
                                   int &mapped_db_size,
                                   cuda_segmented_scan *scanner)
{

  CUDA_EXEC(cudaMemset(d_graph_boundaries_scan, 0, sizeof(int) * exts_size), *logger);

  get_graph_boundaries_in_exts grb(d_graph_boundaries_scan, d_exts, exts_size, max_graph_vertex_count);

  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e = b + exts_size;
  thrust::for_each(b, e, grb);
  store_segment_boundaries store_segment(d_graph_boundaries_scan, mapped_db_size);

  int last_entry;
  CUDA_EXEC(cudaMemcpy(&last_entry, d_graph_boundaries_scan + (exts_size - 1), sizeof(int), cudaMemcpyDeviceToHost), *logger);

  TRACE5(*logger, "Before scan:" << print_d_array(d_graph_boundaries_scan, exts_size));

  //Now get the offsets for the remapped graphs in the db, this will be used in the d_graph_flags array
  //thrust::exclusive_scan(thrust::device_pointer_cast<int>(d_graph_boundaries_scan),
  //thrust::device_pointer_cast<int>(d_graph_boundaries_scan + exts_size),
  //thrust::device_pointer_cast<int>(d_graph_boundaries_scan));
  scanner->scan((uint*)d_graph_boundaries_scan, exts_size, EXCLUSIVE);

  TRACE5(*logger, "After scan:" << print_d_array(d_graph_boundaries_scan, exts_size));

  //int mapped_db_size;
  CUDA_EXEC(cudaMemcpy(&mapped_db_size, d_graph_boundaries_scan + exts_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  mapped_db_size += (last_entry == 1) ? 1 : 0;
}


struct store_mapped_db_flag_multiple_dfs_blocks {
  int *d_graph_flags;
  extension_element_t *d_exts;
  int exts_size;
  types::DFS *dfs_elem;
  int dfs_elem_length;
  int max_graph_vertex_count;
  int *d_gdb_map;
  int db_size;

  __device__ __host__
  store_mapped_db_flag_multiple_dfs_blocks(int *d_graph_flags,
                                           extension_element_t *d_exts,
                                           int exts_size,
                                           types::DFS *dfs_elem,
                                           int dfs_elem_length,
                                           int max_graph_vertex_count,
                                           int *d_gdb_map,
                                           int db_size)
  {
    this->d_graph_flags = d_graph_flags;
    this->d_exts = d_exts;
    this->exts_size = exts_size;
    this->dfs_elem = dfs_elem;
    this->max_graph_vertex_count = max_graph_vertex_count;
    this->d_gdb_map = d_gdb_map;
    this->dfs_elem_length = dfs_elem_length;
    this->db_size = db_size;
  }

  __device__ __host__
  void store_graph_flag(int thread_idx, int dfs_idx, int *d_flags) {
    int exts_idx = thread_idx % exts_size;
    //printf("Thread id = %d, dfs id = %d, exts_size = %d\n", thread_idx, dfs_idx, exts_size);
    if(d_exts[exts_idx].from == dfs_elem[dfs_idx].from &&
       d_exts[exts_idx].to == dfs_elem[dfs_idx].to &&
       d_exts[exts_idx].fromlabel == dfs_elem[dfs_idx].fromlabel &&
       d_exts[exts_idx].elabel == dfs_elem[dfs_idx].elabel &&
       d_exts[exts_idx].tolabel == dfs_elem[dfs_idx].tolabel)
    {
      int gid = d_gdb_map[exts_idx];
      d_flags[gid] = 1;

      //int k = dfs_idx;
      //printf("Thread id = %d, DFS: (%d %d %d %d %d) , dfs id = %d gid = %d, actual index in flags = %d\n", thread_idx, dfs_elem[k].from, dfs_elem[k].to, dfs_elem[k].fromlabel, dfs_elem[k].elabel, dfs_elem[k].tolabel, k, gid, (dfs_idx * db_size + gid ));
    }
  }

  __device__ __host__
  void operator()(int thread_idx) {
    int dfs_idx = thread_idx / exts_size;
    int *d_flags = d_graph_flags + dfs_idx * db_size;
    store_graph_flag(thread_idx, dfs_idx, d_flags);
  } // operator()
};

/*
   TODO: A new version of the compute support with multiple dfs extensions
       uses multiple blocks from the extension element array
   TODO: We should try to statically allocate d_dfs_elem_index and d_supports
 */


void compute_support_remapped_db_multiple_dfs_blocks(extension_element_t *d_exts,
                                                     int exts_size,
                                                     types::DFS *dfs_elem,
                                                     int dfs_elem_length,
                                                     int *&supports,
                                                     int max_graph_vertex_count,
                                                     int *&d_graph_flags,
                                                     int &d_graph_flags_length,
                                                     int *&d_graph_boundaries_scan,
                                                     int &d_graph_boundaries_scan_length,
                                                     int &mapped_db_size,
                                                     cuda_segmented_reduction *reduction)
{

  TRACE4(*logger, "Mapped db size: " << mapped_db_size);

#ifdef REUSEMEM
  //reallocate d_graph_flags array if needed
  if(d_graph_flags_length < mapped_db_size * dfs_elem_length) {
    if(d_graph_flags != 0)
      CUDAFREE(d_graph_flags, *logger);
    CUDAMALLOC(&d_graph_flags, sizeof(int) * mapped_db_size * dfs_elem_length, *logger);
    d_graph_flags_length = mapped_db_size * dfs_elem_length;
  }
#else
    if(d_graph_flags != 0)
      CUDAFREE(d_graph_flags, *logger);
    CUDAMALLOC(&d_graph_flags, sizeof(int) * mapped_db_size * dfs_elem_length, *logger);
    d_graph_flags_length = mapped_db_size * dfs_elem_length;
#endif

  CUDA_EXEC(cudaMemset(d_graph_flags, 0, sizeof(int) * mapped_db_size * dfs_elem_length), *logger);
  TRACE4(*logger, "storing indices, dfs_elem_length: " << dfs_elem_length);

  store_mapped_db_flag_multiple_dfs_blocks store_flag(d_graph_flags, d_exts, exts_size, dfs_elem, dfs_elem_length, max_graph_vertex_count, d_graph_boundaries_scan, mapped_db_size);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1 = b1 + (exts_size * dfs_elem_length);
  thrust::for_each(b1, e1, store_flag);

  store_segment_boundaries store_segment(d_graph_flags, mapped_db_size);
  thrust::for_each(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(dfs_elem_length), store_segment);

  TRACE4(*logger, "storing indices - finished");
  TRACE4(*logger, "stored indices in d_graph_flags: " << print_d_array(d_graph_flags, mapped_db_size * dfs_elem_length) );

  int *d_dfs_elem_index = 0;
  int *d_supports = 0;
  CUDAMALLOC(&d_dfs_elem_index, sizeof(int) * dfs_elem_length, *logger);
  CUDAMALLOC(&d_supports, sizeof(int) * dfs_elem_length, *logger);

  TRACE4(*logger, "mapped_db_size: " << mapped_db_size << "; dfs_elem_length: " << dfs_elem_length << "; array size: " << (mapped_db_size * dfs_elem_length));
  //INFO(*logger, "d_graph_flags: " << print_d_array(d_graph_flags, mapped_db_size * dfs_elem_length));

  std::vector<uint> segment_count;
  segment_count.push_back(dfs_elem_length);
  std::vector<uint> segment_sizes;
  segment_sizes.push_back(mapped_db_size);
  std::vector<uint> results;
  reduction->reduce_inclusive((uint*)d_graph_flags, mapped_db_size * dfs_elem_length, segment_count, segment_sizes, results);


  TRACE4(*logger, "reduce_by_key finished");

  delete [] supports;
  supports = new int[dfs_elem_length];
  memcpy(supports, results.data(), sizeof(uint) * dfs_elem_length);


  CUDAFREE(d_dfs_elem_index, *logger);
  CUDAFREE(d_supports, *logger);

} // compute_support_remapped_db



} // namespace gspan_cuda

