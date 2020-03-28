#include <embedding_lists.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_computation_parameters.hpp>
#include <kernel_execution.hpp>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>
#include <cuda_tools.hpp>
#include <cuda_gspan_ops.hpp>
#include <thrust/transform_reduce.h>

#include <cuda.h>

using namespace types;

namespace gspan_cuda {

static Logger *logger = Logger::get_logger("EXTEND_EXTS");


//computes the backlink_offset base array for both input embedding column
struct compute_backlink_offset_base {

  embedding_element *embedding_col;
  backlink_offset_t *backlink_offsets;
  int *offset_boundaries;
  int len;

  __device__ __host__
  compute_backlink_offset_base(embedding_element *embedding_col, backlink_offset_t *backlink_offsets, int *offset_boundaries, int len){
    this->embedding_col  = embedding_col;
    this->backlink_offsets = backlink_offsets;
    this->offset_boundaries = offset_boundaries;
    this->len = len;
  }

  __device__ __host__
  void operator() (int idx){

    backlink_offsets[idx].offset = idx;
    backlink_offsets[idx].back_link = embedding_col[idx].back_link;

    if(idx > 0) {

      if(embedding_col[idx - 1].back_link != embedding_col[idx].back_link)
        offset_boundaries[idx] = 1;
      else offset_boundaries[idx] = 0;

    }else{ //idx == 0
      offset_boundaries[idx] =  1;
    }

  }

};


//computes the lengths of each backlink blocks in the input embedding column
struct compute_lens {

  backlink_offset_t *backlink_offsets;
  int len, max_offset;

  __device__ __host__
  compute_lens(backlink_offset_t *backlink_offsets, int len, int max_offset){
    this->backlink_offsets = backlink_offsets;
    this->len = len;
    this->max_offset = max_offset;
  }

  __device__ __host__
  void operator() (int idx){
    if(idx < (len - 1) )
      backlink_offsets[idx].len = backlink_offsets[idx + 1].offset - backlink_offsets[idx].offset;
    else //if(idx == (len - 1) )
      backlink_offsets[idx].len = max_offset - backlink_offsets[idx].offset;
  }


};


//compute the size of the intersections for blocks from two columns. b1 x b2
struct compute_intersection_sizes {
  backlink_offset_t *col1, *col2;

  __device__ __host__
  compute_intersection_sizes(backlink_offset_t *col1, backlink_offset_t *col2 ){
    this->col1 = col1;
    this->col2 = col2;
  }

  __device__ __host__
  int operator() (int idx){
    return col1[idx].len * col2[idx].len;
  }

};


//get the intersection column with the appropriate vertex ids from the col 2 and the backlinks pointing to the rows in col 1
struct compute_extension_column {
  backlink_offset_t *bo_col_1, *bo_col_2;
  embedding_element *emb_col_1, *emb_col_2, *emb_col_result;
  int *valid_elements;
  int max_len;

  __device__ __host__
  compute_extension_column(embedding_element* emb_col_1, embedding_element* emb_col_2, backlink_offset_t *bo_col_1, backlink_offset_t *bo_col_2, int max_len, embedding_element *emb_col_result, int *valid_elements){
    this->bo_col_1 = bo_col_1;
    this->bo_col_2 = bo_col_2;
    this->emb_col_1 = emb_col_1;
    this->emb_col_2 = emb_col_2;
    this->max_len   = max_len;
    this->emb_col_result = emb_col_result;
    this->valid_elements = valid_elements;
  }

  __device__ __host__
  void operator() (int idx){


    for(int i = bo_col_1[idx].offset,k = 0; i < (bo_col_1[idx].offset + bo_col_1[idx].len); i++)
      for(int j = bo_col_2[idx].offset; j < (bo_col_2[idx].offset + bo_col_2[idx].len); j++) {

        if(emb_col_1[i].vertex_id != emb_col_2[j].vertex_id) {
          emb_col_result[idx * max_len + k].vertex_id = emb_col_2[j].vertex_id;
          //emb_col_result[idx*max_len + k].back_link = emb_col_1[i].vertex_id;
          emb_col_result[idx * max_len + k].back_link = i;
          valid_elements[idx * max_len + k] = 1;
          k++;
        }

      }
  }


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





void get_backlink_offsets_for_embeddings(embedding_extension_t &d_col){

  embedding_element *d_embedding_col = d_col.embedding_column;
  int len  = d_col.col_length;

  TRACE4(*Logger::get_logger("FWD_FWD"), "In get backlink offsets");

  //allocate the backlink_offset and the block_boundaries arrays
  backlink_offset_t *d_backlink_offsets_base;
  CUDAMALLOC(&d_backlink_offsets_base, sizeof(backlink_offset_t) * len, *logger);

  int *d_offset_boundaries;
  CUDAMALLOC(&d_offset_boundaries, sizeof(int) * len, *logger);

  // prepare the backlink_offset arrays and the backlink block boundaries for the embedding column
  //cudapp::cuda_computation_parameters params1 = exec_conf->get_exec_config("compute_support.boundaries",len_1);
  //cudapp::cuda_computation_parameters params1(1,len);
  compute_backlink_offset_base bo_col(d_embedding_col, d_backlink_offsets_base, d_offset_boundaries, len);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1(len);
  thrust::for_each(b1, e1, bo_col);

  // Now get the total number of backlink blocks in embedding column
  d_col.num_backlinks = thrust::reduce(thrust::device_pointer_cast(d_offset_boundaries),
                                       thrust::device_pointer_cast(d_offset_boundaries) + len,
                                       (int) 0,
                                       thrust::plus<int>());

  //offsets array has the same length as the total number of backlink blocks
  //backlink_offset_t *d_backlink_offsets_col_1;
  CUDAMALLOC(&(d_col.backlink_offsets), d_col.num_backlinks * sizeof(backlink_offset_t), *logger);
  thrust::copy_if(thrust::device_pointer_cast(d_backlink_offsets_base), // starts at  0
                  thrust::device_pointer_cast(d_backlink_offsets_base + len),  // d_col.num_backlinks
                  thrust::device_pointer_cast(d_offset_boundaries), // input : streams of 1 and 0 - 100010001010110
                  thrust::device_pointer_cast(d_col.backlink_offsets),    // output : backlink offsets array
                  is_one());   // operator ( if 1 return true)

  //populate lengths of blocks in the offset array
  //cudapp::cuda_computation_parameters params2 = exec_conf->get_exec_config("compute_support.boundaries",len_2);
  //cudapp::cuda_computation_parameters params3(1, d_col.num_backlinks);
  compute_lens comp_lens(d_col.backlink_offsets, d_col.num_backlinks, len);
  thrust::counting_iterator<int> b2(0);
  thrust::counting_iterator<int> e2(d_col.num_backlinks);
  thrust::for_each(b2, e2, comp_lens);


  //print backlink_offsets array
  TRACE4(*Logger::get_logger("FWD_FWD"), " backlink offsets array:" << print_d_array2(d_col.backlink_offsets,d_col.num_backlinks));

  CUDAFREE(d_backlink_offsets_base,*logger);
  CUDAFREE(d_offset_boundaries,*logger);
}



// Get the intersection between two embedding columns in fwd extensions
// The output is stored in the result embedding column

void intersection_fwd_fwd(embedding_extension_t &d_embedding_col_1,
                          embedding_extension_t &d_embedding_col_2,
                          embedding_extension_t &d_result,
                          cudapp::cuda_configurator *exec_conf)
{


  TRACE4(*Logger::get_logger("FWD_FWD"), "In forward forward intersection: the input columns:");

  TRACE4(*Logger::get_logger("FWD_FWD"), "d_embedding_col_1: " << print_d_array2(d_embedding_col_1.embedding_column, d_embedding_col_1.col_length));
  TRACE4(*Logger::get_logger("FWD_FWD"), "d_embedding_col_2: " << print_d_array2(d_embedding_col_2.embedding_column, d_embedding_col_2.col_length));

  //if(!d_embedding_col_1.backlink_offsets && !d_embedding_col_2.backlink_offsets) get_backlink_offsets_for_embeddings(d_embedding_col_1,d_embedding_col_2);
  if(!d_embedding_col_1.backlink_offsets ) get_backlink_offsets_for_embeddings(d_embedding_col_1);
  if(!d_embedding_col_2.backlink_offsets ) get_backlink_offsets_for_embeddings(d_embedding_col_2);

  //backlink_offset_t *d_backlink_offsets_col_1 = d_embedding_col_1.backlink_offsets;
  //backlink_offset_t *d_backlink_offsets_col_2 = d_embedding_col_2.backlink_offsets;

  TRACE4(*Logger::get_logger("FWD_FWD"), "Before intersection");
  backlink_offset_t *d_backlink_offsets_result_1 = 0;
  int min_size = (d_embedding_col_1.num_backlinks < d_embedding_col_2.num_backlinks) ? d_embedding_col_1.num_backlinks : d_embedding_col_2.num_backlinks;
  CUDAMALLOC(&d_backlink_offsets_result_1,  min_size * sizeof(backlink_offset_t), *logger);
  TRACE4(*Logger::get_logger("FWD_FWD"), "calling set_intersection 1");
  thrust::device_ptr<backlink_offset_t>  r1 = thrust::set_intersection(thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_1.backlink_offsets),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_1.backlink_offsets + d_embedding_col_1.num_backlinks),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_2.backlink_offsets),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_2.backlink_offsets + d_embedding_col_2.num_backlinks),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_backlink_offsets_result_1) );

  TRACE4(*Logger::get_logger("FWD_FWD"), "size: " << (r1 - thrust::device_pointer_cast<backlink_offset_t>(d_backlink_offsets_result_1) ));

  backlink_offset_t *d_backlink_offsets_result_2 = 0;
  CUDAMALLOC(&d_backlink_offsets_result_2,  min_size * sizeof(backlink_offset_t), *logger);
  TRACE4(*Logger::get_logger("FWD_FWD"), "calling set_intersection 2");
  thrust::device_ptr<backlink_offset_t>  r2 = thrust::set_intersection(thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_2.backlink_offsets),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_2.backlink_offsets + d_embedding_col_2.num_backlinks),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_1.backlink_offsets),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_embedding_col_1.backlink_offsets + d_embedding_col_1.num_backlinks),
                                                                       thrust::device_pointer_cast<backlink_offset_t>(d_backlink_offsets_result_2) );

  TRACE4(*Logger::get_logger("FWD_FWD"), "size: " << (r2 - thrust::device_pointer_cast<backlink_offset_t>(d_backlink_offsets_result_2)));

  TRACE4(*Logger::get_logger("FWD_FWD"), " After Intersection");
  TRACE4(*Logger::get_logger("FWD_FWD"), "d_backlink_offsets_result_1: " << print_d_array2(d_backlink_offsets_result_1, min_size));
  TRACE4(*Logger::get_logger("FWD_FWD"), "d_backlink_offsets_result_2: " << print_d_array2(d_backlink_offsets_result_2, min_size));

  //If the sizes don't match, then there is a critical problem
  int intersection_len = r2 - thrust::device_pointer_cast<backlink_offset_t>(d_backlink_offsets_result_2);

  TRACE4(*Logger::get_logger("FWD_FWD"), " After Intersection (chopped)");
  TRACE4(*Logger::get_logger("FWD_FWD"), "d_backlink_offsets_result_1: " << print_d_array2(d_backlink_offsets_result_1, intersection_len));
  TRACE4(*Logger::get_logger("FWD_FWD"), "d_backlink_offsets_result_2: " << print_d_array2(d_backlink_offsets_result_2, intersection_len));

  //CUDAMALLOC(&d_intersection_sizes, sizeof(int)*intersection_len, *logger);
  thrust::counting_iterator<int> b(0);
  thrust::counting_iterator<int> e(intersection_len);
  compute_intersection_sizes cis(d_backlink_offsets_result_1, d_backlink_offsets_result_2);
  int max_intersection_size = thrust::transform_reduce(b, e, cis, 0, thrust::maximum<int>());

  TRACE4(*Logger::get_logger("FWD_FWD"), " max intersection size:" << max_intersection_size);

  embedding_element *d_intersection_column = 0;
  CUDAMALLOC(&d_intersection_column, sizeof(embedding_element) * intersection_len * max_intersection_size, *logger);

  int *d_valid_elements = 0;
  CUDAMALLOC(&d_valid_elements, sizeof(int) * intersection_len * max_intersection_size, *logger);
  CUDA_EXEC(cudaMemset(d_valid_elements, 0, sizeof(int) * intersection_len * max_intersection_size), *logger);

  //cudapp::cuda_computation_parameters params5(1, intersection_len );
  cudapp::cuda_computation_parameters params5 = exec_conf->get_exec_config("fwd_fwd.compute_extension", intersection_len);
  compute_extension_column cec(d_embedding_col_1.embedding_column,
                               d_embedding_col_2.embedding_column,
                               d_backlink_offsets_result_1,
                               d_backlink_offsets_result_2,
                               max_intersection_size,
                               d_intersection_column,
                               d_valid_elements);
  for_each(0, intersection_len, cec, params5);


  TRACE4(*Logger::get_logger("FWD_FWD"), " new column:");
#ifdef LOG_TRACE
  int *h_valid_elements = new int[intersection_len * max_intersection_size];
  CUDA_EXEC(cudaMemcpy(h_valid_elements, d_valid_elements, intersection_len * max_intersection_size * sizeof(int), cudaMemcpyDeviceToHost ), *logger);
  std::stringstream ss;
  for(int i = 0; i<( intersection_len * max_intersection_size ); i++) {
    ss << h_valid_elements[i] << " ";
  }
  TRACE4(*Logger::get_logger("FWD_FWD"), " valid elements: " << ss.str());
  delete [] h_valid_elements;
#endif


  //get the number of elements in the result column
  int num_result_elements = thrust::reduce(thrust::device_pointer_cast(d_valid_elements),
                                           thrust::device_pointer_cast(d_valid_elements) + intersection_len * max_intersection_size,
                                           (int) 0,
                                           thrust::plus<int>());

  embedding_element *d_embedding_col_result;
  //The final copy if to the result column
  CUDAMALLOC(&d_embedding_col_result, sizeof(embedding_element) * num_result_elements, *logger);
  thrust::copy_if(thrust::device_pointer_cast(d_intersection_column), // starts at  0
                  thrust::device_pointer_cast(d_intersection_column + intersection_len * max_intersection_size),
                  thrust::device_pointer_cast(d_valid_elements), // input : streams of 1 and 0 - 100010001010110
                  thrust::device_pointer_cast(d_embedding_col_result),    // output : offsets of 1's in the array
                  is_one());   // operator ( if 1 return true)


  //check final output

  TRACE4(*Logger::get_logger("FWD_FWD"), "final output: ");

  d_result.col_length = num_result_elements;
  d_result.embedding_column = d_embedding_col_result;
  d_result.ext_type = FRWD;
  //CUDAFREE(d_backlink_offsets_col_1,*logger);
  //CUDAFREE(d_backlink_offsets_col_2,*logger);
  CUDAFREE(d_backlink_offsets_result_1,*logger);
  CUDAFREE(d_backlink_offsets_result_2,*logger);
  CUDAFREE(d_intersection_column,*logger);
  CUDAFREE(d_valid_elements,*logger);

}


// Get the intersection between two embedding columns ( forward and backward )
// The output is stored in the result embedding column

void intersection_fwd_bwd(embedding_extension_t d_embedding_col_1,
                          embedding_extension_t d_embedding_col_2,
                          embedding_extension_t &d_result)
{

  std::cout << "In forward backward intersection: the input columns:\n";
  //int len_1 = d_embedding_col_1.col_length;
  //int len_2 = d_embedding_col_2.col_length;


  throw std::runtime_error("intersection_fwd_bwd not implemeneted");

}

struct populate_pseudo_fwd_exts {

  embedding_element *emb;
  int *offsets;
  __device__ __host__
  populate_pseudo_fwd_exts(embedding_element *emb, int *offsets){
    this->emb = emb;
    this->offsets = offsets;
  }

  __device__ __host__
  void operator() (int idx){
    emb[idx].back_link = offsets[idx];
    emb[idx].vertex_id = -1;
  }

};

// Get the intersection between two embedding columns ( backward and forward )
// The output is stored in the result embedding column
//
// @note This function should output also the filtered last column of d_embeddings
// @param d_embedding_col_1 first embedding (supposed to be the backward embedding)
// @param d_embedding_col_2 second embedding (supposed to be the forward embedding)
// @param filtered_last_col_emb is the column that will replace the last column in the d_embeddings
// @param d_result is the resulting column, i.e., the extension
// HERE IS THE ERROR ...
//
// TODO: In the bwd_fwd intersection the fwd_fwd intersection is performed between the filtered backlink offsets of the bwd exts and the fwd exts
// It should be modified to do just an additional filtering on the fwd exts
void intersection_bwd_fwd(const types::embedding_list_columns &d_embeddings,
                          const types::graph_database_cuda &cuda_gdb,
                          embedding_extension_t &d_embedding_col_1,
                          embedding_extension_t &d_embedding_col_2,
                          embedding_extension_t &filtered_last_col_emb,
                          embedding_extension_t &d_result,
                          cudapp::cuda_configurator *exec_conf)
{
  Logger *logger = Logger::get_logger("BWD_FWD");
  if(filtered_last_col_emb.embedding_column == 0) {
    // fill offsets
    // compute d_result
    //TRACE(*logger, "In backword forward intersection: the input columns:");
    TRACE(*logger, "d_embedding_col_1: " << d_embedding_col_1.to_string());
    TRACE(*logger, "d_embedding_col_1(content): " << print_d_array(d_embedding_col_1.filtered_emb_offsets, d_embedding_col_1.filtered_emb_offsets_length));
    TRACE(*logger, "d_embedding_col_2: " << d_embedding_col_2.to_string());
    TRACE(*logger, "d_embedding_col_2(content): " << print_d_array(d_embedding_col_2.embedding_column, d_embedding_col_2.col_length));
    types::embedding_element *last_col;
    int last_col_length = 0;
    CUDA_EXEC(cudaMemcpy(&last_col, d_embeddings.columns + d_embeddings.columns_count - 1, sizeof(types::embedding_element*), cudaMemcpyDeviceToHost), *logger);
    CUDA_EXEC(cudaMemcpy(&last_col_length, d_embeddings.columns_lengths + d_embeddings.columns_count - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
    TRACE(*logger, "last_col: " << last_col);
    TRACE(*logger, "column lengths: " << print_d_array(d_embeddings.columns_lengths, d_embeddings.columns_count));

    //embedding_extension_t filtered_last_col_emb(false, FRWD); // aka new embedding collumn, aka first column for the fwd-fwd intersection
    filtered_last_col_emb = embedding_extension_t(false, FRWD);
    TRACE(*logger, "calling filter_backward_embeddings");

    filter_backward_embeddings(last_col,
                               last_col_length,
                               filtered_last_col_emb.embedding_column,
                               filtered_last_col_emb.col_length,
                               d_embedding_col_1.filtered_emb_offsets,
                               d_embedding_col_1.filtered_emb_offsets_length,
                               exec_conf);

    TRACE(*logger, "filter_backward_embeddings input(content): " << print_d_array(last_col, last_col_length));
    filtered_last_col_emb.dfs_elem = d_embedding_col_1.dfs_elem;
    TRACE(*logger, "filtered_last_col_emb.embedding_column: " << print_d_array(filtered_last_col_emb.embedding_column, filtered_last_col_emb.col_length));
    TRACE(*logger, "produced forward embedding: " << filtered_last_col_emb.to_string());
  }

  TRACE(*logger, "creating the pseudo FWD extension for the FWD-FWD operation");
  embedding_extension_t d_embedding_col_pseudo_fwd_ext(false);  // this column contains only the filtered offsets as the backlink and an invalid vertex id (-1) for each entry
  CUDAMALLOC(&(d_embedding_col_pseudo_fwd_ext.embedding_column), sizeof(embedding_element) * d_embedding_col_1.filtered_emb_offsets_length, *logger);
  d_embedding_col_pseudo_fwd_ext.col_length = d_embedding_col_1.filtered_emb_offsets_length;
  populate_pseudo_fwd_exts pfe(d_embedding_col_pseudo_fwd_ext.embedding_column, d_embedding_col_1.filtered_emb_offsets);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1(d_embedding_col_1.filtered_emb_offsets_length);
  thrust::for_each(b1,e1,pfe);

  TRACE(*logger, "executing FWD-FWD operation");
  intersection_fwd_fwd(d_embedding_col_pseudo_fwd_ext, d_embedding_col_2, d_result, exec_conf);

  d_embedding_col_pseudo_fwd_ext.device_free();

  TRACE(*logger, "computing support");

/*
   TRACE(*logger, "executing FWD-FWD operation");
   intersection_fwd_fwd(filtered_last_col_emb, d_embedding_col_2, d_result, exec_conf);
   TRACE(*logger, "computing support");

   // TODO: remove!?
   compute_support_for_fwd_ext(filtered_last_col_emb, cuda_gdb.max_graph_vertex_count);
   TRACE(*logger, "produced forward embedding(filtered_last_col_emb): " << filtered_last_col_emb.to_string());
 */

  compute_support_for_fwd_ext(d_result, cuda_gdb.max_graph_vertex_count);
  TRACE(*logger, "produced forward embedding(d_result): " << d_result.to_string());
  d_result.dfs_elem = d_embedding_col_2.dfs_elem;
}

/*
   struct embedding_element_iterator {

   embedding_element *p;
   int i;
   public:
   __device__ __host__
   embedding_element_iterator(embedding_element *p,int i) {
    this->p = p; this->i = i;
   }
   //__device__ __host__
   //embedding_element *begin() {return p;}
   //__device__ __host__
   //embedding_element *end() {return p+len;}

   __device__ __host__
   embedding_element_iterator& operator++() {
   ++p; ++i; return *this;
   }
   __device__ __host__
   embedding_element_iterator operator++(int) {
    embedding_element_iterator tmp(*this); operator++(); return tmp;
   }

   __device__ __host__
   int operator*() {
    return i;
    // return *p;
   }


   };



   struct embedding_element_lt_op
   //  : public thrust::binary_function<int,int,bool>
   {

   embedding_element *e1,*e2;

   __device__ __host__
   embedding_element_lt_op(embedding_element *e1, embedding_element *e2){
    this->e1 = e1;
    this->e2 = e2;
   }

   __device__ __host__
   bool operator() (const int i, const int j) const {
    int a = (e1[i].back_link < e2[j].back_link || (e1[i].back_link == e2[j].back_link && e1[i].vertex_id < e2[j].vertex_id) );
    printf(" i = %d (%d %d), j = %d (%d %d) result = %d e1 =%p e2 =%p\n", i,e1[i].back_link,e1[i].vertex_id,j,e2[j].back_link,e2[j].vertex_id,a,e1,e2);

    if(e1[i].back_link < e2[j].back_link) return true;
    if(e1[i].back_link == e2[j].back_link && e1[i].vertex_id < e2[j].vertex_id ) return true;
    return false;

    //return (i < j);
    //return (e1[i].back_link < e2[j].back_link || (e1[i].back_link == e2[j].back_link && e1[i].vertex_id < e2[j].vertex_id) );
   }
   };

   // Get the intersection between two embedding columns ( backward and backward )
   // The output is stored in the result embedding column

   void intersection_bwd_bwd(embedding_extension_t d_embedding_col_1,
                          embedding_extension_t d_embedding_col_2,
                          int *&d_filtered_offsets,
                          int &len_result)
   {

   int len_1 = d_embedding_col_1.col_length;
   int len_2 = d_embedding_col_2.col_length;

   std::cout << "In backward backward intersection: the input columns:\n";
   print_d_array2(d_embedding_col_1.embedding_column, len_1); print_d_array2(d_embedding_col_2.embedding_column, len_2);
   std::cout << len_1 << " " << len_2 << std::endl;

   int *d_result;
   int max_len = (len_1 > len_2) ? len_1 : len_2;
   CUDAMALLOC(&d_result,max_len * sizeof(int), *logger);
   CUDA_EXEC(cudaMemset(d_result,0,max_len * sizeof(int) ), *logger);

   printf("e1 = %p, e2 = %p", d_embedding_col_1.embedding_column, d_embedding_col_2.embedding_column);
   embedding_element_lt_op op1(d_embedding_col_1.embedding_column, d_embedding_col_2.embedding_column);


   thrust::counting_iterator<int> b1(0);
   thrust::counting_iterator<int> e1(len_1);
   thrust::counting_iterator<int> b2(0);
   thrust::counting_iterator<int> e2(len_2);

   thrust::device_ptr<int> r2 = thrust::set_intersection(b1,e1,b2,e2,
                                                        thrust::device_pointer_cast<int>(d_result),
                                                        op1);

   //embedding_element_iterator it1_start(d_embedding_col_1.embedding_column,0);
   //embedding_element_iterator it1_end(d_embedding_col_1.embedding_column+len_1,len_1-1);

   //embedding_element_iterator it2_start(d_embedding_col_2.embedding_column,0);
   //embedding_element_iterator it2_end(d_embedding_col_2.embedding_column+len_2,len_2-1);

   //r2 = thrust::set_intersection(it1_start,it1_end, it2_start, it2_end,
   //                                                     thrust::device_pointer_cast<int>(d_result),
   //                                                     op1);

   std::cout << "printing result array:\n";
   print_d_array(d_result,max_len);
   len_result = r2 - thrust::device_pointer_cast<int>(d_result);
   std::cout << "intersection len = " << len_result << std::endl;

   int *h_array = new int[len_result];
   CUDA_EXEC(cudaMemcpy(h_array, d_result, len_result * sizeof(int), cudaMemcpyDeviceToHost), *logger );
   for(int i = 0; i<len_result; i++)
    std::cout << h_array[i] << " ";
   std::cout << "\n";

   CUDAMALLOC(&d_filtered_offsets,len_result * sizeof(int),*logger);
   CUDA_EXEC(cudaMemcpy(d_filtered_offsets,h_array, len_result * sizeof(int), cudaMemcpyHostToDevice), *logger );

   delete [] h_array;
   CUDAFREE(d_result, *logger);

   }
 */




struct embedding_offset_t {

  int back_link;
  int vertex_id;
  int offset;

  __device__ __host__
  bool operator <( const embedding_offset_t &r) const {
    return ( back_link < r.back_link || (back_link == r.back_link &&  vertex_id < r.vertex_id) );
  }

  std::string to_string() const {
    std::stringstream ss;
    ss << "( " << offset << " " << back_link << " " << vertex_id << " ) ";
    return ss.str();
  }

};

struct compute_embedding_offsets {

  embedding_element *d_embedding_col_1, *d_embedding_col_2;
  embedding_offset_t *d_embedding_offset_col_1, *d_embedding_offset_col_2;
  int len_1, len_2;

  __device__ __host__
  compute_embedding_offsets(embedding_element *d_embedding_col_1, embedding_element *d_embedding_col_2, embedding_offset_t *d_embedding_offset_col_1, embedding_offset_t *d_embedding_offset_col_2, int len_1, int len_2){
    this->d_embedding_col_1 = d_embedding_col_1;
    this->d_embedding_col_2 = d_embedding_col_2;
    this->d_embedding_offset_col_1 = d_embedding_offset_col_1;
    this->d_embedding_offset_col_2 = d_embedding_offset_col_2;
    this->len_1 = len_1;
    this->len_2 = len_2;
  }

  __device__ __host__
  void operator() (int idx){

    if(idx < len_1) {
      d_embedding_offset_col_1[idx].back_link = d_embedding_col_1[idx].back_link;
      d_embedding_offset_col_1[idx].vertex_id = d_embedding_col_1[idx].vertex_id;
      d_embedding_offset_col_1[idx].offset = idx;
    }

    if(idx < len_2) {
      d_embedding_offset_col_2[idx].back_link = d_embedding_col_2[idx].back_link;
      d_embedding_offset_col_2[idx].vertex_id = d_embedding_col_2[idx].vertex_id;
      d_embedding_offset_col_2[idx].offset = idx;
    }

  }

};


struct get_filtered_offsets {
  int *d_filtered_offsets;
  embedding_offset_t *d_embedding_offsets;

  __device__ __host__
  get_filtered_offsets(int *d_filtered_offsets, embedding_offset_t *d_embedding_offsets){
    this->d_filtered_offsets = d_filtered_offsets;
    this->d_embedding_offsets = d_embedding_offsets;
  }

  __device__ __host__
  void operator() (int idx){
    d_filtered_offsets[idx] = d_embedding_offsets[idx].offset;
  }

};


// Get the intersection between two embedding columns ( backward and backward )
// The output is stored in the result embedding column

void exp_intersection_bwd_bwd(embedding_extension_t d_embedding_col_1,
                              embedding_extension_t d_embedding_col_2,
                              embedding_extension_t &d_embedding_col_result)
{

  std::cout << "In backward backward intersection: the input columns:\n";
  int len_1 = d_embedding_col_1.col_length;
  int len_2 = d_embedding_col_2.col_length;
  print_d_array2(d_embedding_col_1.embedding_column, len_1); print_d_array2(d_embedding_col_2.embedding_column, len_2);

  embedding_offset_t *d_embedding_offset_col_1, *d_embedding_offset_col_2, *d_embedding_offset_result_col;
  CUDAMALLOC(&(d_embedding_offset_col_1),  len_1 * sizeof(embedding_offset_t), *logger);
  CUDAMALLOC(&(d_embedding_offset_col_2),  len_2 * sizeof(embedding_offset_t), *logger);


  int max_len = (len_1 > len_2) ? len_1 : len_2;
  cudapp::cuda_computation_parameters params = cudapp::cuda_configurator::get_computation_parameters(max_len, 512);
  compute_embedding_offsets ceo(d_embedding_col_1.embedding_column, d_embedding_col_2.embedding_column, d_embedding_offset_col_1, d_embedding_offset_col_2, len_1, len_2);
  for_each(0,max_len,ceo,params);

  print_d_array2(d_embedding_offset_col_1, len_1); print_d_array2(d_embedding_offset_col_2, len_2);

  int min_len = (len_1 < len_2) ? len_1 : len_2;
  CUDAMALLOC(&d_embedding_offset_result_col,  min_len * sizeof(embedding_offset_t), *logger);
  std::cout << "calling set_intersection " << std::endl;
  thrust::device_ptr<embedding_offset_t>  r1 = thrust::set_intersection(thrust::device_pointer_cast<embedding_offset_t>(d_embedding_offset_col_1),
                                                                        thrust::device_pointer_cast<embedding_offset_t>(d_embedding_offset_col_1 + len_1),
                                                                        thrust::device_pointer_cast<embedding_offset_t>(d_embedding_offset_col_2),
                                                                        thrust::device_pointer_cast<embedding_offset_t>(d_embedding_offset_col_2 + len_2),
                                                                        thrust::device_pointer_cast<embedding_offset_t>(d_embedding_offset_result_col));

  int len_result = r1 - thrust::device_pointer_cast<embedding_offset_t>(d_embedding_offset_result_col);

  print_d_array2(d_embedding_offset_result_col, len_result);

  int *d_filtered_offsets;
  CUDAMALLOC(&d_filtered_offsets, len_result * sizeof(int), *logger);
  cudapp::cuda_computation_parameters params2 = cudapp::cuda_configurator::get_computation_parameters(len_result, 512);
  for_each(0,len_result,get_filtered_offsets(d_filtered_offsets,d_embedding_offset_result_col),params2);


  int* h_array = new int[len_result];
  CUDA_EXEC(cudaMemcpy(h_array, d_filtered_offsets, len_result * sizeof(int), cudaMemcpyDeviceToHost), *logger );

  for(int i = 0; i<len_result; i++)
    std::cout << h_array[i] << " ";
  std::cout << "\n";

  delete [] h_array;

  d_embedding_col_result.filtered_emb_offsets = d_filtered_offsets;
  d_embedding_col_result.filtered_emb_offsets_length = len_result;
  d_embedding_col_result.ext_type = BKWD;

  CUDAFREE(d_embedding_offset_col_1,*logger);
  CUDAFREE(d_embedding_offset_col_2,*logger);
  CUDAFREE(d_embedding_offset_result_col,*logger);

}

struct set_support_flag {

  int *flags;
  int max_vertex_id;
  embedding_element *col;
  int ext_type;
  __device__ __host__
  set_support_flag(embedding_extension_t ee, int *flags, int max_vertex_id){
    col = ee.embedding_column;
    ext_type = ee.ext_type;
    this->flags = flags;
    this->max_vertex_id = max_vertex_id;
  }

  __device__ __host__
  void operator() (int idx){
    if(ext_type == FRWD) {
      int graph_id = col[idx].vertex_id / max_vertex_id;
      flags[ graph_id] = 1;
    } else if(ext_type == BKWD) {
      int graph_id = col[idx].vertex_id / max_vertex_id;
      flags[ graph_id] = 1;
    } // if-else
  } // operator()

};


int compute_support_for_fwd_ext(embedding_extension_t &d_emb,
                                int max_vertex_id)
{

  //throw std::runtime_error("compute_support_for_fwd_ext not implemented");

  if(d_emb.ext_type != FRWD) throw std::runtime_error("compute_support_for_fwd_ext can't compute support for a backward extension");

  if(d_emb.col_length == 0) {
    d_emb.support = 0;
    return 0;
  }

  TRACE(*logger, "emb col length: " << d_emb.col_length  << "; embedding ptr: " << d_emb.embedding_column);
  embedding_element last_embedding_element;
  CUDA_EXEC(cudaMemcpy(&last_embedding_element,
                       d_emb.embedding_column + d_emb.col_length - 1,
                       sizeof(embedding_element),
                       cudaMemcpyDeviceToHost), *logger);
  int num_graphs = (last_embedding_element.vertex_id / max_vertex_id) + 1;

  int *support_flags;
  CUDAMALLOC(&support_flags, sizeof(int) * num_graphs,*logger);
  CUDA_EXEC(cudaMemset(support_flags, 0, sizeof(int) * num_graphs), *logger);
  TRACE(*logger, "storing graph info, num_graphs: " << num_graphs << "; d_emb.col_length: " << d_emb.col_length);
  cudapp::cuda_computation_parameters param = cudapp::cuda_configurator::get_computation_parameters(d_emb.col_length, 512);
  set_support_flag ssf(d_emb, support_flags, max_vertex_id);
  thrust::counting_iterator<int> b1(0);
  thrust::counting_iterator<int> e1(d_emb.col_length);
  thrust::for_each(b1, e1, ssf);

  TRACE(*logger, "reducing graph info");
  d_emb.support = thrust::reduce(thrust::device_pointer_cast<int>(support_flags),
                                 thrust::device_pointer_cast<int>(support_flags + num_graphs),
                                 0,
                                 thrust::plus<int>());
  if(!d_emb.graph_id_list) {

    TRACE(*logger, "getting the graph ids");
    int *graph_id_list;
    CUDAMALLOC(&graph_id_list, sizeof(int) * d_emb.support,*logger);
    thrust::counting_iterator<int> b(0);
    thrust::counting_iterator<int> e = b + num_graphs;
    thrust::copy_if(b,        //starts at 0
                    e,       //ends at num_graphs - 1
                    thrust::device_pointer_cast<int>(support_flags),       // key flags 100100010 etc
                    thrust::device_pointer_cast<int>(graph_id_list),        // output graph id arrays
                    is_one());

    d_emb.graph_id_list = graph_id_list;
  }
  CUDAFREE(support_flags,*logger);
  return d_emb.support;

}


int compute_support_for_bwd_ext(embedding_extension_t &d_emb,
                                int max_vertex_id)
{

  throw std::runtime_error("compute_support_for_bwd_ext not implemented");

  //if(d_emb.ext_type != BKWD) throw std::runtime_error("compute_support_for_bwd_ext can't compute support for a forward extension");

  //return d_emb.support;
}


int get_graph_id_list_intersection_size(embedding_extension_t &d_emb1,
                                        embedding_extension_t &d_emb2,
                                        int max_vertex_id)
{
  if(!d_emb1.graph_id_list) compute_support_for_fwd_ext(d_emb1,max_vertex_id);
  if(!d_emb2.graph_id_list) compute_support_for_fwd_ext(d_emb2,max_vertex_id);

  int min_len = (d_emb1.support < d_emb2.support) ? d_emb1.support : d_emb2.support;
  int *d_result;
  CUDAMALLOC(&d_result,  min_len * sizeof(int), *logger);
  //std::cout << "calling set_intersection " << std::endl;
  thrust::device_ptr<int>  r1 = thrust::set_intersection(thrust::device_pointer_cast<int>(d_emb1.graph_id_list),
                                                         thrust::device_pointer_cast<int>(d_emb1.graph_id_list + d_emb1.support),
                                                         thrust::device_pointer_cast<int>(d_emb2.graph_id_list),
                                                         thrust::device_pointer_cast<int>(d_emb2.graph_id_list + d_emb2.support),
                                                         thrust::device_pointer_cast<int>(d_result));
  int len = r1 - thrust::device_pointer_cast<int>(d_result);
  CUDAFREE(d_result,*logger);

  return len;
}


} // namespace gspan_cuda

