#include <embedding_lists.hpp>
#include <cuda_graph_types.hpp>
#include <cuda_computation_parameters.hpp>
#include <kernel_execution.hpp>
#include <thrust/scan.h>
#include <cuda_tools.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>

#include <cuda_configurator.hpp>

#include <cuda.h>

using namespace types;

namespace gspan_cuda {


/**
 * A functor that just checks whether the 'from' vertex in the dfs
 * code is valid and has the same vertex label as the 'fromlabel' in
 * the dfs.
 *
 */
struct check_vertex_existence {
  graph_database_cuda gdb;
  types::DFS dfs;
  int *d_vertex_valid;

  check_vertex_existence(graph_database_cuda gdb, types::DFS dfs, int *d_vertex_valid) : gdb(gdb), dfs(dfs), d_vertex_valid(d_vertex_valid) {
  }

  __device__
  void operator()(int vertex_id) {
    //-1 == invalid offset.
    //printf("%d\n", vertex_id);
    if(gdb.vertex_is_valid(vertex_id)) {
      d_vertex_valid[vertex_id] = (gdb.vertex_labels[vertex_id] == dfs.fromlabel);
    } else {
      d_vertex_valid[vertex_id] = 0;
    }
  } // operator()
};


/**
 * stores in thread_vertex_mapping[i] the vertex id that should be
 * assigned to thread i.
 *
 * PRECONDITION: threads are mapped to vertex ids
 *
 * each thread (aka vertex) then checks whether is valid and if yes it
 * stores its vertex id in thread_vertex_mapping.
 */
struct store_thread_to_vertex_id_map {
  int *vertex_order;
  int *thread_vertex_mapping;
  int *vertex_valid;
  store_thread_to_vertex_id_map(int *vo, int *tvm, int *vertex_valid) {
    vertex_order = vo;
    thread_vertex_mapping = tvm;
    this->vertex_valid = vertex_valid;
  }

  __device__
  void operator()(int v_idx) {
    if(vertex_valid[v_idx]) {
      int idx = vertex_order[v_idx]; // read the order of the vertex
      thread_vertex_mapping[idx] = v_idx; // and map the thread to the vertex id
    } // if
  } // operator
};



/**
 * PRECONDITION: each thread is allocated to one element in
 * thread_vertex_mapping.
 *
 * Thread i then reads vertex id from thread_vertex_mapping[i] and
 * goes through the neighborhood, checking the 'edge' and 'to' label.
 *
 * The length of thread_vertex_mapping does not needs to be stored,
 * because thread_idx < length of thread_vertex_mapping.
 *
 * The template has two modes of operation according to the value of
 * store_embed_data:
 *
 * 1) The operator is a template with one argument store_embed_data.
 *    If store_embed_data is true the operator stores the vertex id of
 *    'from' and 'to' into the first_embed_column and
 *    second_embed_column (respectively). The position of the stored
 *    edge is taken from valid_edge.
 *
 * 2) If the store_embed_data is false the thread i just stores 1 into
 *    valid_edge[i]. This can be then used for "packing" or
 *    "remapping" the edges to threads.
 *
 */
template<bool store_embed_data>
struct valid_edges_op {
  types::DFS dfs_element;
  int *thread_vertex_mapping;
  graph_database_cuda gdb;
  int *valid_edge;

  embedding_element *first_embed_column;
  embedding_element *second_embed_column;

  valid_edges_op(types::DFS d,
                 int *thread_vertex_mapping,
                 int *valid_edge,
                 embedding_element *first,
                 embedding_element *second,
                 graph_database_cuda gdb) : gdb(gdb)
  {
    dfs_element = d;
    this->thread_vertex_mapping = thread_vertex_mapping;
    this->valid_edge = valid_edge;
    this->first_embed_column = first;
    this->second_embed_column = second;
    //this->gdb = gdb;
  } // create_embedding




  // last vertex !?!?!?!?
  // the d_vertex_offsets[n] is probably filled with the size of d_edges
  __device__
  void operator()(int thread_idx) {
    int vertex_idx = thread_vertex_mapping[thread_idx];
    int vertex_offset = gdb.vertex_offsets[vertex_idx];
    int vertex_degree = gdb.get_vertex_degree(vertex_idx);

    //printf("thread_idx: %d; vertex_idx: %d;\n", thread_idx, vertex_idx);

    int vertex_label =  gdb.vertex_labels[vertex_idx];
    //printf("thread_idx: %d;vertex_idx: %d;vertex_offsets: %d; vertex_degree: %d; vertex_label: %d\n", thread_idx, vertex_idx, vertex_offset, vertex_degree, vertex_label);


    for(int i = 0; i < vertex_degree; i++) {
      int other_vertex_id = gdb.edges[vertex_offset + i];
      int other_vertex_label = gdb.vertex_labels[other_vertex_id];
      int edge_label = gdb.edges_labels[vertex_offset + i];

      if(dfs_element.fromlabel == vertex_label &&
         dfs_element.elabel == edge_label &&
         dfs_element.tolabel == other_vertex_label) {
        if(store_embed_data == false) {
          valid_edge[vertex_offset + i] = 1;
        } else {
          int idx = valid_edge[vertex_offset + i];
          first_embed_column[idx].vertex_id = vertex_idx;
          first_embed_column[idx].back_link = -1;

          second_embed_column[idx].vertex_id = other_vertex_id;
          second_embed_column[idx].back_link = idx;
        } // if-else
      } else {
        if(store_embed_data == false) {
          valid_edge[vertex_offset + i] = 0;
        }
      } // else
    } // for i
  } // operator
};


/**
 * Algorithm:
 *
 *
 * TODO: fix the execution configuration, it should be given as the argument.
 *
 */
void create_first_embeddings(types::DFS first_dfs,
                             graph_database_cuda *gdb,
                             embedding_list_columns *embeddings,
                             cudapp::cuda_configurator *exec_conf)
{
  // TODO: is this allocation of threads necessary !?!?!?!?!?
  // create allocation of threads to vertices

  Logger *logger = Logger::get_logger("FE");

  TRACE(*logger, "first projection for dfs element: " << first_dfs.to_string());


  int *d_vertex_order = 0;
  int *d_vertex_valid = 0;
  CUDAMALLOC(&d_vertex_order, sizeof(int) * (gdb->max_graph_vertex_count * gdb->db_size + 1), *logger);
  CUDAMALLOC(&d_vertex_valid, sizeof(int) * (gdb->max_graph_vertex_count * gdb->db_size + 1), *logger);

  // find all vertices that has the correct starting label and compute
  // the ordering of these vertices using exclusive_scan
  check_vertex_existence check(*gdb, first_dfs, d_vertex_valid);
  TRACE4(*logger, "transform, start idx: " << 0 << "; end idx: " << gdb->max_graph_vertex_count * gdb->db_size);
  cudapp::cuda_computation_parameters valid_vertex_params = exec_conf->get_exec_config("create_first_embeddings.valid_vertex", gdb->max_graph_vertex_count * gdb->db_size);
  //cudapp::transform(0, gdb->max_graph_vertex_count * gdb->db_size, check, d_vertex_valid, valid_vertex_params);
  cudapp::for_each(0, gdb->max_graph_vertex_count * gdb->db_size, check, valid_vertex_params);

  //thrust::for_each(thrust::counting_iterator<int>(0),
  //thrust::counting_iterator<int>(gdb->max_graph_vertex_count * gdb->db_size),
  //check);
  TRACE4(*logger, "finished thrust::transform");

  int last_valid = 0;
  CUDA_EXEC(cudaMemcpy(&last_valid, d_vertex_valid + gdb->max_graph_vertex_count * gdb->db_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  TRACE(*logger, "last_valid: " << last_valid << "; offset: " << (gdb->max_graph_vertex_count * gdb->db_size - 1));

  //print_d_array(d_vertex_valid, gdb->max_graph_vertex_count * gdb->db_size, "d_vertex_valid");


  TRACE4(*logger, "executing exclusive_scan");
  thrust::exclusive_scan(thrust::device_pointer_cast(d_vertex_valid), // input begin
                         thrust::device_pointer_cast(d_vertex_valid + gdb->max_graph_vertex_count * gdb->db_size), // input end
                         thrust::device_pointer_cast(d_vertex_order), // ouput begin
                         0, // init value
                         thrust::plus<int>());

  // store the number of vertices with particular 'fromlabel' label
  int vertices_with_label = 0;
  CUDA_EXEC(cudaMemcpy(&vertices_with_label, d_vertex_order + gdb->max_graph_vertex_count * gdb->db_size - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  TRACE3(*logger, "vertices with label: " << vertices_with_label
         << "; total vertices: " << gdb->vertex_count
         << "; gdb->max_graph_vertex_count: " << gdb->max_graph_vertex_count);
  vertices_with_label += last_valid;

  // PRECONDITION: the number of edges with particular combination of
  // labels is much lower then the total number of edges
  //
  // "compress" the thread->vertex mapping
  int *d_thread_vertex_mapping = 0;
  CUDAMALLOC(&d_thread_vertex_mapping, sizeof(int) * vertices_with_label, *logger);
  TRACE4(*logger, "cuda: store_thread_to_vertex_id_map");
  store_thread_to_vertex_id_map store_map(d_vertex_order, d_thread_vertex_mapping, d_vertex_valid);
  cudapp::cuda_computation_parameters store_map_params = exec_conf->get_exec_config("create_first_embeddings.store_map", gdb->max_graph_vertex_count * gdb->db_size);
  cudapp::for_each(0, gdb->max_graph_vertex_count * gdb->db_size, store_map, store_map_params);


  // now we are prepared to make the first embeddings, i.e., the first two columns of the embeddings lists.
  // 1) find out valid edges and create packed mapping of edges to the first two columns
  TRACE(*logger, "cuda: 1) find out valid edges and create packed mapping of edges to the first two columns");

  int *valid_edge = 0;
  CUDAMALLOC(&valid_edge, sizeof(int) * gdb->edges_sizes, *logger);
  CUDA_EXEC(cudaMemset(valid_edge, 0, sizeof(int) * gdb->edges_sizes), *logger);

  embedding_element *d_first_column = 0;
  embedding_element *d_second_column = 0;

  // find valid edges and store '1' into an array valid_edge
  // then execute exclusive_scan which computes the ordering of the edges in the d_first_column and d_second_column.
  valid_edges_op<false> find_valid_edge_op(first_dfs, d_thread_vertex_mapping, valid_edge, d_first_column, d_second_column, *gdb);
  TRACE4(*logger, "finding valid edges");
  cudapp::cuda_computation_parameters find_valid_edge_params = exec_conf->get_exec_config("create_first_embeddings.find_valid_edge", vertices_with_label);
  for_each(0, vertices_with_label, find_valid_edge_op, find_valid_edge_params);

  int add_total_edge_count = 0;
  CUDA_EXEC(cudaMemcpy(&add_total_edge_count, valid_edge + gdb->edges_sizes - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);


  TRACE4(*logger, "scheduling by executing exclusive_scan");
  thrust::exclusive_scan(thrust::device_pointer_cast(valid_edge), // input begin
                         thrust::device_pointer_cast(valid_edge + gdb->edges_sizes),
                         thrust::device_pointer_cast(valid_edge)); // output begin

  int total_edge_count = 0;
  CUDA_EXEC(cudaMemcpy(&total_edge_count, valid_edge + gdb->edges_sizes - 1, sizeof(int), cudaMemcpyDeviceToHost), *logger);
  total_edge_count += add_total_edge_count;


  CUDAMALLOC(&d_first_column, sizeof(types::embedding_element) * total_edge_count, *logger);
  CUDAMALLOC(&d_second_column, sizeof(types::embedding_element) * total_edge_count, *logger);


  // Almost finally: copy the vertex ids of the edge (i.e., dfs code elements) into the two columns
  valid_edges_op<true> store_edge_op(first_dfs, d_thread_vertex_mapping, valid_edge, d_first_column, d_second_column, *gdb);
  TRACE(*logger, "storing edges into first two columns of the embedding storage");
  cudapp::cuda_computation_parameters edge_store_exec_config = exec_conf->get_exec_config("create_first_embeddings.edge_store", vertices_with_label);
  for_each(0, vertices_with_label, store_edge_op, edge_store_exec_config);


  // Finally: prepare the the memory on the device, copy all the pointers into the embeddings datastructure
  embeddings->columns_count = 2;
  CUDAMALLOC(&embeddings->columns_lengths, sizeof(int) * embeddings->columns_count, *logger);

  TRACE(*logger, "total_edge_count: " << total_edge_count);
  int col_lengths[2] = {total_edge_count, total_edge_count};
  CUDA_EXEC(cudaMemcpy(embeddings->columns_lengths, col_lengths, sizeof(int) * embeddings->columns_count, cudaMemcpyHostToDevice), *logger);
  CUDAMALLOC(&embeddings->columns, sizeof(embedding_element*) * embeddings->columns_count, *logger);

  embedding_element *cols[2] = {d_first_column, d_second_column};
  CUDA_EXEC(cudaMemcpy(embeddings->columns, cols, sizeof(embedding_element*) * embeddings->columns_count, cudaMemcpyHostToDevice), *logger);

  CUDAMALLOC(&embeddings->dfscode, sizeof(types::DFS), *logger);
  CUDA_EXEC(cudaMemcpy(embeddings->dfscode, &first_dfs, sizeof(types::DFS), cudaMemcpyHostToDevice), *logger);
  embeddings->dfscode_length = 1;

  CUDAFREE(d_vertex_order, *logger);
  CUDAFREE(d_vertex_valid, *logger);
  CUDAFREE(d_thread_vertex_mapping, *logger);
  CUDAFREE(valid_edge, *logger);

  TRACE(*logger, "create_first_embeddings finished");


} // create_first_embeddings

} // namespace gspan_cuda
