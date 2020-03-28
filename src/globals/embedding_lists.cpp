#include <embedding_lists.hpp>
#include <cuda_tools.hpp>
#include <logger.hpp>


namespace types {

static Logger *logger = Logger::get_logger("EL");

void embedding_list_columns::d_half_deallocate()
{
  CUDAFREE(columns, *logger);
  columns = 0;
  CUDAFREE(dfscode, *logger);
  dfscode = 0;
  CUDAFREE(columns_lengths, *logger);
  columns_lengths = 0;
}

/**
 * Copy only parts of the embedding_list_columns that are not large:
 * 1) dfscode; 2) just pointers in columns (the data are not
 * duplicated); 3)columns_lengths
 * + all other integers.
 *
 */
embedding_list_columns embedding_list_columns::d_get_half_copy()
{
  if(located_on_host == true) {
    throw std::runtime_error("d_get_half_copy does not work on embedding_list_columns located in host memory.");
  }
  embedding_list_columns copy(false);
  copy.located_on_host = false;
  copy.columns_count = columns_count;
  types::DFS *h_tmp_dfscode = new types::DFS[dfscode_length];


  TRACE3(*logger, "copying dfscode");
  TRACE3(*logger, "columns_count: " << columns_count);
  CUDAMALLOC(&copy.dfscode, sizeof(types::DFS) * (dfscode_length), *logger);
  CUDA_EXEC(cudaMemcpy(h_tmp_dfscode, dfscode, sizeof(types::DFS) * (dfscode_length), cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(copy.dfscode, h_tmp_dfscode, sizeof(types::DFS) * (dfscode_length), cudaMemcpyHostToDevice), *logger);
  delete [] h_tmp_dfscode;
  copy.dfscode_length = dfscode_length;


  TRACE3(*logger, "copying columns");
  embedding_element **h_tmp_columns = new embedding_element *[columns_count];
  CUDAMALLOC(&copy.columns, sizeof(embedding_element*) * columns_count, *logger);
  CUDA_EXEC(cudaMemcpy(h_tmp_columns, columns, sizeof(embedding_element*) * columns_count, cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(copy.columns, h_tmp_columns, sizeof(embedding_element*) * columns_count, cudaMemcpyHostToDevice), *logger);
  delete [] h_tmp_columns;


  TRACE3(*logger, "copying columns_lengths");
  int *h_tmp_columns_lengths = new int[columns_count];
  CUDAMALLOC(&copy.columns_lengths, sizeof(int) * columns_count, *logger);
  TRACE3(*logger, "allocated: " << copy.columns_lengths);
  CUDA_EXEC(cudaMemcpy(h_tmp_columns_lengths, columns_lengths, sizeof(int) * columns_count, cudaMemcpyDeviceToHost), *logger);
  CUDA_EXEC(cudaMemcpy(copy.columns_lengths, h_tmp_columns_lengths, sizeof(int) * columns_count, cudaMemcpyHostToDevice), *logger);
  delete [] h_tmp_columns_lengths;

  return copy;
}

/**
 * Extends (host stored) embeddings by one column stored on HOST in
 * new_col. The length of the new column is given in new_col_length.
 *
 *
 *
 */
void embedding_list_columns::h_extend_by_one_column(types::DFS dfs_elem, embedding_element *new_col, int new_col_length)
{
  if(located_on_host == false) {
    throw std::runtime_error("cannot extend host embedding_list_columns: stored in wrong memory.");
  } // if
  DEBUG(*logger, "h_extend_by_one_column, columns_count: " << columns_count);
  DEBUG(*logger, "extending dfs code");
  if(dfscode != 0) {
    std::cout << "dfscode_length: " << dfscode_length << std::endl;
    types::DFS *new_dfscode = new types::DFS[dfscode_length + 1];
    memcpy(new_dfscode, dfscode, sizeof(types::DFS) * (dfscode_length));
    memcpy(new_dfscode + dfscode_length, &dfs_elem, sizeof(types::DFS));
    delete [] dfscode;
    dfscode = new_dfscode;
    dfscode_length++;
  } else {
    abort();
  }


  DEBUG(*logger, "columns_count: " << columns_count);
  embedding_element **new_columns = new embedding_element *[columns_count + 1];
  memcpy(new_columns, columns, sizeof(embedding_element*) * columns_count);
  new_columns[columns_count] = new_col;
  delete [] columns;
  columns = new_columns;


  DEBUG(*logger, "extending columns_lengths");
  int *new_columns_lengths = new int[columns_count + 1];
  memcpy(new_columns_lengths, columns_lengths, sizeof(int) * columns_count);
  new_columns_lengths[columns_count] = new_col_length;
  delete [] columns_lengths;
  columns_lengths = new_columns_lengths;
  columns_count++;
} // embedding_list_columns::h_extend_by_one_column


void embedding_list_columns::d_replace_last_column(embedding_element *new_col, int new_col_length)
{
  if(located_on_host == true) {
    throw std::runtime_error("cannot extend host embedding_list_columns: stored in wrong memory.");
  } // if

  CUDA_EXEC(cudaMemcpy(columns + columns_count - 1, &new_col, sizeof(embedding_element*), cudaMemcpyHostToDevice), *logger);
  CUDA_EXEC(cudaMemcpy(columns_lengths + columns_count - 1, &new_col_length, sizeof(int), cudaMemcpyHostToDevice), *logger);

}


/**
 * Extends (device stored) embeddings by one column stored on DEVICE in new_col. The
 * length of the new column is given in new_col_length.
 *
 */
void embedding_list_columns::d_extend_by_one_column(types::DFS dfs_elem, embedding_element *new_col, int new_col_length)
{
  if(located_on_host == true) {
    throw std::runtime_error("cannot extend host embedding_list_columns: stored in wrong memory.");
  } // if

  TRACE3(*logger, "extending dfscode by " << dfs_elem.to_string() << "; dfscode_length: " << dfscode_length);
  types::DFS *new_dfscode = new types::DFS[dfscode_length + 1];
  CUDA_EXEC(cudaMemcpy(new_dfscode, dfscode, sizeof(types::DFS) * (dfscode_length), cudaMemcpyDeviceToHost), *logger);
  memcpy(new_dfscode + dfscode_length, &dfs_elem, sizeof(types::DFS));
  CUDAFREE(dfscode, *logger);
  dfscode = 0;
  CUDAMALLOC(&dfscode, sizeof(types::DFS) * (dfscode_length + 1), *logger);
  for(int i = 0; i < dfscode_length + 1; i++) {
    TRACE(*logger, "new_dfscode[" << i << "]: " << new_dfscode[i].to_string());
  }
  CUDA_EXEC(cudaMemcpy(dfscode, new_dfscode, sizeof(types::DFS) * (dfscode_length + 1), cudaMemcpyHostToDevice), *logger);
  delete [] new_dfscode;
  dfscode_length++;


  TRACE3(*logger, "extending columns");
  embedding_element **new_columns = new embedding_element *[columns_count + 1];
  CUDA_EXEC(cudaMemcpy(new_columns, columns, sizeof(embedding_element *) * columns_count, cudaMemcpyDeviceToHost), *logger);
  new_columns[columns_count] = new_col;
  CUDAFREE(columns, *logger);
  CUDAMALLOC(&columns, sizeof(embedding_element*) * (columns_count + 1), *logger);
  CUDA_EXEC(cudaMemcpy(columns, new_columns, sizeof(embedding_element*) * (columns_count + 1), cudaMemcpyHostToDevice), *logger);
  delete [] new_columns;


  TRACE3(*logger, "extending columns_lengths");
  int *new_columns_lengths = new int[columns_count + 1];
  CUDA_EXEC(cudaMemcpy(new_columns_lengths, columns_lengths, sizeof(int) * columns_count, cudaMemcpyDeviceToHost), *logger);
  new_columns_lengths[columns_count] = new_col_length;
  CUDAFREE(columns_lengths, *logger);
  columns_lengths = 0;
  CUDAMALLOC(&columns_lengths, sizeof(int) * (columns_count + 1), *logger);
  CUDA_EXEC(cudaMemcpy(columns_lengths, new_columns_lengths, sizeof(int) * (columns_count + 1), cudaMemcpyHostToDevice), *logger);
  delete [] new_columns_lengths;


  columns_count++;
}


std::string embedding_list_columns::embedding_to_string(int row) const
{
  if(located_on_host == false) {
    throw std::runtime_error("cannot print embedding: stored in wrong memory.");
  } // if
  int col = columns_count - 1;
  std::stringstream ss;
  ss << "columns_count: " << columns_count << "; ";
  for(; col >= 0; col--) {
    ss << "(" << columns[col][row].vertex_id << ", " << columns[col][row].back_link << "); ";
    row = columns[col][row].back_link;
  } // for col

  return ss.str();
} // embedding_list_columns::print_embedding

void embedding_list_columns::copy_to_device(embedding_list_columns *elc)
{
  if(located_on_host == false || elc->located_on_host == true) {
    throw std::runtime_error("cannot copy the data: source or destination stored in wrong memory.");
  }

  elc->delete_from_device();

  CUDAMALLOC(&elc->columns_lengths, sizeof(int) * columns_count, *logger);
  CUDA_EXEC(cudaMemcpy(elc->columns_lengths, columns_lengths, sizeof(int) * columns_count, cudaMemcpyHostToDevice), *logger);

  CUDAMALLOC(&elc->dfscode, sizeof(types::DFS) * dfscode_length, *logger);
  CUDA_EXEC(cudaMemcpy(elc->dfscode, dfscode, sizeof(types::DFS) * dfscode_length, cudaMemcpyHostToDevice), *logger);
  elc->dfscode_length = dfscode_length;

  CUDAMALLOC(&elc->columns, sizeof(embedding_element*) * columns_count, *logger);
  for(int i = 0; i < columns_count; i++) {
    embedding_element *curr_col_ptr = 0;
    CUDAMALLOC(&curr_col_ptr, sizeof(embedding_element) * columns_lengths[i], *logger);
    CUDA_EXEC(cudaMemcpy(curr_col_ptr, columns[i], sizeof(embedding_element) * columns_lengths[i], cudaMemcpyHostToDevice), *logger);
    CUDA_EXEC(cudaMemcpy(elc->columns + i, &curr_col_ptr, sizeof(embedding_element*), cudaMemcpyHostToDevice), *logger);
  } // for i

  elc->columns_count = columns_count;
} // embedding_list_columns::copy_to_device



void embedding_list_columns::copy_from_device(embedding_list_columns *d_elc)
{
  if(located_on_host == false || d_elc->located_on_host == true) {
    throw std::runtime_error("cannot copy the data: source or destination stored in wrong memory.");
  } // if

  delete_from_host();

  columns_count = d_elc->columns_count;
  dfscode_length = d_elc->dfscode_length;
  dfscode = new types::DFS[d_elc->dfscode_length];
  CUDA_EXEC(cudaMemcpy(dfscode, d_elc->dfscode, sizeof(types::DFS) * (d_elc->dfscode_length), cudaMemcpyDeviceToHost), *logger);

  columns_lengths = new int[columns_count];
  CUDA_EXEC(cudaMemcpy(columns_lengths, d_elc->columns_lengths, sizeof(int) * columns_count, cudaMemcpyDeviceToHost), *logger);

  columns = new embedding_element *[columns_count];
  embedding_element **col_ptrs = new embedding_element *[columns_count];
  CUDA_EXEC(cudaMemcpy(col_ptrs, d_elc->columns, sizeof(embedding_element*) * columns_count, cudaMemcpyDeviceToHost), *logger);

  for(int i = 0; i < columns_count; i++) {
    columns[i] = new embedding_element[columns_lengths[i]];
    CUDA_EXEC(cudaMemcpy(columns[i], col_ptrs[i], sizeof(embedding_element) * columns_lengths[i], cudaMemcpyDeviceToHost), *logger);
  } // for i

  delete [] col_ptrs;
} // embedding_list_columns::copy_from_device



void embedding_list_columns::delete_from_host()
{
  if(located_on_host == false) {
    throw std::runtime_error("delete_from_host: data located in device memory.");
  }

  delete [] dfscode;
  dfscode = 0;

  delete [] columns_lengths;
  columns_lengths = 0;

  for(int i = 0; i < columns_count; i++) {
    delete [] columns[i];
  } // for i

  delete [] columns;
  columns = 0;

  columns_count = -1;
}


void embedding_list_columns::delete_from_device()
{
  if(located_on_host == true) {
    throw std::runtime_error("delete_from_device: data located in host memory.");
  }

  CUDAFREE(dfscode, *logger);
  dfscode = 0;

  CUDAFREE(columns_lengths, *logger);
  columns_lengths = 0;

  if(columns_count > 0) {
    embedding_element **h_cols = new embedding_element *[columns_count];
    CUDA_EXEC(cudaMemcpy(h_cols, columns, sizeof(embedding_element*) * columns_count, cudaMemcpyDeviceToHost), *logger);
    for(int i = 0; i < columns_count; i++) {
      CUDAFREE(h_cols[i], *logger);
    } // for i
    delete [] h_cols;
  }

  CUDAFREE(columns, *logger);
  columns = 0;

  columns_count = -1;
}


std::string embedding_list_columns::to_string() const
{
  if(located_on_host == false) {
    throw std::runtime_error("to_string: data located in device memory.");
  }

  std::stringstream ss;
  ss << "columns_count: " << columns_count << "; dfscode: ";
  for(int i = 0; i < dfscode_length; i++) ss << dfscode[i].to_string() << "; ";
  ss << endl;
  for(int i = 0; i < columns_count; i++) {
    ss << "col: " << i << " [";
    for(int j = 0; j < columns_lengths[i]; j++) {
      ss << j << ":" << columns[i][j].to_string() << " ";
    } // for j
    ss << "]" << endl;
  } // for i
  return ss.str();
}


std::string embedding_list_columns::to_string_with_labels(const types::graph_database_cuda &gdb, const types::DFSCode &code) const
{
  if(located_on_host == false) {
    throw std::runtime_error("to_string_with_labels: data located in device memory.");
  }
  if(gdb.located_on_host == false) {
    throw std::runtime_error("to_string_with_labels: database located in device memory.");
  }

  int last_col_size = columns_lengths[columns_count - 1];

  std::stringstream ss;

  for(int embed = 0; embed < last_col_size; embed++) {
    int curr_row = embed;
    ss << embed << ", gid " << gdb.get_graph_id(columns[columns_count - 1][curr_row].vertex_id) << ": ";
    for(int col = columns_count - 1; col >= 0; col--) {
      curr_row = columns[col][curr_row].back_link;
    }
    for(int i = 0; i < code.size(); i++) {
      int from = code[i].from;
      int to = code[i].to;
      int from_row = -1;
      int to_row = -1;
      int curr_row = embed;
      int col = 0;
      for(col = columns_count - 1; col >= 0; col--) {
        if(col == from) from_row = curr_row;
        if(col == to) to_row = curr_row;
        curr_row = columns[col][curr_row].back_link;
      } // for i

      int from_vertex = columns[from][from_row].vertex_id;
      int to_vertex = columns[to][to_row].vertex_id;
      int elabel = gdb.get_edge_label(from_vertex, to_vertex);
      ss << "(" << gdb.get_vertex_label(from_vertex) << ") " << from_vertex << " = " << elabel << " = " << to_vertex << " (" << gdb.get_vertex_label(to_vertex) << ");  ";
    } // for i
    ss << endl;
  } // for embed

  return ss.str();
}



} // namespace types

