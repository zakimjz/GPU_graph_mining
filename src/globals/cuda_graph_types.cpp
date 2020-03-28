#include <cuda.h>


#include <cuda_graph_types.hpp>
#include <graph_types.hpp>
#include <types.hpp>
#include <logger.hpp>
#include <cuda_tools.hpp>


#include <numeric>
#include <cassert>

namespace types {

static Logger *logger = Logger::get_logger("GDB_CUDA");


__host__
std::string graph_database_cuda::graph_to_string(int grph_id) const
{
  std::stringstream ss;


  int vid_from = grph_id * max_graph_vertex_count;
  int vid_to = vid_from + max_graph_vertex_count;
  for(int vid = vid_from; vid < vid_to; vid++) {
    if(get_vertex_label(vid) == -1) continue;
    ss << "vertex: " << vid << "; label: " << get_vertex_label(vid) <<  "; edges: ";
    int n_offset = vertex_offsets[vid];
    for(int n = 0; n < get_vertex_degree(vid); n++) {
      int nvid = edges[n_offset + n];
      int elabel = edges_labels[n_offset + n];
      ss << nvid << "/" << elabel << "(" << get_vertex_label(nvid) << ")  ";
    } // for n
    ss << std::endl;
  } // for i
  return ss.str();
} // graph_database_cuda::graph_to_string


graph_database_cuda & graph_database_cuda::operator=(const graph_database_cuda &other) {
  if(other.located_on_host != located_on_host) {
    throw std::runtime_error("cannot assign to a database located on host(device) a database located on device(host).");
  } // if

  if(located_on_host) {
    delete_from_host();
  } else {
    delete_from_device();
  }

  edges_sizes = other.edges_sizes;
  edges = other.edges;
  edges_labels = other.edges_labels;
  vertex_labels = other.vertex_labels;
  vertex_offsets = other.vertex_offsets;
  max_graph_vertex_count = other.max_graph_vertex_count;
  vertex_count = other.vertex_count;
  db_size = other.db_size;
  return *this;
}


graph_database_cuda graph_database_cuda::create_from_host_representation(const types::graph_database_t &gdb)
{
  graph_database_cuda result(true);
  result.edges_sizes = 0;
  result.vertex_count = 0;
  result.max_graph_vertex_count = 0;
  for(int i = 0; i < gdb.size(); i++) {
    if(gdb[i].vertex_size() > result.max_graph_vertex_count) {
      result.max_graph_vertex_count = gdb[i].vertex_size();
    } // if
    result.edges_sizes += gdb[i].edge_size();
    result.vertex_count += gdb[i].vertex_size();
  } // for i

  result.edges_sizes = 2 * result.edges_sizes;

  //DEBUG(*logger, "max_graph_vertex_count: " << result.max_graph_vertex_count);
  //DEBUG(*logger, "edges_sizes: " << result.edges_sizes);
  //DEBUG(*logger, "vertex_count: " << result.vertex_count);

  result.edges = new int[result.edges_sizes];
  result.edges_labels = new int[result.edges_sizes];


  result.vertex_offsets = new int[result.max_graph_vertex_count * gdb.size() + 1];
  result.vertex_labels = new int[result.max_graph_vertex_count * gdb.size()];
  result.db_size = gdb.size();


  result.vertex_offsets[result.max_graph_vertex_count * gdb.size()] = result.edges_sizes;


  int edge_offset = 0;
  for(int i = 0; i < gdb.size(); i++) {
    int vertex_idx_offset = i * result.max_graph_vertex_count;

    for(int j = 0; j < gdb[i].vertex_size(); j++) {
      result.vertex_offsets[vertex_idx_offset + j] = edge_offset;
      result.vertex_labels[vertex_idx_offset + j] = gdb[i][j].label;

      for(int k = 0; k < gdb[i][j].edge.size(); k++) {
        if(edge_offset >= result.edges_sizes) {
          assert(edge_offset < result.edges_sizes);
        }
        result.edges[edge_offset] = gdb[i][j].edge[k].to + vertex_idx_offset;
        result.edges_labels[edge_offset] = gdb[i][j].edge[k].elabel;
        edge_offset++;
      } // for k
    } // for j


    for(int j = gdb[i].vertex_size(); j < result.max_graph_vertex_count; j++) {
      result.vertex_offsets[vertex_idx_offset + j] = -1;
      result.vertex_labels[vertex_idx_offset + j] = -1;
    } // for j

  } // for i

  result.located_on_host = true;

  return result;
} // create_from_host_representation



void graph_database_cuda::copy_to_device(graph_database_cuda *device_gdb)
{
  if(located_on_host == false || device_gdb->located_on_host == true) {
    throw std::runtime_error("cannot copy the data: source or destination stored in wrong memory.");
  }

  device_gdb->edges_sizes = edges_sizes;
  device_gdb->max_graph_vertex_count = max_graph_vertex_count;
  device_gdb->vertex_count = vertex_count;
  device_gdb->db_size = db_size;

  CUDAMALLOC(&device_gdb->edges, edges_sizes * sizeof(int), *logger);
  CUDA_EXEC(cudaMemcpy(device_gdb->edges, edges, edges_sizes * sizeof(int), cudaMemcpyHostToDevice), *logger);

  CUDAMALLOC(&device_gdb->edges_labels, edges_sizes * sizeof(int), *logger);
  CUDA_EXEC(cudaMemcpy(device_gdb->edges_labels, edges_labels, edges_sizes * sizeof(int), cudaMemcpyHostToDevice), *logger);

  CUDAMALLOC(&device_gdb->vertex_offsets, (db_size * max_graph_vertex_count + 1) * sizeof(int), *logger);
  CUDA_EXEC(cudaMemcpy(device_gdb->vertex_offsets, vertex_offsets, (db_size * max_graph_vertex_count + 1) * sizeof(int), cudaMemcpyHostToDevice), *logger);

  CUDAMALLOC(&device_gdb->vertex_labels, db_size * max_graph_vertex_count * sizeof(int), *logger);
  CUDA_EXEC(cudaMemcpy(device_gdb->vertex_labels, vertex_labels, db_size * max_graph_vertex_count * sizeof(int), cudaMemcpyHostToDevice), *logger);

} // copy_to_device


void graph_database_cuda::copy_from_device(graph_database_cuda *device_gdb)
{
  if(located_on_host == false || device_gdb->located_on_host == true) {
    throw std::runtime_error("cannot copy the data: source or destination stored in wrong memory.");
  }


  edges_sizes = device_gdb->edges_sizes;
  max_graph_vertex_count = device_gdb->max_graph_vertex_count;
  vertex_count = device_gdb->vertex_count;
  db_size = device_gdb->db_size;


  delete [] edges;
  edges = new int[edges_sizes];
  CUDA_EXEC(cudaMemcpy(edges, device_gdb->edges, edges_sizes * sizeof(int), cudaMemcpyDeviceToHost), *logger);

  delete [] edges_labels;
  edges_labels = new int[edges_sizes];
  CUDA_EXEC(cudaMemcpy(edges_labels, device_gdb->edges_labels, edges_sizes * sizeof(int), cudaMemcpyDeviceToHost), *logger);


  delete [] vertex_offsets;
  vertex_offsets = new int[db_size * max_graph_vertex_count + 1];
  CUDA_EXEC(cudaMemcpy(vertex_offsets, device_gdb->vertex_offsets, (db_size * max_graph_vertex_count + 1) * sizeof(int), cudaMemcpyDeviceToHost), *logger);

  delete [] vertex_labels;
  vertex_labels = new int[db_size * max_graph_vertex_count];
  CUDA_EXEC(cudaMemcpy(vertex_labels, device_gdb->vertex_labels, db_size * max_graph_vertex_count * sizeof(int), cudaMemcpyDeviceToHost), *logger);
}




graph_database_cuda graph_database_cuda::host_copy()
{
  if(located_on_host == false) {
    throw std::runtime_error("cannot copy the data: source or destination stored in wrong memory.");
  }

  graph_database_cuda result(true);

  result.edges_sizes = edges_sizes;
  result.max_graph_vertex_count = max_graph_vertex_count;
  result.vertex_count = vertex_count;
  result.db_size = db_size;


  result.edges = new int[edges_sizes];
  memcpy(result.edges, edges, edges_sizes * sizeof(int));


  result.edges_labels = new int[edges_sizes];
  memcpy(result.edges_labels, edges_labels, edges_sizes * sizeof(int));


  result.vertex_offsets = new int[db_size * max_graph_vertex_count + 1];
  memcpy(result.vertex_offsets, vertex_offsets, (db_size * max_graph_vertex_count + 1) * sizeof(int));

  result.vertex_labels = new int[db_size * max_graph_vertex_count];
  memcpy(result.vertex_labels, vertex_labels, db_size * max_graph_vertex_count * sizeof(int));
  //return result;
  throw std::runtime_error("graph_database_cuda::host_copy not implemented correctly (probably)");
}




void graph_database_cuda::delete_from_host()
{
  if(located_on_host == false) {
    throw std::runtime_error("Content not allocated in the host memory.");
  }

  delete [] edges;
  delete [] edges_labels;
  delete [] vertex_offsets;
  delete [] vertex_labels;

  edges = 0;
  edges_labels = 0;
  vertex_offsets = 0;
  vertex_labels = 0;

  edges_sizes = -1;
  max_graph_vertex_count = -1;
  vertex_count = -1;
  db_size = -1;
} // delete_from_host

void graph_database_cuda::delete_from_device()
{
  if(located_on_host == true) {
    throw std::runtime_error("Content not allocated in the device memory.");
  }

  CUDAFREE(edges, *logger);
  CUDAFREE(edges_labels, *logger);
  CUDAFREE(vertex_offsets, *logger);
  CUDAFREE(vertex_labels, *logger);

  edges = 0;
  edges_labels = 0;
  vertex_offsets = 0;
  vertex_labels = 0;

  edges_sizes = -1;
  max_graph_vertex_count = -1;
  vertex_count = -1;
  db_size = -1;
}

void graph_database_cuda::convert_to_host_representation(types::graph_database_t &gdb)
{
  if(located_on_host == false) {
    throw std::runtime_error("Content not allocated in host memory.");
  }


  for(int i = 0; i < db_size; i++) {
    types::Graph grph;
    int start_offset = i * max_graph_vertex_count;
    for(int j = 0; j < max_graph_vertex_count; j++) {
      if(!vertex_is_valid(start_offset + j)) {
        break;
      }

      int vertex_degree = get_vertex_degree(start_offset + j); //vertex_offsets[start_offset + j + 1] - vertex_offsets[start_offset + j];
      int neighbrhood_start = vertex_offsets[start_offset + j];
      types::Vertex newv;
      newv.label = vertex_labels[start_offset + j];
      for(int v = 0; v < vertex_degree; v++) {
        int edge_label = edges_labels[neighbrhood_start + v];
        int other_vertex_id = edges[neighbrhood_start + v];
        newv.push(j, other_vertex_id - start_offset, edge_label);
      } // for v
      grph.push_back(newv);
    } // for j

    grph.buildEdge();
    gdb.push_back(grph);
  } // for i
} // convert



std::string graph_database_cuda::to_string() const
{
  if(located_on_host == false) {
    CRITICAL_ERROR(*logger, "located_on_host: " << located_on_host);
  }
  std::stringstream ss;

  ss << "located_on_host: " << BOOL2STR(located_on_host) << "; db_size: " << db_size << endl;
  ss << "edges_sizes: " << edges_sizes << "; max_graph_vertex_count: " << max_graph_vertex_count << "; vertex_count: " << vertex_count << "; vertex size: " << (max_graph_vertex_count * db_size) << endl;

  ss << "edge labels: ";
  for(int i = 0; i < edges_sizes; i++) {
    ss << edges_labels[i] << " ";
  } // for i
  ss << std::endl;


  ss << "edges: ";
  for(int v = 0; v < max_graph_vertex_count * db_size; v++) {
    if(vertex_is_valid(v)) {
      int neigh_offset = vertex_offsets[v];
      int degree = get_vertex_degree(v);
      ss << v << ": ";
      for(int i = 0; i < degree; i++) {
        ss << edges[neigh_offset + i] << "(" << edges_labels[neigh_offset + i] << ")  ";
      }
      ss << " |";
    }
  }
  ss << endl;
  /*
     for(int i = 0; i < edges_sizes; i++) {
     ss << edges[i] << " ";
     } // for i
     ss << std::endl;
   */
  ss << "vertex_labels: ";
  for(int i = 0; i < max_graph_vertex_count * db_size; i++) {
    ss << i << ":" << vertex_labels[i] << " ";
  }
  ss << std::endl;


  return ss.str();
} // graph_database_cuda::to_string









std::string graph_database_cuda::to_string_no_content() const
{
  std::stringstream ss;

  ss << "located_on_host: " << BOOL2STR(located_on_host) << "; db_size: " << db_size << endl;
  ss << "edges_sizes: " << edges_sizes << "; max_graph_vertex_count: " << max_graph_vertex_count << "; vertex_count: " << vertex_count << "; vertex size: " << (max_graph_vertex_count * db_size) << endl;
  ss << "vertex_labels: " << vertex_labels << "; edges_labels: "  << edges_labels << "; vertex_offsets: " << vertex_offsets << endl;


  return ss.str();
} // graph_database_cuda::to_string







void graph_database_cuda::shrink_database(int minimal_support)
{
  //types::EdgeList edges;
  //Projected_map3 root;
  //edge_gid_list3_t root;
  //int max_vertex_count = db_size * max_graph_vertex_count;
  /*
     for(int vid = 0; vid < max_graph_vertex_count; vid++) {
     if(!vertex_is_valid(vid)) continue;
     int vid_degree = get_vertex_degree(vid);
     for(int eid = 0; eid < vid_degree; eid++) {
      root[g[from].label][(*it)->elabel][g[(*it)->to].label].insert(id);
     } // for eid_
     } // for vid
   */
  throw std::runtime_error("graph_database_cuda::shrink_database not implemented");
}






bool graph_database_cuda::test_database() const
{
  bool found_error = false;
  for(int g = 0; g < db_size; g++) {
    int vertex_offset = g * max_graph_vertex_count;
    for(int v = vertex_offset; get_vertex_label(v) != -1 && v < max_graph_vertex_count; v++) {
      int v_offset = vertex_offsets[v];
      for(int e = 0; e < get_vertex_degree(v); e++) {
        int v_to = edges[v_offset + e];

        int v_v_to_label = edges_labels[v_offset + e];
        int v_to_v_label = get_edge_label(v_to, v);
        int v_v_to_label2 = get_edge_label(v, v_to);
        if(v_v_to_label != v_to_v_label || v_v_to_label != v_v_to_label2) {
          CRITICAL_ERROR(*logger, "graph: " << g << "; error for cuda edge: (" << v << "," << v_to << "); translated to host: (" << get_host_graph_vertex_id(v)
                         << ", " << get_host_graph_vertex_id(v_to) << "); v_v_to_label: " << v_v_to_label
                         << "; v_to_v_label: " << v_to_v_label
                         << "; v_v_to_label: " << v_v_to_label);
          CRITICAL_ERROR(*logger, "edge (" << v << "," << v_to << ") label: " << v_v_to_label << "; (" << v_to << "," << v << ") label: " << v_to_v_label);
          found_error = true;
        } // if

      } // for e
    } // for v
  } // for g


  return found_error;
} // graph_database_cuda::test_database

} // namespace cuda


