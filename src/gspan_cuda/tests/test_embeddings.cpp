#include <cuda_graph_types.hpp>
#include <gtest/gtest.h>
#include <logger.hpp>
#include <test_support.hpp>
#include <embedding_lists.hpp>
#include <cuda_computation_parameters.hpp>

#include <cuda_gspan_ops.hpp>
#include <cuda_tools.hpp>

#include <thrust/sort.h>

using namespace types;
using std::string;

using gspan_cuda::create_first_embeddings;
using gspan_cuda::get_all_extensions;
using gspan_cuda::extension_element_t;

static Logger *logger = Logger::get_logger("CUDA_FE_TEST");

std::vector<extension_element_t>
get_graph_extensions(extension_element_t *exts, int exts_length, int grph_id,
                     types::graph_database_cuda &cuda_gdb, types::embedding_list_columns h_embeddings, int row)
{
  std::vector<extension_element_t> result;
  for(int i = 0; i < exts_length; i++) {
    if(exts[i].row == row) {
      result.push_back(exts[i]);
    } // if
  } // for i
  return result;
} // get_graph_extensions






void store_valid_forward_extension(std::set<int> &host_embedding_extensions,
                                   types::Graph &h_grph,
                                   types::graph_database_cuda &cuda_gdb,
                                   int h_grph_from,
                                   int d_grph_from,
                                   std::set<int> &embedding_vids,
                                   std::vector<extension_element_t> &curr_embedding_extensions)
{
  //cout << "store_valid_forward_extension, h_grph_from: " << h_grph_from << endl;
  for(int i = 0; i < h_grph[h_grph_from].edge.size(); i++) {
    // 1) check whether the edge h_grph[h_grph_from].edge[i] is valid extension
    int h_grph_to = h_grph[h_grph_from].edge[i].to;
    if(embedding_vids.find(h_grph_to) != embedding_vids.end()) {
      continue;
    }

    // 2) then check whether it exists in the cuda extension representation
    for(int j = 0; j < curr_embedding_extensions.size(); j++) {
      if(curr_embedding_extensions[j].is_forward() == false) continue;
      int cuda_ext_grph_from = cuda_gdb.get_host_graph_vertex_id(curr_embedding_extensions[j].from_grph);
      int cuda_ext_grph_to = cuda_gdb.get_host_graph_vertex_id(curr_embedding_extensions[j].to_grph);
      if(h_grph_from == cuda_ext_grph_from && h_grph_to == cuda_ext_grph_to) {
        //assert(host_embedding_extensions.find(j) == host_embedding_extensions.end());
        //std::cout << "found embedding(forward), " << j << ": " << curr_embedding_extensions[j].to_string() << std::endl;
        host_embedding_extensions.insert(j);
        break;
      } // if
    } // for j
  } // for i
} // store_valid_forward_extension






void store_valid_backward_extension(std::set<int> &host_embedding_extensions,
                                    types::Graph &h_grph,
                                    types::graph_database_cuda &cuda_gdb,
                                    int h_grph_from,
                                    int d_grph_from,
                                    std::set<int> &rmpath_embedding_vids,
                                    std::vector<extension_element_t> &curr_embedding_extensions)
{
  //cout << "store_valid_backward_extension, h_grph_from: " << h_grph_from << endl;
  for(int i = 0; i < h_grph[h_grph_from].edge.size(); i++) {
    // 1) check whether the edge h_grph[h_rm_vertex_id].edge[i] is valid
    // extension, i.e., whether the 'to' is on the right-most path
    // and the to vertex does not point into the embedding
    int h_grph_to = h_grph[h_grph_from].edge[i].to;
    if(rmpath_embedding_vids.find(h_grph_to) == rmpath_embedding_vids.end()) {
      continue;
    }

    // 2) then check whether it exists in the cuda extension representation
    for(int j = 0; j < curr_embedding_extensions.size(); j++) {
      if(curr_embedding_extensions[j].is_backward() == false) continue;
      int cuda_ext_grph_from = cuda_gdb.get_host_graph_vertex_id(curr_embedding_extensions[j].from_grph);
      int cuda_ext_grph_to = cuda_gdb.get_host_graph_vertex_id(curr_embedding_extensions[j].to_grph);
      if(h_grph_from == cuda_ext_grph_from && h_grph_to == cuda_ext_grph_to) {
        //std::cout << "found embedding(backward), " << j << ": " << curr_embedding_extensions[j].to_string() << std::endl;
        //if(host_embedding_extensions.find(j) != host_embedding_extensions.end()) {
        //for(std::set<int>::iterator it = host_embedding_extensions.begin(); it != host_embedding_extensions.end(); it++) {
        //std::cout << *it << "; ";
        //} // for it
        //std::cout << std::endl;
        //}
        //assert(host_embedding_extensions.find(j) == host_embedding_extensions.end());
        host_embedding_extensions.insert(j);
        break;
      } // if
    } // for j
  } // for i
} // store_valid_forward_extension






// row is the embedding index in the last column
//
//
//
bool check_one_embedding(types::graph_database_t &gdb,
                         types::graph_database_cuda &cuda_gdb,
                         types::RMPath rmpath,
                         types::embedding_list_columns h_embeddings,
                         int row,
                         extension_element_t *extensions,
                         int extensions_length)
{
  std::stringstream ss;
  for(int i = 0; i < rmpath.size(); i++) ss << rmpath[i] << ",";
  //DEBUG(*logger, "check_one_embedding, row: " << row << "; rmpath: " << ss.str());

  std::set<int> embedding_vids;
  std::set<int> rmpath_embedding_vids;
  int last_col_id = h_embeddings.columns_count;
  int h_graph_id = cuda_gdb.get_graph_id(h_embeddings.columns[last_col_id - 1][row].vertex_id);

  std::vector<extension_element_t> device_embedding_extensions =
    get_graph_extensions(extensions, extensions_length, h_graph_id, cuda_gdb, h_embeddings, row);
  //get_graph_extensions(extension_element_t *exts, int exts_length, int grph_id,
  //types::graph_database_cuda &cuda_gdb, types::embedding_list_columns h_embeddings, int row)

  /* store the indexes into device_embedding_extensions of valid(checked)
   * extensions then compare it with device_embedding_extensions: if
   * host_embedding_extensions.size() == device_embedding_extensions.size()
   * everything is ok
   */
  std::set<int> host_embedding_extensions;

  types::Graph &h_grph = gdb[h_graph_id];

  int curr_row = row;
  // store all embedding vertex ids into embedding_vids
  for(int i = h_embeddings.columns_count - 1; i > 0; i--) {
    int vid = h_embeddings.columns[i][curr_row].vertex_id;
    embedding_vids.insert(vid);
    curr_row = h_embeddings.columns[i][curr_row].back_link;
  } // for i

  curr_row = row;
  int rmpath_idx = rmpath.size() - 1;
  // store all embedding vertex ids into embedding_vids
  for(int i = h_embeddings.columns_count - 1; i >= 0; i--) {
    if(rmpath[rmpath_idx] == i) {
      int vid = h_embeddings.columns[i][curr_row].vertex_id;
      vid = cuda_gdb.get_host_graph_vertex_id(vid);
      rmpath_embedding_vids.insert(vid);
      rmpath_idx--;
    } // if

    curr_row = h_embeddings.columns[i][curr_row].back_link;
  } // for i


  // check all forward edges from the right-most path
  // for each extension from the right-most vertex check whether there is equivalent extension in device_embedding_extensions.
  curr_row = row;
  for(std::set<int>::iterator it = rmpath_embedding_vids.begin(); it != rmpath_embedding_vids.end(); it++) {
    int d_rm_vertex_id = cuda_gdb.get_device_graph_vertex_id(*it, h_graph_id);
    int h_rm_vertex_id = *it;
    store_valid_forward_extension(host_embedding_extensions, h_grph, cuda_gdb, h_rm_vertex_id, d_rm_vertex_id, embedding_vids, device_embedding_extensions);
  } // for i


  // check backward edges going from the right-most vertex
  int d_rm_vertex_id = h_embeddings.columns[last_col_id - 1][row].vertex_id;
  int h_rm_vertex_id = cuda_gdb.get_host_graph_vertex_id(d_rm_vertex_id);
  for(std::set<int>::iterator it = rmpath_embedding_vids.begin(); it != rmpath_embedding_vids.end(); it++) {
    if(h_rm_vertex_id == *it) continue;
    store_valid_backward_extension(host_embedding_extensions, h_grph, cuda_gdb, h_rm_vertex_id, d_rm_vertex_id, rmpath_embedding_vids, device_embedding_extensions);
  }

  // finally, check whether the host_embedding_extensions contains all indexes 0..device_embedding_extensions.size()-1
  if(device_embedding_extensions.size() != host_embedding_extensions.size()) {
    INFO(*logger, "returning false at this point: device_embedding_extensions.size(): " << device_embedding_extensions.size()
         << "; host_embedding_extensions.size(): " << host_embedding_extensions.size());

    std::stringstream ss;
    for(std::set<int>::iterator it = host_embedding_extensions.begin(); it != host_embedding_extensions.end(); it++) {
      ss << *it << "; ";
    }
    DEBUG(*logger, "host_embedding_extensions: " << ss.str());

    ss.str("");
    for(std::set<int>::iterator it = embedding_vids.begin(); it != embedding_vids.end(); it++) {
      ss << *it << ", ";
    } // for it
    DEBUG(*logger, "embedding vertex ids: " << ss.str());

    ss.str("");
    for(std::set<int>::iterator it = rmpath_embedding_vids.begin(); it != rmpath_embedding_vids.end(); it++) {
      ss << *it << ", ";
    } // for it
    DEBUG(*logger, "rmpath vertex ids: " << ss.str());
    DEBUG(*logger, "h_graph_id: " << h_graph_id);

    ss.str("");
    for(int i = 0; i < device_embedding_extensions.size(); i++) {
      ss << device_embedding_extensions[i].to_string() << "; " << endl;
    }
    DEBUG(*logger, "device_embedding_extensions: " << ss.str());
    DEBUG(*logger, "embedding: " << h_embeddings.embedding_to_string(device_embedding_extensions.front().row));
    DEBUG(*logger, "graph: " << cuda_gdb.graph_to_string(h_graph_id));

    return false;
  }
  for(int i = 0; i < device_embedding_extensions.size(); i++) {
    if(host_embedding_extensions.find(i) == host_embedding_extensions.end()) {
      INFO(*logger, "returning false at this point");
      return false;
    } // if
  } // for i

  return true;
} // check_one_embedding


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





static int get_max_vertex_count(const types::graph_database_t &gdb)
{
  int max = 0;
  for(int i = 0; i < gdb.size(); i++) {
    if(max < gdb[i].vertex_size()) max = gdb[i].vertex_size();
  } // for i
  return max;
} // get_max_vertex_count


static bool edge_exist(const types::embedding_list_columns &h_embed, int from, int to)
{
  //DEBUG(*logger, "# of embeddings: " << h_embed.columns_lengths[0]);
  for(int i = 0; i < h_embed.columns_lengths[0]; i++) {
    if(h_embed.columns[0][i].vertex_id == from &&
       h_embed.columns[1][i].vertex_id == to)
    {
      return true;
    }

  } // for i

  return false;
}

/**
 * tests the result of create_first_embeddings in h_embed. The test is
 * one-way: it checks all edges in gdb and if the edge matches the dfs
 * it checks whether it exists in the embeddings.
 *
 * TODO: do the opposite ! look into h_embed for each one-edge
 * embedding and test that the embedded edge actually exists in gdb.
 */
void check_host_db_agains_embedding(const types::graph_database_t &gdb, types::DFS dfs, const types::embedding_list_columns &h_embed)
{
  int max_vertex_count = get_max_vertex_count(gdb);

  TRACE5(*logger, h_embed.to_string());

  for(int i = 0; i < gdb.size(); i++) {

    for(int j = 0; j < gdb[i].vertex_size(); j++) {
      const types::Vertex &fromv = gdb[i][j];

      for(int k = 0; k < fromv.edge.size(); k++) {
        const types::Edge &e = fromv.edge[k];
        if(fromv.label == dfs.fromlabel && e.elabel == dfs.elabel && gdb[i].get_vertex_label(e.to) == dfs.tolabel) {
          //ASSERT_TRUE(edge_exist(h_embed, j+max_vertex_count*i, e.to+max_vertex_count*i));
          if(edge_exist(h_embed, j + max_vertex_count * i, e.to + max_vertex_count * i) == false) {
            CRITICAL_ERROR(*logger, "dfs labels: " << dfs.fromlabel << ", " << dfs.elabel << ", " << dfs.tolabel);
            CRITICAL_ERROR(*logger, "labels: " << fromv.label << ", " << e.elabel << ", " << gdb[i].get_vertex_label(e.to));
            CRITICAL_ERROR(*logger, "graph: " << i << "; vertex from: " << j << "; vertex to: " << e.to);
            CRITICAL_ERROR(*logger, "cuda graph: " << i << "; vertex from: " << (j + max_vertex_count * i) << "; vertex to: " << (e.to + max_vertex_count * i));
            throw std::runtime_error("test failed");
          }
        } // if
      } // for k

    } // for j
  } // for i
} // check_result


void check_embedding_against_host_db(const types::graph_database_t &gdb, types::DFS dfs, const types::embedding_list_columns &h_embed)
{
  int max_vertex_count = get_max_vertex_count(gdb);

  for(int i = 0; i < h_embed.columns_lengths[0]; i++) {
    int from = h_embed.columns[0][i].vertex_id;
    int to = h_embed.columns[1][i].vertex_id;
    int gid = from / max_vertex_count;
    //ASSERT_EQ(gdb[gid].get_vertex_label(from % max_vertex_count), dfs.fromlabel);
    if(gdb[gid].get_vertex_label(from % max_vertex_count) != dfs.fromlabel) {
      CRITICAL_ERROR(*logger, "i: " << i);
      CRITICAL_ERROR(*logger, "graph: " << gid << "; vertex from: " << from << "; vertex to: " << to
                     << "; dfs.fromlabel: " << dfs.fromlabel << "; host graph label: " << gdb[gid].get_vertex_label(from % max_vertex_count));
      throw std::runtime_error("test failed");
    }
    //ASSERT_EQ(gdb[gid].get_vertex_label(to % max_vertex_count), dfs.tolabel);
    if(gdb[gid].get_vertex_label(to % max_vertex_count) != dfs.tolabel) {
      CRITICAL_ERROR(*logger, "graph: " << gid << "; vertex from: " << from << "; vertex to: " << to
                     << "; dfs.tolabel: " << dfs.tolabel << "; host graph label: " << gdb[gid].get_vertex_label(to % max_vertex_count));
      throw std::runtime_error("test failed");
    }
    for(int j = 0; j < gdb[gid][from % max_vertex_count].edge.size(); j++) {
      if(gdb[gid][from % max_vertex_count].edge[j].to == to % max_vertex_count) {
        //ASSERT_EQ(gdb[gid][from % max_vertex_count].edge[j].elabel, dfs.elabel);
        if(gdb[gid][from % max_vertex_count].edge[j].elabel != dfs.elabel)  {
          CRITICAL_ERROR(*logger, "graph: " << gid << "; vertex from: " << from << "; vertex to: " << to);
          throw std::runtime_error("test failed");
        }
      } // if
    } // for i
  } // for i
} // check_result2




