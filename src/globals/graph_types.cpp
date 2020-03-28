#include <graph_types.hpp>
#include <algorithm>
#include <cassert>
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <logger.hpp>
#include <utils.hpp>
#include <cstring>

namespace types {

template <class T, class Iterator>
void tokenize(const char *str, Iterator iterator)
{
  std::istringstream is(std::string(str));
  std::copy(std::istream_iterator <T> (is), std::istream_iterator <T> (), iterator);
}



Graph::Graph() : edge_size_(0), directed(false)
{

}

Graph::Graph(bool _directed)
{
  directed = _directed;
}

void Graph::buildEdge()
{
  char buf[512];
  std::map <std::string, unsigned int> tmp;

  unsigned int id = 0;
  for(int from = 0; from < (int)size(); ++from) {
    for(Vertex::edge_iterator it = (*this)[from].edge.begin();
        it != (*this)[from].edge.end(); ++it) {
      if(directed || from <= it->to)
        std::sprintf(buf, "%d %d %d", from, it->to, it->elabel);
      else
        std::sprintf(buf, "%d %d %d", it->to, from, it->elabel);

      // Assign unique id's for the edges.
      if(tmp.find(buf) == tmp.end()) {
        it->id = id;
        tmp[buf] = id;
        ++id;
      } else {
        it->id = tmp[buf];
      }
    }
  }

  edge_size_ = id;
}





std::istream &Graph::read(std::istream &is)
{
  char line[1024];
  std::vector<std::string> result;

  clear();

  while(true) {
    unsigned int pos = is.tellg();
    if(!is.getline(line, 1024)) {
      break;
    }
    result.clear();
    utils::split(line, result);

    if(result.empty()) {
      // do nothing
    } else if(result[0] == "t") {
      if(!empty()) {   // use as delimiter
        is.seekg(pos, std::ios_base::beg);
        break;
      } else {
        // y = atoi (result[3].c_str());
      }
    } else if(result[0] == "v" && result.size() >= 3) {
      unsigned int id    = atoi(result[1].c_str());
      this->resize(id + 1);
      (*this)[id].label = atoi(result[2].c_str());
    } else if(result[0] == "e" && result.size() >= 4) {
      int from   = atoi(result[1].c_str());
      int to     = atoi(result[2].c_str());
      int elabel = atoi(result[3].c_str());

      if((int)size() <= from || (int)size() <= to) {
        std::cerr << "Format Error:  define vertex lists before edges, from: " << from << "; to: " << to << "; vertex count: " << size() << std::endl;
        throw std::runtime_error("Format Error:  define vertex lists before edges");
      }

      (*this)[from].push(from, to, elabel);
      if(directed == false)
        (*this)[to].push(to, from, elabel);
    }
  }

  buildEdge();

  return is;
}

std::istream &Graph::read_fsg(std::istream &is)
{
  char line[1024];
  std::vector<std::string> result;

  clear();

  std::map<std::string,int > vertex_labels;
  std::map<std::string,int > edge_labels;
  int num_vertex_labels = 0, num_edge_labels = 0;

  while(true) {
    unsigned int pos = is.tellg();
    if(!is.getline(line, 1024)) {
      break;
    }
    result.clear();
    utils::split(line, result);

    if(result.empty()) {
      // do nothing
    } else if(result[0] == "t") {
      if(!empty()) {   // use as delimiter
        is.seekg(pos, std::ios_base::beg);
        break;
      } else {
        // y = atoi (result[3].c_str());
      }
    } else if(result[0] == "v" && result.size() >= 3) {
      unsigned int id    = atoi(result[1].c_str());
      this->resize(id + 1);
       if(vertex_labels.count(result[2]) == 0 )
           vertex_labels[result[2]] = num_vertex_labels++;
       (*this)[id].label = vertex_labels[result[2]];
      //(*this)[id].label = atoi(result[2].c_str());
    } else if(result[0] == "u" && result.size() >= 4) {
      int from   = atoi(result[1].c_str());
      int to     = atoi(result[2].c_str());
      //int elabel = atoi(result[3].c_str());
       
       if(edge_labels.count(result[3]) == 0 )
          edge_labels[result[3]] = num_edge_labels++;
      int elabel = edge_labels[result[3]];

      if((int)size() <= from || (int)size() <= to) {
        std::cerr << "Format Error:  define vertex lists before edges, from: " << from << "; to: " << to << "; vertex count: " << size() << std::endl;
        throw std::runtime_error("Format Error:  define vertex lists before edges");
      }

      (*this)[from].push(from, to, elabel);
      if(directed == false)
        (*this)[to].push(to, from, elabel);
    }
  }

  buildEdge();

  return is;
}


std::ostream &Graph::write(std::ostream &os)
{
  char buf[512];
  std::set <std::string> tmp;

  for(int from = 0; from < (int)size(); ++from) {
    os << "v " << from << " " << (*this)[from].label << std::endl;

    for(Vertex::edge_iterator it = (*this)[from].edge.begin();
        it != (*this)[from].edge.end(); ++it) {
      if(directed || from <= it->to) {
        std::sprintf(buf, "%d %d %d", from, it->to,   it->elabel);
      } else {
        std::sprintf(buf, "%d %d %d", it->to,   from, it->elabel);
      }
      tmp.insert(buf);
    } // for it
  } // for from

  for(std::set<std::string>::iterator it = tmp.begin(); it != tmp.end(); ++it) {
    os << "e " << *it << std::endl;
  } // for it

  return os;
}

void Graph::check(void)
{
  // Check all indices
  for(int from = 0; from < (int)size(); ++from) {
    //mexPrintf ("check vertex %d, label %d\n", from, (*this)[from].label);

    for(Vertex::edge_iterator it = (*this)[from].edge.begin();
        it != (*this)[from].edge.end(); ++it) {
      //mexPrintf ("   check edge from %d to %d, label %d\n", it->from, it->to, it->elabel);
      assert(it->from >= 0 && it->from < size());
      assert(it->to >= 0 && it->to < size());
    }
  }
}


std::string Edge::to_string() const {
  std::stringstream ss;
  ss << "e(" << from << "," << to << "," << elabel << ")";
  return ss.str();
}

std::string PDFS::to_string() const {
  std::stringstream ss;
  ss << "[" << id << "," << edge->to_string() << "]";
  return ss.str();
}

std::string PDFS::to_string_projection(types::graph_database_t &gdb, types::graph_database_cuda &cgdb) const
{
  const PDFS *curr = this;
  std::string result;
  while(curr != 0) {
    std::stringstream ss;
    types::Graph &grph = gdb[curr->id];
    int cuda_grph_from = cgdb.translate_to_device(curr->id, curr->edge->from);
    int cuda_grph_to = cgdb.translate_to_device(curr->id, curr->edge->to);
    ss << "(" << grph.get_vertex_label(curr->edge->from) << ") " << curr->edge->from << "/" << cuda_grph_from << " = " << curr->edge->elabel << " = " << curr->edge->to << "/" << cuda_grph_to << " (" <<  grph.get_vertex_label(curr->edge->to) << ");  ";
    result = ss.str() + result; //curr->to_string();
    curr = curr->prev;
  }
  return result;
}

std::string Projected::to_string() const
{
  std::stringstream ss;

  for(int i = 0; i < size(); i++) {
    ss << (*this)[i].to_string() << "; ";
  } // for i

  return ss.str();
} // Projected::to_string


void History::build(const Graph &graph, PDFS *e)
{
  // first build history
  clear();
  edge.clear();
  edge.resize(graph.edge_size());
  vertex.clear();
  vertex.resize(graph.size());

  if(e) {
    push_back(e->edge);
    edge[e->edge->id] = vertex[e->edge->from] = vertex[e->edge->to] = 1;

    for(PDFS *p = e->prev; p; p = p->prev) {
      push_back(p->edge);       // this line eats 8% of overall instructions(!)
      edge[p->edge->id] = vertex[p->edge->from] = vertex[p->edge->to] = 1;
    }
    std::reverse(begin(), end());
  }
}

std::string History::to_string() const
{
  std::stringstream ss;

  //ostream_iterator<
  for(int i = 0; i < size(); i++) {
    ss << at(i)->to_string() << "; ";
  }
  return ss.str();
}

/* Original comment:
 * get_forward_pure ()
 *  e1 (from1, elabel1, to1)
 *  from から繋がる edge e2(from2, elabel2, to2) を返す.
 *
 *  minlabel <= elabel2,
 *  (elabel1 < elabel2 ||
 *  (elabel == elabel2 && tolabel1 < tolabel2) の条件をみたす.
 *  (elabel1, to1) のほうが先に探索されるべき
 *  また, いままで見た vertex には逝かない (backward のやくめ)
 *
 * RK comment:
 * ???? gets the edge that starts and extends the right-most path.
 *
 */
bool get_forward_rmpath(const Graph &graph, Edge *e, int minlabel, History& history, types::EdgeList &result)
{
  result.clear();
  assert(e->to >= 0 && e->to < graph.size());
  assert(e->from >= 0 && e->from < graph.size());
  int tolabel = graph[e->to].label;

  for(Vertex::const_edge_iterator it = graph[e->from].edge.begin();
      it != graph[e->from].edge.end(); ++it) {
    int tolabel2 = graph[it->to].label;
    if(e->to == it->to || minlabel > tolabel2 || history.hasVertex(it->to))
      continue;

    if(e->elabel < it->elabel || (e->elabel == it->elabel && tolabel <= tolabel2))
      result.push_back(const_cast<Edge*>(&(*it)));
  }

  return (!result.empty());
}

/* Original comment:
 * get_forward_pure ()
 *  e (from, elabel, to)
 *  to から繋がる edge を返す
 *  ただし, minlabel より大きいものにしかいかない (DFSの制約)
 *  また, いままで見た vertex には逝かない (backward のやくめ)
 *
 * RK comment: this function takes a "pure" forward edge, that is: an
 * edge that extends the last node of the right-most path, i.e., the
 * right-most node.
 *
 */
bool get_forward_pure(const Graph &graph, Edge *e, int minlabel, History& history, types::EdgeList &result)
{
  result.clear();

  assert(e->to >= 0 && e->to < graph.size());

  // Walk all edges leaving from vertex e->to.
  for(Vertex::const_edge_iterator it = graph[e->to].edge.begin();
      it != graph[e->to].edge.end(); ++it) {
    // -e-> [e->to] -it-> [it->to]
    assert(it->to >= 0 && it->to < graph.size());
    if(minlabel > graph[it->to].label || history.hasVertex(it->to))
      continue;

    result.push_back(const_cast<Edge*>(&(*it)));
  }

  return (!result.empty());
}

/*
 * Original comment:
 * graph の vertex からはえる edge を探す
 * ただし, fromlabel <= tolabel の性質を満たす.
 *
 * RK comment:
 *
 */
bool get_forward_root(const Graph &g, const Vertex &v, types::EdgeList &result)
{
  result.clear();
  for(Vertex::const_edge_iterator it = v.edge.begin(); it != v.edge.end(); ++it) {
    assert(it->to >= 0 && it->to < g.size());
    if(v.label <= g[it->to].label)
      result.push_back(const_cast<Edge*>(&(*it)));
  }

  return (!result.empty());
}

/* Original comment:
 *  get_backward (graph, e1, e2, history);
 *  e1 (from1, elabel1, to1)
 *  e2 (from2, elabel2, to2)
 *  to2 -> from1 に繋がるかどうかしらべる.
 *
 *  (elabel1 < elabel2 ||
 *  (elabel == elabel2 && tolabel1 < tolabel2) の条件をみたす. (elabel1, to1) のほうが先に探索されるべき
 *
 * RK comment: gets backward edge that starts and ends at the right most path
 * e1 is the forward edge and the backward edge goes to e1->from
 */
Edge *get_backward(const Graph &graph, Edge* e1, Edge* e2, History& history)
{
  if(e1 == e2)
    return 0;

  assert(e1->from >= 0 && e1->from < graph.size());
  assert(e1->to >= 0 && e1->to < graph.size());
  assert(e2->to >= 0 && e2->to < graph.size());

  for(Vertex::const_edge_iterator it = graph[e2->to].edge.begin();
      it != graph[e2->to].edge.end(); ++it) {
    if(history.hasEdge(it->id))
      continue;

    if((it->to == e1->from) &&
       ((e1->elabel < it->elabel) ||
        (e1->elabel == it->elabel) &&
        (graph[e1->to].label <= graph[e2->to].label)
       )) {
      return const_cast<Edge*>(&(*it));
    } // if(...)
  } // for(it)

  return 0;
}

std::string Graph::to_string() const
{
  std::stringstream ss;
  for(int i = 0; i < vertex_size(); i++) {
    const Vertex &v = at(i);
    for(int k = 0; k < v.edge.size(); k++) {
      ss << "from: " << v.edge[k].from << "; to: " << v.edge[k].to
      << "; (" << v.label << ", " << v.edge[k].elabel << ", " << get_vertex_label(v.edge[k].to) << ")" << std::endl;
      //<< get_vertex_label(v.edge[k].from) << ", " << get_vertex_label(v.edge[k].to)<< std::endl;
    } // for k
  } // for i
  return ss.str();

  //DFSCode dfs = get_min_dfs_code();
  //return dfs.to_string();
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Serialization
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





size_t Vertex::get_serialized_size(const Vertex &vrtx)
{
  //    vertex label + 4 * #of edges  * sizeof(int)    +   number of edges  + label;
  return sizeof(int) + 4 * vrtx.edge.size() * sizeof(int) + sizeof(int)     + sizeof(int);
}


size_t Vertex::get_serialized_size(char *buffer, size_t buffer_size)
{
  int s = *((int*)buffer);
  return s;
}


size_t Vertex::serialize(const Vertex &vrtx, char *buffer, size_t buffer_size)
{
  if(buffer_size < get_serialized_size(vrtx)) throw std::runtime_error("Buffer too small.");
  int pos = 0;

  // size of this serialized vertex in bytes.
  *((int*)(buffer + pos)) = get_serialized_size(vrtx);
  pos += sizeof(int);

  // store the vertex label
  *((int*)(buffer + pos)) = vrtx.label;
  pos += sizeof(int);


  // store number of edges
  *((int*)(buffer + pos)) = vrtx.edge.size();
  pos += sizeof(int);

  for(int i = 0; i < vrtx.edge.size(); i++) {
    *((int*)(buffer + pos)) = vrtx.edge[i].from;
    pos += sizeof(int);

    *((int*)(buffer + pos)) = vrtx.edge[i].to;
    pos += sizeof(int);

    *((int*)(buffer + pos)) = vrtx.edge[i].elabel;
    pos += sizeof(int);

    *((int*)(buffer + pos)) = vrtx.edge[i].id;
    pos += sizeof(int);
  } // for i

  return pos;
} // Vertex::serialize


size_t Vertex::deserialize(Vertex &vrtx, char *buffer, size_t buffer_size)
{
  // TODO: check minimum buffer size
  if(buffer_size < get_serialized_size(buffer, buffer_size)) throw std::runtime_error("Buffer too small.");
  int pos = 0;
  vrtx.edge.clear();

  // read buffer s
  pos += sizeof(int);


  // read the vertex label
  vrtx.label = *((int*)(buffer + pos));
  pos += sizeof(int);


  // read the number of edges
  int edge_count = *((int*)(buffer + pos));
  pos += sizeof(int);


  for(int i = 0; i < edge_count; i++) {
    Edge tmp_edge;
    tmp_edge.from = *((int*)(buffer + pos));
    pos += sizeof(int);
    tmp_edge.to = *((int*)(buffer + pos));
    pos += sizeof(int);
    tmp_edge.elabel = *((int*)(buffer + pos));
    pos += sizeof(int);
    tmp_edge.id = *((int*)(buffer + pos));
    pos += sizeof(int);
    vrtx.edge.push_back(tmp_edge);
  } // for i

  return pos;
} // Vertex::deserialize




size_t Graph::get_serialized_size(const Graph &grph)
{
  size_t s = sizeof(int) + sizeof(int) + sizeof(int) + sizeof(bool); // edge_size_ + total buffer size + number of vertices + variable directed(bool)
  for(int i = 0; i < grph.size(); i++) {
    s += Vertex::get_serialized_size(grph[i]);
  } // for i
  return s;
} // Graph::get_serialized_size


size_t Graph::get_serialized_size(char *buffer, size_t buffer_size)
{
  return *((int*) buffer);
}


size_t Graph::serialize(const Graph &grph, char *buffer, size_t buffer_size)
{
  if(get_serialized_size(grph) > buffer_size) throw std::runtime_error("Buffer too small.");
  int pos = 0;

  // store buffer size
  *((int*)(buffer + pos)) = get_serialized_size(grph);
  pos += sizeof(int);

  // store edge_size_
  *((int*)(buffer + pos)) = grph.edge_size_;
  pos += sizeof(int);

  // store number of vertices
  *((bool*)(buffer + pos)) = grph.directed;
  pos += sizeof(grph.directed);


  // store number of vertices
  *((int*)(buffer + pos)) = grph.size();
  pos += sizeof(int);

  for(int i = 0; i < grph.size(); i++) {
    int tmp_pos = Vertex::serialize(grph.at(i), buffer + pos, buffer_size - pos);
    pos += tmp_pos;
  } // for i

  return pos;
} // Graph::serialize


size_t Graph::deserialize(Graph &grph, char *buffer, size_t buffer_size)
{
  if(Graph::get_serialized_size(buffer, buffer_size) > buffer_size) throw std::runtime_error("Buffer too small.");

  grph.clear();
  int pos = 0;

  // store buffer size
  pos += sizeof(int);

  // store edge_size_
  grph.edge_size_ = *((int*)(buffer + pos));
  pos += sizeof(int);

  // store number of vertices
  grph.directed = *((bool*)(buffer + pos));
  pos += sizeof(grph.directed);


  // store number of vertices
  int vert_count = *((int*)(buffer + pos));
  pos += sizeof(int);

  for(int i = 0; i < vert_count; i++) {
    Vertex tmp_vert;
    int tmp_pos = Vertex::deserialize(tmp_vert, buffer + pos, buffer_size - pos);
    grph.push_back(tmp_vert);
    pos += tmp_pos;
  } // for i

  return pos;
} // Graph::deserialize





size_t Graph::get_serialized_size(const graph_database_t &grph_db)
{
  size_t min_buff_size = 0;

  min_buff_size += sizeof(int) + sizeof(int); // size of the database + size of the buffer

  for(size_t i = 0; i < grph_db.size(); i++) {
    min_buff_size += get_serialized_size(grph_db[i]);
  } // for i

  return min_buff_size;
} // Graph::get_serialized_size


size_t Graph::get_serialized_size_db(char *buffer, size_t buffer_size)
{
  //abort();
  return *((int*) buffer);
} // Graph::get_serialized_size


size_t Graph::serialize(const graph_database_t &grph_db, char *buffer, size_t buffer_size)
{
  size_t pos = 0;

  int min_buff_size = get_serialized_size(grph_db);
  if(min_buff_size > buffer_size) throw std::runtime_error("Buffer too small.");

  *((int*)(buffer + pos)) = min_buff_size;
  pos += sizeof(int);

  *((int*)(buffer + pos)) = grph_db.size();
  pos += sizeof(int);

  for(int i = 0; i < grph_db.size(); i++) {
    size_t tmp_pos = serialize(grph_db[i], buffer + pos, buffer_size - pos);
    pos += tmp_pos;
  }

  return pos;
} // Graph::serialize


size_t Graph::deserialize(graph_database_t &grph_db, char *buffer, size_t buffer_size)
{
  int min_buf_size = get_serialized_size_db(buffer, buffer_size);

  if(buffer_size < min_buf_size) throw std::runtime_error("Buffer too small.");

  grph_db.clear();

  size_t pos = 0;
  // skip buffer size
  pos += sizeof(int);

  int grph_db_size = *((int*)(buffer + pos));
  pos += sizeof(int);

  for(int i = 0; i < grph_db_size; i++) {
    Graph grph;
    size_t tmp_pos = deserialize(grph, buffer + pos, buffer_size - pos);
    pos += tmp_pos;
    grph_db.push_back(grph);
  } // for i


  return pos;
} // Graph::deserialize


} // namespace types


