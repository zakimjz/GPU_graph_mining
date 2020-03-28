//! \file graph_can_code.h - implementation of graph canonical code.
#ifndef _GRAPH_CAN_CODE_H
#define _GRAPH_CAN_CODE_H

using namespace std;

#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <hash_map>
#include "helper_funs.h"

namespace hasan_sampler {

template<typename V_T, typename E_T>
struct five_tuple;

template<typename V_T, typename E_T>
ostream& operator<< (ostream&, const five_tuple<V_T, E_T>&);


/**
 * \brief Storing a five_tuple that represent a labeled edge of a graph.
 *
 * <dfs_id1, dfs_id2, vertex_label1, edge_label, vertex_label2> are the 5-tuple.
 * It is used as part of the canonical code of a graph.
 */
template<typename V_T, typename E_T>
struct five_tuple {
  five_tuple() {}

  five_tuple(const int& id1, const int& id2, const V_T& li, const E_T& lij, const V_T& lj): _i(id1), _j(id2), _li(li), _lj(lj), _lij(lij) {}

  bool operator== (const five_tuple<V_T, E_T>& rhs) const {

    if((_i == rhs._i) && (_j == rhs._j) && (_li == rhs._li) &&
        (_lj == rhs._lj) && (_lij == rhs._lij))
      return true;

    // When the dest id is negative and all other things are the same
    // even then the tuples are the same.
    if((_i == rhs._i) && (_j < 0) && (rhs._j < 0) && (_li == rhs._li) &&
        (_lj == rhs._lj) && (_lij == rhs._lij))
      return true;


    return false;
  }

  bool operator< (const five_tuple<V_T, E_T>& rhs) const {

    bool is_fwd=(_i<_j);
    bool rhs_is_fwd=(rhs._i<rhs._j);

    if(!is_fwd && rhs_is_fwd) {                                      // back-edge < forward-edge
      return true;
    }

    if(!is_fwd && !rhs_is_fwd && _j<rhs._j) {                        // if both back edge, and _j < rhs._j
      return true;
    }

    if(!is_fwd && !rhs_is_fwd && _j==rhs._j && _lij<rhs._lij) {      // if both back edge, _j==rhs._j, and _lij < rhs._lij
      return true;
    }

    // Added by VC...
    if(!is_fwd && !rhs_is_fwd && _j==rhs._j && _lij==rhs._lij && _i<rhs._i) { // if both back edge, _j==rhs._j, and _lij == rhs._lij
      return true;
    }

    if(is_fwd && rhs_is_fwd && _i>rhs._i) {                          // if both forward edge, _i > rhs._i
      return true;
    }

    if(is_fwd && rhs_is_fwd && _i==rhs._i && _li<rhs._li) {         // if both forward edge, _i == rhs._i and _li < rhs._li
      return true;
    }

    if(is_fwd && rhs_is_fwd && _i==rhs._i &&
        _li==rhs._li && _lij<rhs._lij) {            // if both forward, _i == rhs._i, _li == rhs._li, then _lij < rhs._lij
      return true;
    }

    if(is_fwd && rhs_is_fwd && _i==rhs._i && _li==rhs._li &&
        _lij==rhs._lij && _lj<rhs._lj) {   // if both forward, _i == rhs._i, _lij == rhs._lij, _lj<rhs._lj
      return true;
    }

    // If both forward, everything same other than the destination
    if(is_fwd && rhs_is_fwd && _i==rhs._i && _li==rhs._li &&
        _lij==rhs._lij && _lj==rhs._lj && _j<rhs._j) {
      return true;
    }

    return false;
  }//end operator<

  friend ostream& operator<< <>(ostream&, const five_tuple<V_T, E_T>&);

  int _i;
  int _j;
  V_T _li;
  V_T _lj;
  E_T _lij;

};//end struct five_tuple


template<typename V_T, typename E_T>
ostream& operator<< (ostream& ostr, const five_tuple<V_T, E_T>& tuple)
{
  ostr<<tuple._i<<" "<<tuple._j<<" "<<tuple._li<<" "<<tuple._lij<<" "<<tuple._lj;
  return ostr;
}

/**
 * \struct less_than for edge sets, represented as a 5-tuple.
 * Used to order a list of edges. This function looks only
 * at the vertex labels and the edge label.
 * This function is not used for ordering the edges in canonical
 * order.
 */
template<typename V_T, typename E_T>
struct lt_five_tuple {

  /**
   * Returns true if t1 < t2
   */
  bool operator()(const five_tuple<V_T, E_T > t1, const five_tuple<V_T, E_T > t2) const {

    if(t1._li < t2._li) {
      return true;
    } else if(t1._li == t2._li) {  // Source edge is the same.

      if(t1._j < 0 && t2._j >= 0)  // t1 is fwd, t2 is back.
        return false;

      if(t1._j >= 0 && t2._j < 0)  // t1 is back, t2 is fwd.
        return true;

      if(t1._j >= 0 && t2._j >= 0) {  // Both back edges.
        if(t1._j > t2._j)  // Back edges to lower numbered edges come first.
          return true;
        else
          return false;
      }

      // Reach here only if both forward edges.
      if(t1._lij < t2._lij)   // edge label of t1 < t2 edge label.
        return true;
      else if(t1._lij > t2._lij)
        return false;

      // Both edge labels are same, then the last criterion
      if(t1._lj < t2._lj)
        return true;
    }

    return false;
  }
};


/**
 * \struct less_than for candidate edge sets, represented as a 5-tuple.
 * Used to order a list of edges by the canonical ordering.
 * The first one will be used to extend the current pattern.
 *
 * This code assumes that the first node in both the edges is
 * the same.
 */
template<typename V_T, typename E_T>
struct lt_five_tuple_can_order {


  //! \brief Returns true if t1 < t2
  bool operator()(const five_tuple<V_T, E_T> t1, const five_tuple<V_T, E_T> t2) const {

    bool is_t1_back = true, is_t2_back = true;

    if(t1._j == -1)     // Is t1 back_edge?
      is_t1_back = false;

    if(t2._j == -1)     // Is t2 back_edge?
      is_t2_back = false;

    if((is_t1_back && is_t2_back) || (!is_t1_back && !is_t2_back)) { // Both back edges or both fwd.

      if(t1._j < t2._j)
        return true;
      else
        return false;

    } else if(is_t1_back && !is_t2_back) { // t1 back, t2 forward.
      return true;
    } else if(!is_t1_back && is_t2_back) { // t2 back, t1 forward.
      return false;
    }
  }
};


template <typename V_T, typename E_T>
class canonical_code;

template<typename V_T, typename E_T>
ostream& operator<< (ostream&, const canonical_code<V_T, E_T>&);


/**
 * \brief Graph canonical Code class by partial specialization of
 * generic canonical_code class.
 *
 * pattern_prop is set to undirected (graph property)
 */
template<typename V_T, typename E_T>
class canonical_code {
public:

  typedef int STORAGE_TYPE;
  typedef five_tuple<V_T, E_T> FIVE_TUPLE;
  typedef FIVE_TUPLE INIT_TYPE;
  typedef eqint COMPARISON_FUNC;

  typedef vector<FIVE_TUPLE> TUPLES;
  typedef typename TUPLES::const_iterator CONST_IT;
  typedef typename TUPLES::iterator IT;
  typedef canonical_code<V_T, E_T> CAN_CODE;  //!< this class type
  typedef HASHNS::hash_map<int, int, HASHNS::hash<int>, std::equal_to<int> > VID_HMAP; //!< hash an int-->int
  typedef typename VID_HMAP::const_iterator VM_CONST_IT;
  typedef vector<int> RMP_T;

  //! Constructor
  canonical_code() : _can_code(id_generator++) {

  } // defunct default constructor

  /** Parameterized constructor that inserts ft as first tuple into
      DFS code, it also takes two vertex-id and store them in hashmap */
  canonical_code(const FIVE_TUPLE& ft, const int&gi, const int& gj) {
    append(ft, gi, gj);
  }
  //! Destructor
  ~canonical_code() {
  }
  //!<dfs code is just a vector of five_tuple, this begin() returns the five-tuple of 1st edge
  IT begin() {
    return _dfs_code.begin();
  }
  CONST_IT begin() const {
    return _dfs_code.begin();
  }
  IT end() {
    return _dfs_code.end();
  }
  CONST_IT end() const {
    return _dfs_code.end();
  }

  /*! \fn bool is_present(const FIVE_TUPLE& ft)
   *  \brief A member function to check whether the canonical code is already been derived.
   *	\param ft a constant reference of FIVE_TUPLE.
   * \return boolean
   */
  bool is_present(const FIVE_TUPLE& ft) {

    FIVE_TUPLE other(ft._j, ft._i, ft._lj, ft._lij, ft._li);
    if((_dfs_code.find(ft) == _dfs_code.end()) && (_dfs_code.find(other) == _dfs_code.end()))
      return false;
    else
      return true;
  }

  //! how many edges are there in the code?
  int size() const {
    return _dfs_code.size();
  }

  void clear() {
    _dfs_code.clear();
    _cid_to_gid.clear();
    _gid_to_cid.clear();
    _rmp.clear();
  }

  //! Overload operator []
  const FIVE_TUPLE& operator[](const int& index) const {
    return _dfs_code[index];
  }

  /*! initializing rmp, rmp is a vector of integer, it always inilializes as (0,1)
  * since, in our graph dataset, any graph's vertex id are integer and id starts
   with 0.
  */
  /*  void init_rmp() {
      if(!_rmp.empty())
        _rmp.clear();
      _rmp.push_back(0);
      _rmp.push_back(1);
    }
  */
  /*
    // when a forwarde edge is added to a pattern, its rightmost path may changes;
    // this routine makes the corresponding updates. It is used when we generate
    // a new candidate by adding an edge to a pattern.
    // The parameter passed is the five-tuple corresponding to the new edge
    // THIS ROUTINE IS CALLED IN update_rmpath() in graph_iso_check.h
    void update_rmp(const FIVE_TUPLE& tuple) {
      // if the right most path is empty, it is always
      // a forward edge and added by putting the two
      // id's of the graph
      if(_rmp.empty()) {
        _rmp.push_back(tuple._i);
        _rmp.push_back(tuple._j);
        return;
      }

      // no changes to rmp if it's a back-edge
      if(tuple._i>tuple._j)
        return;

      // Here is an example how rmp can change:
      // consider a graph's rmp is like, 1---4-----3-----2
      // at this point, an edge (4---5) is added with the vertex 4
      // like below:
      //     ---------5
      //     |
      // 1---4----3------2
      // new rightmost path is: 1---4-----5

      typename RMP_T::iterator rmp_it=_rmp.end()-1;
      while(rmp_it>=_rmp.begin()) {
        if(*rmp_it==tuple._i)  // finding whith vertex the forward edge connect's to
          break;
        rmp_it=_rmp.erase(rmp_it); // deleting the vertices that is not part of rmp
        rmp_it--;                  // checking the previous vertex
      }
      _rmp.push_back(tuple._j);   // adding the new edge's other vertex in the rmp

    }//end update_rmp()*/

  template<class PAT>
  void init(const INIT_TYPE& tuple, PAT* pattern) {
    clear();
    _dfs_code.push_back(tuple);

    ostringstream t_ss;
    t_ss << tuple;
    string t_str = t_ss.str();
    HASHNS::hash_map<string, int, hash_func<string>, equal_to<string> >::iterator itr = level_one_hash.find(t_str);
    if(itr != level_one_hash.end()) {
      _can_code = itr->second;
    } else {
      level_one_hash.insert(make_pair(t_str, _can_code));
    }

  }

  void push_back(const FIVE_TUPLE& tuple) {
    _dfs_code.push_back(tuple);
  }

  // append a dfs code, just by inserting this tuple at the end
  void append(const FIVE_TUPLE& tuple) {
    push_back(tuple);
  }

  /*! \fn void append(const FIVE_TUPLE& tuple, const int& gi, const int& gj)
   *  \brief A member function to append a dfs code and create mapping between code id and graph id.
   *	\param ft a constant reference of FIVE_TUPLE.
   *  \param gi,gj a constant reference of integer
   */
  void append(const FIVE_TUPLE& tuple, const int& gi, const int& gj) {
    push_back(tuple);
    _cid_to_gid.insert(make_pair(tuple._i, gi));
    _cid_to_gid.insert(make_pair(tuple._j, gj));
    _gid_to_cid.insert(make_pair(gi, tuple._i));
    _gid_to_cid.insert(make_pair(gj, tuple._j));
  }

  void update_code() {
    _can_code = id_generator++;
  }


  STORAGE_TYPE getCode() const {
    return _can_code;
  }

  //
  /*! \fn bool operator< (const CAN_CODE& rhs) const
   *  \brief A member function to canonical dfs code test, test for every edges lexicographically.
   *	\param rhs a constant reference of CAN_CODE.
   * \return boolean
   */
  bool operator< (const CAN_CODE& rhs) const {
    unsigned int i=0, j=0;
    while(i<_dfs_code.size() && j<rhs._dfs_code.size()) {
      if(_dfs_code[i] < rhs._dfs_code[j])  // comparing individual edge
        return true;
      i++;
      j++;
    }

    return false;

  }

  /*! \fn  int cid(const int& gi) const
   *  \brief A member function to get code id for a given graph id as parameter.
   *	\param gi a constant reference of integer
   * \return integer
   */
  int cid(const int& gi) const {
    VM_CONST_IT it=_gid_to_cid.find(gi);
    if(it==_gid_to_cid.end()) {
      return -1;
    }
    return it->second;
  }

  /*! \fn int gid(const int& ci) const
   *  \brief A member function to get graph id for a given code id as parameter.
   *	\param ci a constant reference of integer.
   * \return integer
   */
  int gid(const int& ci) const {
    VM_CONST_IT it=_cid_to_gid.find(ci);
    if(it==_cid_to_gid.end()) {
      return -1;
    }
    return it->second;
  }

  RMP_T& rmost_path() {
    return _rmp;
  }

  void append_rmp(const int& id) {
    _rmp.push_back(id);
  }

  typedef pair<V_T, pair<E_T, V_T> >  EDGE_T;

  struct ltedge {
    bool operator()(const EDGE_T& e1, const EDGE_T& e2) const {
      return ((e1.first < e2.first) ||
              (e1.first == e2.first && e1.second.first < e2.second.first) ||
              (e1.first == e2.first && e1.second.first == e2.second.first &&
               e1.second.second < e2.second.second));
    }
  };

  /**
   * Converts the canonical code to a string.
   */
  std::string to_string() const {

    if(_dfs_code.size() == 0) return "null";
    ostringstream t_ss;

    for(unsigned int i=0; i < _dfs_code.size(); i++) {
      if(i == 0)
        t_ss << _dfs_code[i];
      else
        t_ss << ":" << _dfs_code[i];
    }

    string t_str = t_ss.str();

    return t_str;
  }

  /*! \fn static double graph_distance(const CAN_CODE& c1, const CAN_CODE& c2)
   *  \brief A member function to find the distance between twwo cannnonical code.
   *	\param c1,c2 a constant reference of CAN_CODE.
   * \return double.
   */
  static double graph_distance(const CAN_CODE& c1, const CAN_CODE& c2) {
    multiset<EDGE_T, ltedge> set1, set2;
    vector<EDGE_T> result;
    CONST_IT cit;
    EDGE_T an_edge;
    for(cit = c1.begin(); cit < c1.end(); cit++) {
      if(cit->_li < cit->_lj)
        an_edge = make_pair(cit->_li, make_pair(cit->_lij, cit->_lj));
      else
        an_edge = make_pair(cit->_lj, make_pair(cit->_lij, cit->_li));
      set1.insert(an_edge);
    }
    for(cit = c2.begin(); cit < c2.end(); cit++) {
      if(cit->_li < cit->_lj)
        an_edge = make_pair(cit->_li, make_pair(cit->_lij, cit->_lj));
      else
        an_edge = make_pair(cit->_lj, make_pair(cit->_lij, cit->_li));
      set2.insert(an_edge);
    }
    set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), back_inserter(result));
    return 1 - (double)result.size()/max(set1.size(), set2.size());
  }
  friend ostream& operator<< <>(ostream&, const canonical_code<V_T, E_T>&);

private:
  STORAGE_TYPE _can_code;
  TUPLES _dfs_code;
  // the following two maps are very important. They maps vertex_id_in_code <---> vertex_id_in_graph
  // while we are making minimal code, we reassign vertex id according to minimal code, say in a graph
  // if we have edges like, D----C----D----B---A, there id's are like 0---1---2----3----4.
  // in min_can_code, A should have id-0, so in _cid_to_gid{0}  = 4, _gid_to_cid{4} = 0
  VID_HMAP _cid_to_gid;  //!< code -> candidate graph
  VID_HMAP _gid_to_cid;  //!< candidate graph -> code
  RMP_T _rmp;
  static int id_generator;
  static HASHNS::hash_map<string, int, hash_func<string>, equal_to<string> > level_one_hash;

};//end class canonical_code for graph

template<typename V_T, typename E_T>
ostream& operator<< (ostream& ostr, const canonical_code<V_T, E_T>& cc)
{
  typename canonical_code<V_T, E_T>::TUPLES::const_iterator it;
  for(it=cc._dfs_code.begin(); it!=cc._dfs_code.end(); it++)
    ostr<<*it<<endl;

  return ostr;
}

template<typename V, typename E>
int canonical_code<V, E>::id_generator = 1;

template<typename v, typename e>
HASHNS::hash_map<string, int, hash_func<string>, equal_to<string> >
canonical_code<v, e>::level_one_hash;

} // namespace hasan_sampler
#endif
