#include "pattern.h"

pattern::pattern() {
    this->sup = 0;
}

void pattern::print() {
    cout<<get_string()<<endl;
}

void pattern::add_vertex(string lab)
{
    int nid = get_size();
    vlabs[nid] = lab;
}

void pattern::get_vertices(vi& v)
{
    v.clear();
    tr(vlabs,it){ v.push_back(it->first);}
}

void pattern::get_vertices_random(vi& v) {
    get_vertices(v);
    random_shuffle(all(v));
}

void pattern::get_vertex_pairs(vii& vp)
{
    vp.clear();
    int id1,id2;
    tr(vlabs,it) {
        tr(vlabs,it2) {
            id1 = it->first;
            id2 = it2->first;
            if(id1!=id2 && !cpresent(edges,ii(id1,id2)) && !cpresent(edges,ii(id2,id1)))
                vp.push_back(ii(id1,id2));
        }
    }
}

void pattern::get_vpairs_random(vii& vp) {
    get_vertex_pairs(vp);
    random_shuffle(all(vp));
}

string pattern::get_lab(int id) {
    return vlabs[id];
}

int pattern::get_size() {
    return vlabs.size();
}

void pattern::add_fwd(int src,string lab) {
    // vertices are all the vertices in the graph
    add_vertex(lab);
    int desid = get_size()-1;
    add_edge(src,desid);
}

void pattern::add_back(int src,int des) {
    add_edge(src,des);
}

void pattern::add_edge(int srcid,int desid) {
    edges.push_back(ii(srcid,desid));
}

void pattern::set_sup(int x) {
    this->sup = x;
}

int pattern::get_sup() {
    return this->sup;
}

string pattern::get_string() {
    // return the pattern as a string
    //DBGVAR(cout,vlabs.size());
    //DBGVAR(cout,edges.size());
    string ret;
    tr(vlabs,it0) {
        ret += "v "+convertInt(it0->first)+" "+it0->second+"\n";
    }
    tr(edges,it) {
        ret += "e "+convertInt(it->first)+" "+convertInt(it->second)+" "+vlabs[it->first]+" "+vlabs[it->second]+"\n";
    }
    //DBGVAR(cout,sup);
    ret += "Support: "+convertInt(sup)+"\n"; 
    return ret;
}


bool pattern::is_empty() {
    return get_size()==0;
}

