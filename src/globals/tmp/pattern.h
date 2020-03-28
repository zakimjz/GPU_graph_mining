/*
 * Stores the candidate pattern
 */
#ifndef PATTERN_H
#define PATTERN_H

#include "Globals.h"

class pattern
{
    public:
        pattern();
        void print();
        void add_vertex(string lab);
        void get_vertices(vi& v);
        void get_vertices_random(vi& rv);
        void get_vertex_pairs(vii& vp);
        void get_vpairs_random(vii& vp);
        string get_lab(int id);
        int get_size();
        void add_fwd(int src,string lab);
        void add_back(int src,int des);
        void add_edge(int srcid,int desid);
        void set_sup(int x);
        int get_sup();
        string get_string();
        bool is_empty();
    private:
        int sup;
        mis vlabs; // key is the vid and the value is the label of the vertex
        vector<ii> edges; // edges in the pattern
};
#endif
