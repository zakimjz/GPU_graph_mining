#ifndef CPEDGES_H
#define CPEDGES_H
// Copy the edges to the device memory
// Format : 
//  Edges : Adjacency list falttened to a single array
//  Index[i] : Offset for the adjacency list of vertex i
//  Count[i] : Degree of the vertex i in the graph
#include "Globals.h"
#include "def.h"
#include "storage.h"

void copy_edges(storage& strg,int** edlist_d,int** count_d,int** offset_d)
{
    // Number of vertices in the graph
  int nv = strg.num_vertices();
  int* count = (int*)malloc(sizeof(int)*nv);
  int* bcount = (int*)malloc(sizeof(int)*nv);
  memset(count,0,sizeof(int)*nv);
  // db.edges is the set of all edges in the graph
  tr(strg.db_edges,it) {
   count[it->first]++;
  }
  //memcpy(bcount,count,sizeof(int)*nv);
  memset(bcount,0,sizeof(int)*nv);
  // compute the offsets
  int* offset = (int*)malloc(sizeof(int)*nv);
  // exclusive prefix sum for the offsets
  eprefix_sum<int>(count,offset,nv,0);
  // Number of edges in graph
  int nedges = ar_sum<int>(count,nv,0);
  // Construct flattened edge list
  int* adjlist = (int*)malloc(sizeof(int)*nedges);
  tr(strg.db_edges,it) {
    int o1 = offset[it->first]+bcount[it->first];
    adjlist[o1] = it->second;
    bcount[it->first]++;
  }
  free(bcount);
  int* samp;
  int** samp2;
  RC0(cudaMalloc((void**)edlist_d,nedges*sizeof(int)));
  RC0(cudaMalloc((void**)count_d,nv*sizeof(int)));
  RC0(cudaMalloc((void**)offset_d,nv*sizeof(int)));
  RC0(cudaMemcpy((void*)*edlist_d,adjlist,nedges*sizeof(int),cudaMemcpyHostToDevice));
  RC0(cudaMemcpy((void*)*count_d,count,nv*sizeof(int),cudaMemcpyHostToDevice));
  RC0(cudaMemcpy((void*)*offset_d,offset,nv*sizeof(int),cudaMemcpyHostToDevice));
  free(count);
  free(offset);
  free(adjlist);
  return;
}

#endif
