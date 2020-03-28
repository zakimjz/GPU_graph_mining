#ifndef DEF_H
#define DEF_H
#include "Globals.h"

typedef struct
{
    int u;
    int v;
} Edge;

typedef struct
{
  int* vids; // Embeddings flattened
  long long int* hash;
  int* status;
  int ncols; // size of the pattern
  int nrows; // Number of embeddings
} Elist;

#endif
