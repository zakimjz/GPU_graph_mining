#ifndef __GRAPH_REPAIR_TOOLS_HPP__
#define __GRAPH_REPAIR_TOOLS_HPP__

void check_graph(const types::Graph &grph);
void check_database(const types::graph_database_t &db);
void fix_graph(types::Graph &grph);
void fix_database(types::graph_database_t &db);

void print_database_statistics(const types::graph_database_t &db);
#endif

