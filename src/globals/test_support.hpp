#ifndef __TEST_SUPPORT_HPP__
#define __TEST_SUPPORT_HPP__

#include <graph_types.hpp>

namespace gspan_cuda {

types::graph_database_t get_test_database();
types::graph_database_t get_labeled_test_database();
types::graph_database_t get_labeled_test_database2();

} // namespace gspan_cuda

#endif

