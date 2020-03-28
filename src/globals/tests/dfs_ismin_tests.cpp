#include <utils.hpp>
#include <dfs_code.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <iostream>
#include <logger.hpp>

using utils::split;
using std::string;
using std::runtime_error;
using types::DFS;
using types::DFSCode;

static Logger *ismin_log = Logger::get_logger("GLOBALSTEST");

/*
   static DFSCode read_from_str(const string &str)
   {
   std::vector<std::string> vec_str;
   split(str, vec_str, ";");

   DFSCode result;

   for(int i = 0; i < vec_str.size(); i++) {
    DFS d = DFS::parse_from_string(vec_str[i].c_str());
    result.push_back(d);
   } // for i

   return result;
   }
 */

static bool static_ismin(string str_dfs_code)
{
  DFSCode dfs_code = DFSCode::read_from_str(str_dfs_code);
  dfs_code.buildRMPath();
  DEBUG(*ismin_log, "parsed dfs code: " << dfs_code.to_string());


  bool ismin = dfs_code.dfs_code_is_min();
  DEBUG(*ismin_log, "dfs_code: " << dfs_code.to_string() << "; ismin: " << ismin);
  return ismin;
}


TEST(dfs_ismin, basic_test)
{
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 0 -1 0 -1)")); // is min
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 -1 0 0)")); // is min
  EXPECT_FALSE(static_ismin("(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 1 -1 0 -1)"));

  //
  EXPECT_FALSE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(1 3 0 0 0)"));
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0)")); // graph sample 1
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0)")); // graph sample 2
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(1 3 0 0 0)")); // graph sample 3
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(1 3 0 0 0);(1 4 0 0 0)")); // graph sample 4
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(1 3 0 0 0);(1 4 0 0 0)")); // graph sample 5
  EXPECT_TRUE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(2 4 0 0 0);(1 5 0 0 0)")); // graph sample 6
  EXPECT_FALSE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 1 0 0 0)"));
  EXPECT_FALSE(static_ismin("(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(1 3 0 0 0)"));
  //EXPECT_TRUE(static_ismin(""));



  /*
     string str_dfs_code = "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 0 -1 0 -1)"; // is min
     //string str_dfs_code = "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 0 0 0 0)"; // is min
     //string str_dfs_code = "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 1 -1 0 -1)"; // is not min
     //string str_dfs_code = "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0)"; // error
     //string str_dfs_code = "(0 1 0 0 0);(1 2 -1 0 0)"; // is min
     //string str_dfs_code = "(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 1 0 0 0);(3 0 0 0 0)"; // ??
     DFSCode dfs_code = DFSCode::read_from_str(str_dfs_code);
     dfs_code.buildRMPath();
     DEBUG(*ismin_log, "parsed dfs code: " << dfs_code.to_string());


     bool ismin = dfs_code.dfs_code_is_min();
     std::cout << "dfs_code: " << dfs_code.to_string() << "; ismin: " << ismin << std::endl;
   */
} //



