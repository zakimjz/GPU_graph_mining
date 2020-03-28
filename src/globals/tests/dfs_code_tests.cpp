#include <dfs_code.hpp>
#include <string>
#include <gtest/gtest.h>
#include <stdexcept>

using std::string;
using std::runtime_error;
using types::DFS;


TEST(dfs_code, basic_to_from_string)
{
  const char *good_dfs_strings[]  = {"(0 1 2 3 4)",
                                     "(-1 -1 -1 -1 -1)",
                                     0};

  DFS good_dfs[] = { DFS(0, 1, 2, 3, 4), DFS(-1, -1, -1, -1, -1) };
  const char *bad_dfs_strings[]  = {"(0 1 2 3)", // missing number
                                    "(0 1 2)", // missing number
                                    "(0 1)", // missing number
                                    "(0)", // missing number
                                    "()", // missing number
                                    "(a b c d f)", // wrong content
                                    "garbage",
                                    0};

  const char **bad_str = bad_dfs_strings;
  while(*bad_str) {
    DFS parsed;
    EXPECT_THROW(parsed = DFS::parse_from_string(*bad_str), runtime_error);
    bad_str++;
  }


  for(int i = 0; good_dfs_strings[i] != 0; i++) {
    DFS fromstr = DFS::parse_from_string(good_dfs_strings[i]);
    EXPECT_EQ(fromstr, good_dfs[i]);
  }

} // TEST(dfs_code, basic_to_from_string)



TEST(dfs_code, compare_test)
{
  string min_dfs_code_maybe_str = "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 0 0 0 0)";
  string codes_str[] = {"(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 1 0 0 0);(3 0 0 0 0)",
                        "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 0 0 0 0)",
                        "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 1 -1 0 -1)",
                        ""}; // ??

  types::DFSCode min_dfs_code_maybe = types::DFSCode::read_from_str(min_dfs_code_maybe_str);
  int idx = 0;
  while(codes_str[idx].size() != 0) {
    types::DFSCode dfs_code_tmp = types::DFSCode::read_from_str(codes_str[idx]);
    idx++;

    //cout << dfs_code_tmp
  } // while
} // dfs_code, compare_test


TEST(dfs_code, serializable_basic_test)
{
  string codes_str[] = {"(0 1 0 0 0);(1 2 0 0 0);(2 3 0 0 0);(3 1 0 0 0);(3 0 0 0 0)",
                        "(0 1 0 0 0);(1 2 0 0 0);(2 0 0 0 0);(2 3 0 0 0);(3 0 0 0 0)",
                        "(0 1 0 0 0);(1 2 -1 0 0);(2 0 -1 0 -1);(2 3 -1 0 0);(3 1 -1 0 -1)",
                        "(-1 -1 -1 -1 -1)",
                        ""}; // ??

  types::DFSCode dfscode;
  int idx = 0;
  DFS tmp_dfs;

  while(codes_str[idx].size() != 0) {
    types::DFSCode dfs_code_tmp = types::DFSCode::read_from_str(codes_str[idx]);
    size_t buff_size = dfs_code_tmp.get_serialized_size();
    char *buffer = new char[buff_size];
    size_t written = dfs_code_tmp.serialize(buffer, buff_size);

    EXPECT_EQ(buff_size, written);

    types::DFSCode deser_dfs_code_tmp;
    size_t read = deser_dfs_code_tmp.deserialize(buffer, written);

    EXPECT_EQ(deser_dfs_code_tmp, dfs_code_tmp);
    EXPECT_EQ(read, written);

    string dfs_code_str = dfs_code_tmp.to_string();
    string deser_dfs_code_str = deser_dfs_code_tmp.to_string();
    EXPECT_EQ(dfs_code_str, deser_dfs_code_str);
    delete [] buffer;
    idx++;
  } // while


  // empty dfs code
  {
    types::DFSCode empty_dfs;
    size_t buff_size = empty_dfs.get_serialized_size();
    char *buffer = new char[buff_size];
    size_t written = empty_dfs.serialize(buffer, buff_size);
    EXPECT_EQ(written, buff_size);
    types::DFSCode deser_empty_dfs;
    size_t read = deser_empty_dfs.deserialize(buffer, buff_size);
    EXPECT_EQ(written, read);
    delete [] buffer;
  }
} // dfs_code, serializable_basic_test

