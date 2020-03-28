#include <cuda_configurator.hpp>
#include <gtest/gtest.h>


TEST(cuda_configurator, basic_test)
{
  {
    cudapp::cuda_configurator cc("tests/test.cfg", "XXXX");
    cc.parse_config_file();

    ASSERT_EQ(cc.get_parameter("compute_extensions.first"), 100);
    ASSERT_EQ(cc.get_parameter("compute_extensions.second"), 200);
    ASSERT_EQ(cc.get_parameter("compute_support.first"), 100);
  }


  {
    cudapp::cuda_configurator cc("tests/test.cfg", "YYYY");
    cc.parse_config_file();

    ASSERT_EQ(cc.get_parameter("compute_extensions.first"), 10);
    ASSERT_EQ(cc.get_parameter("compute_extensions.second"), 20);
    ASSERT_EQ(cc.get_parameter("compute_support.first"), 50);
  }

  try {
    cudapp::cuda_configurator cc("tests/test.cfg", "ZZZZ");
    cc.parse_config_file();

    ASSERT_EQ(cc.get_parameter("compute_extensions.first"), 10);
    ASSERT_EQ(cc.get_parameter("compute_extensions.second"), 20);
    ASSERT_EQ(cc.get_parameter("compute_support.first"), 50);
    ASSERT_TRUE(false);
  } catch(std::exception &e) {
    ASSERT_TRUE(true);
  }
}



TEST(cuda_configurator_vertical_format, basic_test)
{
  {
    cudapp::cuda_configurator_vertical_format cc("tests/vertical_config.cfg", "XXXX");
    cc.parse_config_file();

    ASSERT_EQ(cc.get_parameter("compute_extensions.first"), 100);
    ASSERT_EQ(cc.get_parameter("compute_extensions.second"), 200);
    ASSERT_EQ(cc.get_parameter("compute_support.first"), 100);
  }


  {
    cudapp::cuda_configurator_vertical_format cc("tests/vertical_config.cfg", "YYYY");
    cc.parse_config_file();

    ASSERT_EQ(cc.get_parameter("compute_extensions.first"), 10);
    ASSERT_EQ(cc.get_parameter("compute_extensions.second"), 20);
    ASSERT_EQ(cc.get_parameter("compute_support.first"), 50);
  }

  try {
    cudapp::cuda_configurator_vertical_format cc("tests/vertical_config.cfg", "ZZZZ");
    cc.parse_config_file();

    ASSERT_EQ(cc.get_parameter("compute_extensions.first"), 10);
    ASSERT_EQ(cc.get_parameter("compute_extensions.second"), 20);
    ASSERT_EQ(cc.get_parameter("compute_support.first"), 50);
    ASSERT_TRUE(false);
  } catch(std::exception &e) {
    ASSERT_TRUE(true);
  }
}



