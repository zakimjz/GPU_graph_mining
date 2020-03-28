#ifndef __CUDA_CONFIGURATION_PARAMSX_HPP__
#define __CUDA_CONFIGURATION_PARAMSX_HPP__

#include <iostream>
#include <fstream>
#include <utils.hpp>
#include <vector>
#include <map>

#include <cuda_computation_parameters.hpp>

namespace cudapp {

class cuda_configurator {
protected:
  std::string config_file;
  std::string config_id;
  std::map<std::string, int> parameters;

  static int max_block_dim; //< maximal number of threads in a block (hardware limitation)
  static int max_grid_dim; //< maximal number of blocks in the grid (hardware limitation)
public:
  cuda_configurator();
  cuda_configurator(std::string config_file, std::string config_id);
  virtual cudapp::cuda_computation_parameters get_exec_config(std::string key, int total_threads);
  virtual void set_parameter(std::string key, int threads_per_block);
  virtual int get_parameter(std::string key);
  virtual void parse_config_file();
  virtual std::string to_string();

  static cudapp::cuda_computation_parameters get_computation_parameters(int total_threads, int max_threads_per_block);
};


class cuda_configurator_vertical_format : public cuda_configurator {
protected:
public:
  cuda_configurator_vertical_format(std::string config_file, std::string config_id);
  virtual void parse_config_file();
};


} // namespace cudapp

#endif

