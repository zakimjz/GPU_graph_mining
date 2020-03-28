#include <cuda_configurator.hpp>
#include <cuda_computation_parameters.hpp>
#include <string>
#include <stdexcept>


using std::string;

namespace cudapp {

// as specified for compute capability 2.0
int cuda_configurator::max_block_dim = 1024;
int cuda_configurator::max_grid_dim = 65535;

cuda_configurator::cuda_configurator()
{
}

cuda_configurator::cuda_configurator(std::string config_file, std::string cid) : config_id(cid)
{
  this->config_file = config_file;
  this->config_id = cid;
}


void cuda_configurator::parse_config_file()
{
  std::ifstream is;
  is.open(config_file.c_str());

  if(is.is_open() == false) {
    throw std::runtime_error("cuda_configurator: could not open config file");
  }

  string line;
  std::vector<std::string> result;
  bool header = true;
  bool found_config = false;
  std::map<int, string> col_param_map;

  while(true) {
    unsigned int pos = is.tellg();
    if(!std::getline(is, line)) {
      break;
    }
    result.clear();
    utils::split(line, result, ",");


    if(header) {
      for(int i = 1; i < result.size(); i++) {
        //result[i].erase(std::remove(result[i].begin(),result[i].end(),' '), result[i].end());
        string param = utils::trim(result[i], " ");
        parameters.insert(std::make_pair(param, -1));
        col_param_map[i] = param;
      }
      header = false;
      continue;
    }

    if(result[0] == config_id) {
      for(int i = 1; i < result.size(); i++) {
        int val = atoi(result[i].c_str());
        std::string param = col_param_map[i];
        parameters[param] = val;
      } // for i
      is.close();
      found_config = true;
      break;
    } // if
  } // while

  if(found_config == false) {
    throw std::runtime_error("Did not find configuration for given ID.");
  } // if
}


cudapp::cuda_computation_parameters cuda_configurator::get_exec_config(std::string key, int total_threads)
{
  if(parameters.find(key) == parameters.end()) {
    std::stringstream ss;
    ss << "Requesting invalid key: " << key;
    throw std::runtime_error(ss.str());
  }
  int max_threads_per_block = parameters[key];
  int blocks = total_threads / max_threads_per_block + int(total_threads % max_threads_per_block > 0);
  int values_per_thread = 1;
  if(blocks >= max_grid_dim) {
    blocks = max_grid_dim;
    values_per_thread = total_threads / (blocks * max_threads_per_block) + 1;
  }
  cudapp::cuda_computation_parameters params(blocks, max_threads_per_block, values_per_thread);

  return params;
} // cuda_configurator::get_exec_config


cudapp::cuda_computation_parameters cuda_configurator::get_computation_parameters(int total_threads, int max_threads_per_block)
{
  //int max_threads_per_block = parameters[key];
  int blocks = total_threads / max_threads_per_block + int(total_threads % max_threads_per_block > 0);
  int values_per_thread = 1;
  if(blocks >= max_grid_dim) {
    blocks = max_grid_dim;
    values_per_thread = total_threads / (blocks * max_threads_per_block) + 1;
  }
  cudapp::cuda_computation_parameters params(blocks, max_threads_per_block, values_per_thread);
  return params;
}

void cuda_configurator::set_parameter(std::string key, int threads_per_block)
{
  parameters[key] = threads_per_block;
}


int cuda_configurator::get_parameter(std::string key)
{
  return parameters[key];
}


string cuda_configurator::to_string()
{
  std::map<std::string, int> parameters;
  std::stringstream ss;
  for(std::map<std::string, int>::iterator it = parameters.begin(); it != parameters.end(); it++) {
    ss << "[" << it->first << ", " << it->second << "]; ";
  }
  return ss.str();
}











cuda_configurator_vertical_format::cuda_configurator_vertical_format(std::string config_file, std::string config_id)
  : cuda_configurator(config_file, config_id)
{
}



void cuda_configurator_vertical_format::parse_config_file()
{
  std::ifstream is;
  is.open(config_file.c_str());

  if(is.is_open() == false) {
    throw std::runtime_error("cuda_configurator_vertical_format: could not open config file");
  }

  string line;
  std::vector<std::string> result;
  int column_with_config = -1;

  if(!std::getline(is, line)) {
    throw std::runtime_error("Could not read header from config file.");
  }


  result.clear();
  utils::split(line, result, ",");
  string param = utils::trim(result[0], " ");
  if(param != "parameter") {
    throw std::runtime_error("Invalid configuration file format !? First row, first element should contain \"parameter\"");
  }
  for(int i = 0; i < result.size(); i++) {
    param = utils::trim(result[i], " ");
    if(param == config_id) {
      column_with_config = i;
    } // if
  } // for i

  if(column_with_config == -1) {
    throw std::runtime_error("Configuration parameter not found");
  }

  while(true) {
    if(!std::getline(is, line)) {
      break;
    } // if
    result.clear();
    utils::split(line, result, ",");
    std::string param = result[0];
    parameters[param] = atoi(result[column_with_config].c_str());
  } // while


}

} // namespace cudapp

