#include <logger.hpp>
#include <iostream>

/*
 * compile:
 * g++ -o lgsmpl logger.cpp logger_sample.cpp utils.cpp -I. -DLOG_TRACE
 *
 */

int main()
{
  Logger *lg = Logger::get_logger("MAIN");
  std::filebuf *fbuf = new std::filebuf();
  fbuf->open("test.log", std::ios::out | std::ios::trunc);
  Logger::set_global_streambuf(fbuf, true);

  int var = 0;
  string tst = "aaaa"

  INFO(*lg, "test message: " << var);
  DEBUG(*lg, "test message: " << tst);
  return 0;
}

