#ifndef __EXCEPTIONS_HPP__
#define __EXCEPTIONS_HPP__


#include <stdexcept>


//#define THROW(__exception__, __msg__) throw __exception__(__FILE__, __LINE__, __msg__)


class method_unimplemented : public std::runtime_error {
public:
  method_unimplemented(const char *err) : runtime_error(std::string("Method ") + err + "not implemented") {}
};


/*
class runtime_error_location : public std::runtime_error {
protected:
  int line;
  std::string filename;
public:
  runtime_error_location(std::string msg, std::string filename, int line)
    : runtime_error(msg) {
    this->line = line;
    this->filename = filename;
  }

  ~runtime_error_location() throw() { }

  virtual const char* what() const throw();
};


class runtime_error_method : public runtime_error_location {
protected:
  std::string method_signature;
  std::string extract_method_name();
  std::string extract_class_name();
public:
  runtime_error_method(std::string msg, std::string method_sig, std::string filename, int line)
    : runtime_error_location(msg, filename, line) {
    this->method_signature = method_sig;
  } // runtime_error_method

  ~runtime_error_method() throw() { }

  virtual const char* what() const throw();
};
*/

#endif

