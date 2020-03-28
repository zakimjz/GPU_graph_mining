#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <string>
#include <time.h>
#include <vector>

#define MEGABYTES (1024 * 1024)

namespace utils {

std::string trim(const std::string& s, const std::string& drop = " ");


bool is_na_symbol(const char *str);
bool is_symbol(const char *str);
bool is_string(const char *cstr);

bool parse_int(const char *str, int &ret);
bool parse_uint(const char *cstr, unsigned int &ret);
bool parse_double(const char *str, double &ret);
bool parse_long(const char *str, long &ret);
bool parse_ulong(const char *str, unsigned long &ret);

bool is_int(const char *str);
bool is_double(const char *str);
bool is_uint(const char *cstr);
bool is_long(const char *cstr);
bool is_ulong(const char *cstr);


bool is_symbol(const char *str);
bool is_na_symbol(const char *str);

void split(const std::string& str,
           std::vector<std::string>& tokens,
           const std::string& delimiters = " ");


time_t get_sec(timeval & start, timeval &end);
time_t get_usec(timeval & start, timeval &end);
double get_time_diff(timeval & start, timeval &end);

long get_program_size();
long get_rss();
long get_rss_limit();

std::string get_current_timestamp();
std::string print_vector(const std::vector<int> &vec);

} // namespace utils
#endif
