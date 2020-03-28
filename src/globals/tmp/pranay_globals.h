#ifndef GLOBALS_H
#define GLOBALS_H

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cmath>
#include <queue>
#include <cstdlib>
#include <string>
#include <string.h>
#include <time.h>
#include "stdio.h"
#include <fstream>
#include <sstream>
#include <assert.h>
#include <queue>
#include <limits>
#include <omp.h>


/*#include <unordered_set>
#include <unordered_map>*/
#include "stdlib.h"

using namespace std;
#define tr(container, it) for(typeof(container.begin()) it = container.begin(); it != container.end(); it++)
#define all(c) c.begin(), c.end()
#define present(container, element) (container.find(element) != container.end())
#define cpresent(container, element) (find(all(container),element) != container.end())
#define REP(i, a, b) for (int i = (int)a; i < (int)b; i++)
#define PER(i, a, b) for (int i = (int)a; i < (int)b; i--)
#define vi(a) vector<int> a
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define PR(x) cout<<STR(x)<<":"<<x<<endl;
#define RP(x) cout<<x<<endl;
#define START(x) do{ strg.st->start_stat(x);} while(0)
#define SSTART(x,y) do{ strg.st->add_counter(x,y); strg.st->start_stat(x);} while(0)
#define STOP(x) do{ strg.st->end_stat(x);} while(0)

#define ISBITSET(x,i) ((x[i>>3] & (1<<(i&7)))!=0)
#define SETBIT(x,i) x[i>>3]|=(1<<(i&7));
#define CLEARBIT(x,i) x[i>>3]&=(1<<(i&7))^0xFF;

#define DEBUG_LEVEL 1
#define __(x, line)   (std::cerr << __FUNCTION__ << ":" << line << ": " << #x << " = " << (x) << std::endl)
#define DEBUG_OUTPUT(level , message)   ( (DEBUG_LEVEL >= level) ? (std::cerr << __FUNCTION__ << ":" << __LINE__ << ": " << message << std::endl) : std::cerr )
#define INFO_OUTPUT(level, message)   ( (DEBUG_LEVEL >= level) ? (std::cerr << message ) : std::cerr )
#define DBGVAR( os, var ) \
      (os) << "DBG: " << __FILE__ << "{" << __FUNCTION__ << "} "<<"(" << __LINE__ << ") "\
       << #var << " = [" << (var) << "]" << std::endl

typedef vector<double> vd;
typedef vector<int> vi;
typedef vector<vector<int> > vvi;
typedef vector<vector<double> > vvd;
typedef vector<string> vs;
typedef map<string,long long> msl; // map of string and long long
typedef map<string,bool> msb; // map of string and bool
typedef pair<int,int> ii;
typedef map<int,string> mis;
typedef map<string,int> msi;
typedef vector<ii> vii;
typedef pair<string,string> ss;
typedef map<int,int> mii;
typedef map<string,string> mss;
typedef pair<string,string> ss;
void print_vi(const vi& x);
void print_ii(const ii& x);
void print_vii(const vii& x);
void print_vd(const vd& x);
void print_vvi(const vvi& x);

float RangeFloat(float min, float max);

void split(const std::string &s,vector<string>& words, char delim );

string convertInt(int number);

int convertToInt(string num);
double convertToDouble(string num);
struct compare2nd
{
    template <typename T>
        bool operator()(const T& pLhs, const T& pRhs)
        {
            return pLhs.second < pRhs.second;
        }
};


#endif
