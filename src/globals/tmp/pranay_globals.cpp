#include "Globals.h"

void print_vi(const vi& x)
{
    tr(x,it)
    {
        cout<<","<<*it<<",";
    }
    cout<<endl<<"------------------"<<endl;
}
void print_vd(const vd& x)
{
    tr(x,it)
    {
        cout<<","<<*it<<",";
    }
    cout<<endl<<"------------------"<<endl;
}
void print_vvi(const vvi& x)
{
    tr(x,it)
    {
        print_vi(*it);
    }
}

float RangeFloat(float min, float max)
{
    // generate a float between min and maximum
    float random = ((float) rand() ) / (float) RAND_MAX;
    float range = max - min;
    return (random*range) + min;
}

void print_ii(const ii& x)
{
    cout << "(" << x.first<< "," << x.second << ")" << endl;
}
void print_vii(const vii& x)
{
    tr(x,it)
    {
        print_ii(*it);
    }
}

void split(const std::string &s,vector<string>& words, char delim )
{
    words.clear();
    std::stringstream iss(s);
    std::string item;
    while(std::getline(iss, item, delim)) {
            words.push_back(item);
        }
}

string convertInt(int number)
{
    stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}

int convertToInt(string num) {
    // Convert the string representation of a number to integer
    std::stringstream str(num); 
    int x;  
    str >> x;  
    if (!str) {
        cerr << "Conversion to int failed " << num <<endl;
    } 
    return x;
}

double convertToDouble(string num) {
    // Convert the string representation of a number to integer
    std::stringstream str(num); 
    double x;  
    str >> x;  
    if (str.fail()) {
        cerr << "Conversion to double failed " << num <<endl;
    } 
    return x;
}
