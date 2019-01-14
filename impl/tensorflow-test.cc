#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <sstream>
#include <limits>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

// TODO: should be <tensorflow/c/c_api.h> instead, but cmake is misconfigured
#include <c_api.h>

using namespace std;

int main(int argc, char** argv) {
    cout << "Hello from TensorFlow C library version " << TF_Version() << endl;
}
