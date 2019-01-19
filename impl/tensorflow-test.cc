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

// modified from GPUDeviceName in https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api_test.cc
vector<string> get_device_names(TF_Session* session) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();
  std::unique_ptr<TF_DeviceList, decltype(&TF_DeleteDeviceList)> list(
      TF_SessionListDevices(session, s), TF_DeleteDeviceList);
  TF_DeviceList* device_list = list.get();

  vector<string> device_names;
  const int num_devices = TF_DeviceListCount(device_list);
  for (int i = 0; i < num_devices; ++i) {
    const char* device_name = TF_DeviceListName(device_list, i, s);
    const char* device_type = TF_DeviceListType(device_list, i, s);
    device_names.push_back(string(device_name));
  }
  // No GPU device found.
  return device_names;
}

vector<string> get_device_names() {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Status* s = status.get();
  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> graph(TF_NewGraph(),
                                                             TF_DeleteGraph);

  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* sess = TF_NewSession(graph.get(), opts, s);
  TF_DeleteSessionOptions(opts);

  vector<string> device_names = get_device_names(sess);
  TF_DeleteSession(sess, s);
  return device_names;
}

int main(int argc, char** argv) {
    cout << "Hello from TensorFlow C library version " << TF_Version() << endl;
    cout << "available devices: " << endl;
    for (auto &d : get_device_names()) {
        cout << d << endl;
    }
}
