//
// Created by dfried on 1/21/18.
//

#ifndef CNN_UTILS_H
#define CNN_UTILS_H

#include <stdio.h>
#include <dirent.h>
#include <string>
#include <assert.h>

#include <iomanip>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>

namespace utils {

  std::vector<boost::filesystem::path> glob_files(const std::string &glob) {
      assert(glob.find(' ') == std::string::npos);
      boost::filesystem::path glob_path(glob);
      boost::filesystem::path parent_dir = absolute(glob_path.parent_path());
      std::string file_glob = glob_path.filename().string();
      file_glob = boost::replace_all_copy(file_glob, ".", "\\.");
      file_glob = boost::replace_all_copy(file_glob, "*", ".*");

      const boost::regex file_regex(file_glob);

      std::vector<boost::filesystem::path> all_matching_files;

      // https://stackoverflow.com/questions/1257721/can-i-use-a-mask-to-iterate-files-in-a-directory-with-boost
      boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
      for (boost::filesystem::directory_iterator i(parent_dir); i != end_itr; ++i) {
          // Skip if not a file
          if (!boost::filesystem::is_regular_file(i->status())) continue;

          boost::smatch what;

          // cerr << "filename: " << i->path().filename().string() << " ";
          // cerr << "regex: " << file_regex << endl;

          if (!boost::regex_match(i->path().filename().string(), what, file_regex)){
              // cerr << endl;
              continue;
          }
          // cerr << "match" << endl;

          // File matches, store it
          all_matching_files.push_back(i->path());
      }

      return all_matching_files;
  }

  // https://stackoverflow.com/questions/29200635/convert-float-to-string-with-set-precision-number-of-decimal-digits
  std::string to_string_precision(float value, unsigned precision) {
      stringstream stream;
      stream << fixed << setprecision(precision) << value;
      return stream.str();
  }
}

#endif //CNN_UTILS_H
