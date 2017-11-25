#include "nt-parser/eval.h"

#include <iostream>
#include <cstdio>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <fstream>
#include <regex>
#include <utility>

using namespace std;
namespace io = boost::iostreams;

pair<Metrics, vector<MatchCounts>> metrics_from_evalb(const string& ref_fname, const string& hyp_fname, const string& evalbout_fname) {

  string cmd = "EVALB/evalb -p EVALB/COLLINS.prm " + ref_fname + " " + hyp_fname + " > " + evalbout_fname;
  system(cmd.c_str());

  std::ifstream evalfile(evalbout_fname);

  // adapted from https://github.com/mitchellstern/neural-parser/blob/master/src/common/evaluation.cpp
  std::regex recall_regex{R"(Bracketing Recall\s+=\s+(\d+\.\d+))"};
  std::regex precision_regex{R"(Bracketing Precision\s+=\s+(\d+\.\d+))"};
  std::regex fmeasure_regex{R"(Bracketing FMeasure\s+=\s+(\d+\.\d+))"};

  Metrics results{NAN, NAN, NAN};

    std::string line;

    for (unsigned i{0}; i < 4; ++i)
        std::getline(evalfile, line);

    std::vector<MatchCounts> match_counts;

    unsigned i = 0;
    while (line[0] != '=') {
        i++;
        std::istringstream iss{line};

        unsigned index;
        iss >> index;
        assert(iss);
        assert(index == i);

        unsigned length;
        iss >> length;
        assert(iss);

        unsigned status;
        iss >> status;
        assert(iss);

        double recall;
        iss >> recall;
        assert(iss);

        double precision;
        iss >> precision;
        assert(iss);

        int correct;
        iss >> correct;
        assert(iss);

        int gold;
        iss >> gold;
        assert(iss);

        int predicted;
        iss >> predicted;
        assert(iss);

        match_counts.push_back(MatchCounts(correct, gold, predicted));
        std::getline(evalfile, line);
    }

  while (std::getline(evalfile, line)) {
    std::smatch match;
    if (std::regex_match(line, match, recall_regex))
      results.recall = std::stod(match[1].str());
    else if (std::regex_match(line, match, precision_regex))
      results.precision = std::stod(match[1].str());
    else if (std::regex_match(line, match, fmeasure_regex)) {
      results.f1 = std::stod(match[1].str());
      break;
    }
  }

  return pair<Metrics, vector<MatchCounts>>(results, match_counts);
}

