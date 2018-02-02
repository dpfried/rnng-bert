#include "nt-parser/eval.h"

#include <iostream>
#include <cstdio>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <fstream>
#include <boost/regex.hpp>
#include <utility>

using namespace std;
namespace io = boost::iostreams;

pair<Metrics, vector<MatchCounts>> metrics_from_evalb(const string& ref_fname, const string& hyp_fname, const string& evalbout_fname, bool spmrl) {

    string cmd;
    if (spmrl) {
      cmd = "evalb_spmrl2013/evalb_spmrl -p evalb_spmrl2013/spmrl.prm " + ref_fname + " " + hyp_fname + " > " + evalbout_fname;
    } else {
      cmd = "EVALB/evalb -p EVALB/COLLINS_ch.prm " + ref_fname + " " + hyp_fname + " > " + evalbout_fname;
    }
    system(cmd.c_str());

    std::ifstream evalfile(evalbout_fname);

    // adapted from https://github.com/mitchellstern/neural-parser/blob/master/src/common/evaluation.cpp
    boost::regex recall_regex{R"(Bracketing Recall\s+=\s+(\d+\.\d+))"};
    boost::regex precision_regex{R"(Bracketing Precision\s+=\s+(\d+\.\d+))"};
    boost::regex fmeasure_regex{R"(Bracketing FMeasure\s+=\s+(\d+\.\d+))"};
    boost::regex complete_match_regex{R"(Complete match\s+=\s+(\d+\.\d+))"};

    Metrics results(0.0, 0.0, 0.0);

    std::string line;

    for (unsigned i{0}; i < 4; ++i)
        std::getline(evalfile, line);

    std::vector<MatchCounts> match_counts;

    unsigned i = 0;
    while (line[0] != '=') {
        i++;
        std::istringstream iss{line};
        if (!iss) {
            break;
        }

        unsigned index;
        iss >> index;
        if (!iss || index != i) {
            break;
        }

        unsigned length;
        iss >> length;
        if (!iss) {
            break;
        }

        unsigned status;
        iss >> status;
        if (!iss) {
            break;
        }

        double recall;
        iss >> recall;
        if (!iss) {
            break;
        }

        double precision;
        iss >> precision;
        if (!iss) {
            break;
        }

        int correct;
        iss >> correct;
        if (!iss) {
            break;
        }

        int gold;
        iss >> gold;
        if (!iss) {
            break;
        }

        int predicted;
        iss >> predicted;
        if (!iss) {
            break;
        }

        match_counts.push_back(MatchCounts(correct, gold, predicted));
        std::getline(evalfile, line);
    }

    while (std::getline(evalfile, line)) {
        boost::smatch match;
        if (boost::regex_match(line, match, recall_regex))
            results.recall = std::stod(match[1].str());
        else if (boost::regex_match(line, match, precision_regex))
            results.precision = std::stod(match[1].str());
        else if (boost::regex_match(line, match, fmeasure_regex)) {
            results.f1 = std::stod(match[1].str());
        } else if (boost::regex_match(line, match, complete_match_regex)) {
            results.complete_match = std::stod(match[1].str());
            break;
        }
    }

    return pair<Metrics, vector<MatchCounts>>(results, match_counts);
}

