#ifndef NTPARSER_EVAL_H_
#define NTPARSER_EVAL_H_

#include <string>
#include <vector>
#include <utility>

using namespace std;

struct Metrics {
public:
  Metrics(double precision, double recall, double f1, double complete_match):
          precision(precision), recall(recall), f1(f1), complete_match(complete_match) {}

  Metrics(double precision, double recall, double f1):
      Metrics(precision, recall, f1, 0) {}

  double precision,recall,f1,complete_match;
};

struct MatchCounts {
public:
  MatchCounts() {}

  MatchCounts(unsigned correct, unsigned gold, unsigned predicted):
  correct(correct), gold(gold), predicted(predicted) {};

  unsigned correct = 0;
  unsigned predicted = 0;
  unsigned gold = 0;

  Metrics metrics() const {
      double precision = 0.0;
      if (predicted > 0)
          precision = (100.0 * correct) / predicted;

      double recall = 0.0;
      if (gold > 0)
          recall = (100.0 * correct) / gold;

      double f1 = 0.0;
      if (precision + recall > 0)
          f1 = (2 * precision * recall) / (precision + recall);

      return Metrics(precision, recall, f1);
  }

  MatchCounts& operator+=(const MatchCounts& m) {
      correct += m.correct;
      predicted += m.predicted;
      gold += m.gold;
  }

  bool operator==(const MatchCounts& other) {
      //return m1.correct == m2.correct && m1.predicted == m2.predicted && m1.gold == m2.gold;
      return correct == other.correct && predicted == other.predicted && gold == other.gold;
  }

  bool operator!=(const MatchCounts& other) {
      return !(*this == other);
  }
};

inline MatchCounts operator+(MatchCounts m1, const MatchCounts& m2) {
    m1 += m2;
    return m1;
}

/*
ostream& operator<<(ostream& os, const MatchCounts& counts) {
    std::ostringstream buffer;
    buffer << "correct=" << counts.correct
           << ", gold=" << counts.gold
           << ", predicted=" << counts.predicted;
    os << buffer.str();
    return os;
}
 */

pair<Metrics, vector<MatchCounts>> metrics_from_evalb(const std::string& ref_fname, const std::string& hyp_fname, const std::string& evalbout_fname);

#endif
