//
// Created by dfried on 1/11/18.
//

#ifndef CNN_STREAMING_STATISTICS_H
#define CNN_STREAMING_STATISTICS_H

#include <cmath>

struct StreamingStatistics {

  double total = 0.0;
  double total_standardized = 0.0;
  double m2 = 0.0;
  double mean = 0.0;
  double std = 0.0;
  unsigned num_samples = 0;

  double standardize_and_update(double value) {
      // Welford online variance, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      num_samples++;
      total += value;
      double delta = value - mean;
      mean += delta / num_samples;
      m2 += delta * (value - mean);
      std = sqrt(num_samples > 1 ? m2 / (num_samples - 1) : 1.0);
      double standardized = std > 0 ? (value - mean) / std : 0;
      total_standardized += standardized;
      return standardized;
  }

  double mean_value() {
      return mean;
  }

  double mean_standardized_value() {
      return total_standardized / num_samples;

  }

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
      ar & total;
      ar & total_standardized;
      ar & m2;
      ar & mean;
      ar & std;
      ar & num_samples;
  }
};

#endif //CNN_STREAMING_STATISTICS_H
