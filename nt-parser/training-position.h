//
// Created by dfried on 1/21/18.
//

#ifndef CNN_TRAINER_H
#define CNN_TRAINER_H

namespace parser {
  struct TrainingPosition {
    int epoch = 0;
    int sentence = 0;
    bool in_silver_block = false;

    // number of decodes
    int iter = -1;


    // total number of sentences seen
    int tot_seen = 0;

    double best_dev_f1 = 0.0;
    double best_dev_error = 9e99;

    int tot_seen_last_status = 0;

    int batches = 0;

  private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int) {
        ar & epoch;
        ar & sentence;
        ar & in_silver_block;
        ar & iter;
        ar & tot_seen;
        ar & best_dev_f1;
        ar & best_dev_error;
        ar & tot_seen_last_status;
        ar & batches;
    }
  };
}

#endif //CNN_TRAINER_H
