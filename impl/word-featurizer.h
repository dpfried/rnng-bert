#ifndef WORD_FEATURIZER_H
#define WORD_FEATURIZER_H

#include <vector>

// TODO: should be <tensorflow/c/c_api.h> instead, but cmake is misconfigured
#include <c_api.h>

class WordFeaturizer {
private:
  TF_Graph* graph;
  TF_Status* status;
  TF_Session* sess;
  TF_Operation *input_ids, *word_end_mask, *word_features, *word_features_grad;
  TF_Operation *init_op, *train_op, *save_op, *restore_op;
  TF_Output checkpoint_name;

  const int num_feeds_all = 3;
  const int num_fetches_all = 1;
  const int num_feeds_fw = 2;
  const int num_fetches_fw = 1;
  const int num_feeds_bw = 1;

  TF_Output feeds_all[3] = {{nullptr, 0}, {nullptr, 0}, {nullptr, 0}};
  TF_Output fetches_all[1] = {{nullptr, 0}};
  TF_Output feeds_fw[2] = {{nullptr, 0}, {nullptr, 0}};
  TF_Output fetches_fw[1] = {{nullptr, 0}};
  TF_Output feeds_bw[1] = {{nullptr, 0}};
  const char* handle = nullptr;

  TF_Tensor* feed_values_fw[2] = {nullptr, nullptr};

public:
  WordFeaturizer(const char* graph_path,
                 std::string init_checkpoint_path,
                 float learning_rate,
                 int warmup_steps
               );
  void load_checkpoint(std::string checkpoint_path);
  void save_checkpoint(std::string checkpoint_path);
  void run_fw(int batch_size, int num_subwords,
                    std::vector<int32_t> input_ids_data,
                    std::vector<int32_t> word_end_mask_data,
                    TF_Tensor** features_out, TF_Tensor** features_grad_out
                  );
  void run_bw(TF_Tensor* features_grad);
private:
  void cleanup();
};

#endif
