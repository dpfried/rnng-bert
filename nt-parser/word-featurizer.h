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
  TF_Operation *input_ids, *word_end_mask, *is_training;
  TF_Operation *word_features, *word_features_grad;
  TF_Operation *init_op, *accumulate_op, *train_op, *zero_grad_op, *save_op, *restore_op;
  TF_Output checkpoint_name;
  TF_Output new_learning_rate = {nullptr, 0};
  TF_Output new_warmup_steps = {nullptr, 0};
  TF_Operation *set_learning_rate_op, *set_warmup_steps_op;

  const int num_feeds_all = 4;
  const int num_fetches_all = 1;
  const int num_feeds_fw = 3;
  const int num_fetches_fw = 1;
  const int num_feeds_bw = 1;

  TF_Output feeds_all[4] = {{nullptr, 0}, {nullptr, 0}, {nullptr, 0}, {nullptr, 0}};
  TF_Output fetches_all[1] = {{nullptr, 0}};
  TF_Output feeds_fw[3] = {{nullptr, 0}, {nullptr, 0}, {nullptr, 0}};
  TF_Output fetches_fw[1] = {{nullptr, 0}};
  TF_Output feeds_bw[1] = {{nullptr, 0}};
  const char* handle = nullptr;

  TF_Tensor* feed_values_fw[3] = {nullptr, nullptr, nullptr};

public:
  WordFeaturizer(const char* graph_path,
                 std::string init_checkpoint_path,
                 float learning_rate,
                 int warmup_steps
               );
  void load_checkpoint(std::string checkpoint_path);
  void save_checkpoint(std::string checkpoint_path);
  void set_learning_rate(float learning_rate);
  float get_last_set_learning_rate();
  void run_fw(int batch_size, int num_subwords,
                    std::vector<int32_t> input_ids_data,
                    std::vector<int32_t> word_end_mask_data,
                    TF_Tensor** features_out, TF_Tensor** features_grad_out
                  );
  void run_bw(TF_Tensor* features_grad);
  void run_step(void);
  void run_zero_grad(void);
  static int batch_inputs(
          const std::vector<std::vector<int32_t>>& batch_input_ids_data,
          const std::vector<std::vector<int32_t>>& batch_word_end_mask_data,
          std::vector<int32_t>& input_ids_data,
          std::vector<int32_t>& word_end_mask_data
  );
private:
  void cleanup();
  float last_set_learning_rate;
};


#endif
