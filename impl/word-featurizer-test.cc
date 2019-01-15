#include <iostream>
#include <vector>

// TODO: should be <tensorflow/c/c_api.h> instead, but cmake is misconfigured
#include "word-featurizer.h"

int main() {
  WordFeaturizer f(
    "bert_graph.pb",
    "uncased_L-12_H-768_A-12/bert_model.ckpt",
    5e-4f,
    160
  );
  if (false) {
    f.save_checkpoint("expts/dummy_bert_model.ckpt");
  }

  const std::vector<int32_t> input_ids = {101, 2023, 2003, 1037, 3231, 1012, 102};
  const std::vector<int32_t> word_end_mask = {1, 1, 1, 1, 1, 1, 1};
  // Padding is automatically inferred from word_end_mask: to pad, just ensure
  // that all padding tokens are at the end and have a word_end_mask of 0
  const int batch_size = 1;
  const int num_subwords = input_ids.size() / batch_size;
  TF_Tensor* feats = nullptr;
  f.run_fw(batch_size, num_subwords, input_ids, word_end_mask, &feats, nullptr);
  // The value printed should be ~0.210677 (assumes bert-base-uncased model)
  std::cout << static_cast<float*>(TF_TensorData(feats))[1] << std::endl;
  TF_DeleteTensor(feats);

  TF_Tensor* feats_grad = nullptr;
  f.run_fw(batch_size, num_subwords, input_ids, word_end_mask, &feats, &feats_grad);
  f.run_bw(feats_grad);
  TF_DeleteTensor(feats);
  TF_DeleteTensor(feats_grad);

  return 0;
}
