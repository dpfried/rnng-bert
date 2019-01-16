#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>

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

  const unsigned BERT_DIM = 768;

  const std::vector<int32_t> inst_input_ids = {101, 2023, 2003, 1037, 3231, 3424, 10521, 4355, 7875, 13602, 3672, 12199, 2964, 1012, 102};
  const std::vector<int32_t> inst_word_end_mask = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
  // Padding is automatically inferred from word_end_mask: to pad, just ensure
  // that all padding tokens are at the end and have a word_end_mask of 0
  const int batch_size = 8;

  std::vector<std::vector<int32_t>> batch_input_ids;
  std::vector<std::vector<int32_t>> batch_word_end_mask;

  std::vector<int32_t> input_ids;
  std::vector<int32_t> word_end_mask;

  for (unsigned i =0; i < batch_size; i++) {
    batch_input_ids.push_back(inst_input_ids);
    batch_word_end_mask.push_back(inst_word_end_mask);

    // test masking by appending values to later elements of the batch (
    auto& back_ii = batch_input_ids.back();
    back_ii.insert(back_ii.end(), i, 0);

    auto& back_wem = batch_word_end_mask.back();
    back_wem.insert(back_wem.end(), i, 0);
  }

  const unsigned num_words = f.batch_inputs(batch_input_ids, batch_word_end_mask, input_ids, word_end_mask);

  const int num_subwords = input_ids.size() / batch_size;

  TF_Tensor* feats = nullptr;
  f.run_fw(batch_size, num_subwords, input_ids, word_end_mask, &feats, nullptr);
  // batch x words x
  std::cout << TF_Dim(feats, 0) << " " << TF_Dim(feats, 1) << " " << TF_Dim(feats, 2) << std::endl;
  // The value printed should be 0.132475 (assumes bert-base-uncased model)
  std::cout << static_cast<float*>(TF_TensorData(feats))[1] << std::endl;

  assert(TF_Dim(feats, 0) == batch_size);
  assert(TF_Dim(feats, 1) == num_words);
  assert(TF_Dim(feats, 2) == BERT_DIM);

  float* data_arr = static_cast<float*>(TF_TensorData(feats));
  for (volatile unsigned long word_ix = 0; word_ix < TF_Dim(feats, 1); word_ix++) {
    for (volatile unsigned long value_ix = 0; value_ix < TF_Dim(feats, 2); value_ix++) {
      volatile float v0 = data_arr[word_ix * TF_Dim(feats, 2)
                          + value_ix];
      for (volatile unsigned long batch_ix = 0; batch_ix < batch_size; batch_ix++) {
        volatile float vb = data_arr[batch_ix * TF_Dim(feats, 1) * TF_Dim(feats, 2)
                                     + word_ix * TF_Dim(feats, 2)
                                     + value_ix];
        assert(std::abs(v0 - vb) < 1e-3);
      }
    }
  }
  TF_DeleteTensor(feats);

  TF_Tensor* feats_grad = nullptr;
  f.run_fw(batch_size, num_subwords, input_ids, word_end_mask, &feats, &feats_grad);
  std::cout << "TF_TensorType(feats_grad) " << TF_TensorType(feats_grad) << std::endl;
  f.run_bw(feats_grad);
  TF_DeleteTensor(feats);
  TF_DeleteTensor(feats_grad);

  return 0;
}
