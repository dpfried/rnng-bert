#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cstring>

#include "word-featurizer.h"

static void DeallocateTensor(void* data, std::size_t, void*) {
  std::free(data);
}

static void DeallocateBuffer(void* data, size_t) {
  std::free(data);
}

static TF_Buffer* ReadBufferFromFile(const char* file) {
  const auto f = std::fopen(file, "rb");
  if (f == nullptr) {
    return nullptr;
  }

  std::fseek(f, 0, SEEK_END);
  const auto fsize = ftell(f);
  std::fseek(f, 0, SEEK_SET);

  if (fsize < 1) {
    std::fclose(f);
    return nullptr;
  }

  const auto data = std::malloc(fsize);
  std::fread(data, fsize, 1, f);
  std::fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;

  return buf;
}

TF_Operation* get_operation(TF_Graph* graph, const char* oper_name) {
  TF_Operation* op = TF_GraphOperationByName(graph, oper_name);
  if (op == nullptr) {
    std::cerr << "Could not find operation in graph: " << oper_name << std::endl;
  }
  assert (op != nullptr);
  return op;
}

TF_Tensor* build_string_tensor(std::string input_str, TF_Status* status) {
  // std::string input_str = "abracdabra";  // any input string
  size_t encoded_size = TF_StringEncodedSize(input_str.size());
  size_t total_size = 8 + encoded_size;  // 8 extra bytes - for start_offset
  char *input_encoded = (char*)malloc(total_size);
  for (int i =0; i < 8; ++i) {  // fills start_offset
      input_encoded[i] = 0;
  }
  TF_StringEncode(input_str.c_str(), input_str.size(), input_encoded+8, encoded_size, status); // fills the rest of tensor data
  if (TF_GetCode(status) != TF_OK){
      std::cerr << "ERROR: something wrong with encoding: " <<TF_Message(status) << std::endl;
  }
  TF_Tensor* res = TF_NewTensor(TF_STRING, NULL, 0, input_encoded, total_size, &DeallocateTensor, 0);
  return res;
}

WordFeaturizer::WordFeaturizer(const char* graph_path,
                 std::string init_checkpoint_path,
                 float learning_rate,
                 int warmup_steps
                ) {
    TF_Buffer* buffer = ReadBufferFromFile(graph_path);
    if (buffer == nullptr) {
      std::cerr << "WordFeaturizer: failed to read tensorflow graph from path: " << graph_path << std::endl;
      assert (buffer != nullptr);
    }

    graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(graph, buffer, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    assert (TF_GetCode(status) == TF_OK);
    std::cout << "WordFeaturizer: loaded graph from " << graph_path << std::endl;

    // ------
    input_ids = get_operation(graph, "input_ids");
    word_end_mask = get_operation(graph, "word_end_mask");
    is_training = get_operation(graph, "is_training");

    word_features = get_operation(graph, "word_features");
    word_features_grad = get_operation(graph, "word_features_grad");

    init_op = get_operation(graph, "init");
    train_op = get_operation(graph, "train");

    restore_op = get_operation(graph, "save/restore_all");
    save_op = get_operation(graph, "save/control_dependency");
    checkpoint_name  = {get_operation(graph, "save/Const"), 0};

    new_learning_rate.oper = get_operation(graph, "new_learning_rate");
    set_learning_rate_op = get_operation(graph, "set_learning_rate");
    new_warmup_steps.oper = get_operation(graph, "new_warmup_steps");
    set_warmup_steps_op = get_operation(graph, "set_warmup_steps");

    feeds_all[0].oper = input_ids;
    feeds_all[1].oper = word_end_mask;
    feeds_all[2].oper = is_training;
    feeds_all[3].oper = word_features_grad;
    fetches_all[0].oper = word_features;
    feeds_fw[0].oper = input_ids;
    feeds_fw[1].oper = word_end_mask;
    feeds_fw[2].oper = is_training;
    fetches_fw[0].oper = word_features;
    feeds_bw[0].oper = word_features_grad;

    std::cout << "WordFeaturizer: found required operations in graph" << std::endl;

    // ------
    TF_SessionOptions* options = TF_NewSessionOptions();
    sess = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);
    assert (TF_GetCode(status) == TF_OK);

    TF_SessionRun(sess,
                nullptr, // Run options.
                nullptr, nullptr, 0, // Input tensors, input tensor values, number of inputs.
                nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                &init_op, 1, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
    assert (TF_GetCode(status) == TF_OK);
    std::cout << "WordFeaturizer: initialized session" << std::endl;

    // ------
    load_checkpoint(init_checkpoint_path);

    // ------
    TF_Tensor* new_learning_rate_tensor = TF_AllocateTensor(TF_FLOAT, NULL, 0, sizeof(float));
    assert (new_learning_rate_tensor != nullptr);
    assert (TF_TensorData(new_learning_rate_tensor) != nullptr);
    *(static_cast<float*>(TF_TensorData(new_learning_rate_tensor))) = learning_rate;
    TF_SessionRun(sess,
                nullptr, // Run options.
                &new_learning_rate, &new_learning_rate_tensor, 1, // Input tensors, input tensor values, number of inputs.
                nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                &set_learning_rate_op, 1, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
    assert (TF_GetCode(status) == TF_OK);
    TF_DeleteTensor(new_learning_rate_tensor);
    std::cout << "WordFeaturizer: set learning rate to " << learning_rate << std::endl;

    TF_Tensor* new_warmup_steps_tensor = TF_AllocateTensor(TF_INT32, NULL, 0, sizeof(int32_t));
    assert (new_warmup_steps_tensor != nullptr);
    assert (TF_TensorData(new_warmup_steps_tensor) != nullptr);
    *(static_cast<int32_t*>(TF_TensorData(new_warmup_steps_tensor))) = warmup_steps;
    TF_SessionRun(sess,
                nullptr, // Run options.
                &new_warmup_steps, &new_warmup_steps_tensor, 1, // Input tensors, input tensor values, number of inputs.
                nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                &set_warmup_steps_op, 1, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
    assert (TF_GetCode(status) == TF_OK);
    TF_DeleteTensor(new_warmup_steps_tensor);
    std::cout << "WordFeaturizer: set warmup steps to " << warmup_steps << std::endl;
  }

void WordFeaturizer::load_checkpoint(std::string checkpoint_path) {
    TF_Tensor* checkpoint_tensor = build_string_tensor(checkpoint_path, status);
    assert (TF_GetCode(status) == TF_OK);
    assert (checkpoint_tensor != nullptr);
    assert (TF_TensorData(checkpoint_tensor) != nullptr);

    TF_SessionRun(sess,
                nullptr, // Run options.
                &checkpoint_name, &checkpoint_tensor, 1, // Input tensors, input tensor values, number of inputs.
                nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                &restore_op, 1, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
    assert (TF_GetCode(status) == TF_OK);
    TF_DeleteTensor(checkpoint_tensor);
    std::cout << "WordFeaturizer: loaded checkpoint " << checkpoint_path << std::endl;
  }

void WordFeaturizer::save_checkpoint(std::string checkpoint_path) {
    TF_Tensor* checkpoint_tensor = build_string_tensor(checkpoint_path, status);
    assert (TF_GetCode(status) == TF_OK);
    assert (checkpoint_tensor != nullptr);
    assert (TF_TensorData(checkpoint_tensor) != nullptr);

    TF_SessionRun(sess,
                nullptr, // Run options.
                &checkpoint_name, &checkpoint_tensor, 1, // Input tensors, input tensor values, number of inputs.
                nullptr, nullptr, 0, // Output tensors, output tensor values, number of outputs.
                &save_op, 1, // Target operations, number of targets.
                nullptr, // Run metadata.
                status // Output status.
                );
    assert (TF_GetCode(status) == TF_OK);
    TF_DeleteTensor(checkpoint_tensor);
    std::cout << "WordFeaturizer: saved checkpoint " << checkpoint_path << std::endl;
  }

void WordFeaturizer::run_fw(int batch_size, int num_subwords,
                    std::vector<int32_t> input_ids_data,
                    std::vector<int32_t> word_end_mask_data,
                    TF_Tensor** features_out, TF_Tensor** features_grad_out
                  ) {
    if (handle != nullptr) {
        std::cerr << "WordFeaturizer: Another call is already in progress: "
                     "use run_bw to complete it, or pass nullptr to features_grad_out "
                     "if not planning to perform a backwards pass" << std::endl;
        assert (handle == nullptr);
    }

    // Assume we need gradients if and only if we're training. The is_training
    // flag toggles the use dropout within the tensorflow computation graph.
    bool will_run_bw = (features_grad_out != nullptr);
    bool is_training = will_run_bw;

    const std::vector<int64_t> dims = {batch_size, num_subwords};
    std::size_t data_size = sizeof(int32_t);
    for (auto i : dims) {
      data_size *= i;
    }

    TF_Tensor* input_ids_tensor = TF_AllocateTensor(TF_INT32,
      dims.data(), static_cast<int>(dims.size()), data_size);
    assert (input_ids_tensor != nullptr);
    assert (TF_TensorData(input_ids_tensor) != nullptr);
    TF_Tensor* word_end_mask_tensor = TF_AllocateTensor(TF_INT32,
      dims.data(), static_cast<int>(dims.size()), data_size);
    assert (word_end_mask_tensor != nullptr);
    assert (TF_TensorData(word_end_mask_tensor) != nullptr);
    TF_Tensor* is_training_tensor = TF_AllocateTensor(TF_BOOL, NULL, 0, sizeof(bool));
    assert (is_training_tensor != nullptr);
    assert (TF_TensorData(is_training_tensor) != nullptr);

    std::memcpy(TF_TensorData(input_ids_tensor), input_ids_data.data(), std::min(data_size, TF_TensorByteSize(input_ids_tensor)));
    std::memcpy(TF_TensorData(word_end_mask_tensor), word_end_mask_data.data(), std::min(data_size, TF_TensorByteSize(word_end_mask_tensor)));
    *(static_cast<bool*>(TF_TensorData(is_training_tensor))) = is_training;

    feed_values_fw[0] = input_ids_tensor;
    feed_values_fw[1] = word_end_mask_tensor;
    feed_values_fw[2] = is_training_tensor;
    TF_Tensor* fetch_values_fw[1] = {nullptr};

    if (will_run_bw) {
      TF_SessionPRunSetup(sess,
          feeds_all, num_feeds_all,
          fetches_all, num_fetches_all,
          &train_op, 1,
          &handle,
          status);
      assert (TF_GetCode(status) == TF_OK);

      TF_SessionPRun(sess, handle,
        feeds_fw, feed_values_fw, num_feeds_fw,
        fetches_fw, fetch_values_fw, num_fetches_fw,
        nullptr, 0,
        status
        );
    } else {
      // Cancelling a partial run partway through does not deallocate memory set
      // aside for the parts that have yet to run, so if we are not planning to
      // do a backwards pass we don't want to set up a partial run.
      TF_SessionRun(sess,
                  nullptr, // Run options.
                  feeds_fw, feed_values_fw, num_feeds_fw,
                  fetches_fw, fetch_values_fw, num_fetches_fw, // Output tensors, output tensor values, number of outputs.
                  nullptr, 0, // Target operations, number of targets.
                  nullptr, // Run metadata.
                  status // Output status.
                  );
    }
    assert (TF_GetCode(status) == TF_OK);
    assert (fetch_values_fw[0] != nullptr);
    assert (TF_TensorData(fetch_values_fw[0]) != nullptr);

    *features_out = fetch_values_fw[0];

    if (!will_run_bw) {
      cleanup();
    } else {
      std::vector<int64_t> grad_accumulator_dims;
      for (int i = 0; i < TF_NumDims(*features_out); i++) {
        grad_accumulator_dims.push_back(TF_Dim(*features_out, i));
      }
      *features_grad_out = TF_AllocateTensor(
        TF_TensorType(*features_out),
        grad_accumulator_dims.data(),
        TF_NumDims(*features_out),
        TF_TensorByteSize(*features_out));
      std::memset(TF_TensorData(*features_grad_out), 0, TF_TensorByteSize(*features_grad_out));
    }
  }

void WordFeaturizer::run_bw(TF_Tensor* features_grad) {
    assert (handle != nullptr);
    TF_Tensor* feed_values_bw[] = {features_grad};
    TF_SessionPRun(sess, handle,
      feeds_bw, feed_values_bw, num_feeds_bw,
      nullptr, nullptr, 0,
      &train_op, 1,
      status
      );
    assert (TF_GetCode(status) == TF_OK);

    cleanup();
  }

void WordFeaturizer::cleanup() {
    if (handle != nullptr) {
      TF_DeletePRunHandle(handle);
      handle = nullptr;
    }

    for (int i = 0; i < num_feeds_fw; i++) {
      if (feed_values_fw[i] != nullptr) {
        TF_DeleteTensor(feed_values_fw[i]);
        feed_values_fw[i] = nullptr;
      }
    }
  }

int WordFeaturizer::batch_inputs(const std::vector<std::vector<int32_t>> &batch_input_ids_data,
                                  const std::vector<std::vector<int32_t>> &batch_word_end_mask_data,
                                  std::vector<int32_t> &input_ids_data,
                                  std::vector<int32_t> &word_end_mask_data) {
  int n_inputs = batch_input_ids_data.size();
  assert(batch_word_end_mask_data.size() == n_inputs);
  assert(input_ids_data.empty());
  assert(word_end_mask_data.empty());

  unsigned long max_len = 0;
  for (unsigned long i = 0; i < batch_input_ids_data.size(); i++) {
    unsigned long l = batch_input_ids_data[i].size();
    assert(l == batch_word_end_mask_data[i].size());
    max_len = std::max(max_len, l);
  }

  int max_words = 0;

  for (unsigned long i = 0; i < batch_input_ids_data.size(); i++) {
    auto& this_iid = batch_input_ids_data[i];
    input_ids_data.insert(input_ids_data.end(), this_iid.begin(), this_iid.end());
    input_ids_data.resize(max_len * (i+1), 0);


    auto& this_wem = batch_word_end_mask_data[i];
    word_end_mask_data.insert(word_end_mask_data.end(), this_wem.begin(), this_wem.end());
    word_end_mask_data.resize(max_len * (i+1), 0);
    int this_words = 0;
    for (int w : this_wem) {
        this_words += w;
    }
    max_words = std::max(max_words, this_words);
  }
  return max_words;
}
