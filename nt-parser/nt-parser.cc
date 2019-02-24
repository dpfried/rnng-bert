#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <functional>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/exception/all.hpp>
#include <boost/throw_exception.hpp>

#include "cnn/init.h"
#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "nt-parser/nt-parser.h"
#include "nt-parser/oracle.h"
#include "nt-parser/pretrained.h"
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/eval.h"
#include "nt-parser/stack.h"
#include "nt-parser/tree.h"
#include "nt-parser/training-position.h"
#include "nt-parser/streaming-statistics.h"
#include "nt-parser/utils.h"
#include "nt-parser/word-featurizer.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict, non_unked_termdict;
cnn::Dict morphology_classes;
std::unordered_map<std::string, cnn::Dict> morphology_dicts;
std::unordered_map<std::string, std::vector<bool>> morphology_singletons;

const unsigned START_OF_SENTENCE_ACTION = std::numeric_limits<unsigned>::max();

const string UNK = "UNK";

volatile bool requested_stop = false;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;
unsigned MORPH_DIM = 10;
unsigned BERT_DIM = 0; // should be initialized later

unsigned SHIFT_ACTION = UINT_MAX;
unsigned REDUCE_ACTION = UINT_MAX;
unsigned TERM_ACTION = UINT_MAX; // only used for in-order

int MAX_SENTENCE_LENGTH_EVAL = -1;
int MAX_SENTENCE_LENGTH_TRAIN = -1;

float ALPHA = 1.f;
float DYNAMIC_EXPLORATION_PROBABILITY = 1.f;
int N_SAMPLES = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
unsigned POS_SIZE = 0;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X
std::map<int,int> ntIndex2Action;  // pass in index of X, return index of action NT(X)

bool USE_POS = false;  // in discriminative parser, incorporate POS information in token embedding
bool USE_MORPH_FEATURES = false;
bool USE_PRETRAINED = false;  // in discriminative parser, use pretrained word embeddings (not updated)
bool NO_STACK = false;
bool NO_ACTION_HISTORY = false;
int SILVER_BLOCKS_PER_GOLD = 10;

bool UNNORMALIZED = false;

bool BRACKET_CONSTRAINED_TRAINING = false;

bool IN_ORDER = false;

unsigned MAX_UNARY = 3;

unsigned MAX_CONS_NT = 8;

bool COLLAPSE_UNARY = false;
bool REVERSE_TREES = false;

bool BERT = false;
bool SELF_ATTEND_WORDS = false;

bool BERT_BUFFER_LSTM = false;
bool BERT_BUFFER_USE_CLS = false;
bool BERT_BUFFER_LSTM_USE_CLS_SEP = false;
bool BERT_LARGE = false;

float BERT_LR = 5e-5f;
int BERT_WARMUP_STEPS = 160;

float BERT_FEATURE_DOWNSCALE = 1.0;


string BERT_MODEL_PATH = ""; // will be initialized to one of the following if BERT is passed
const string BERT_BASE_MODEL_PATH = "bert_models/uncased_L-12_H-768_A-12";
const string BERT_LARGE_MODEL_PATH = "bert_models/uncased_L-24_H-1024_A-16";

string BERT_GRAPH_PATH = ""; // will be initialized to one of the following if BERT is passed
const string BERT_BASE_GRAPH_PATH = BERT_BASE_MODEL_PATH + "_graph.pb";
const string BERT_LARGE_GRAPH_PATH = BERT_LARGE_MODEL_PATH + "_graph.pb";

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training

// all of these except nonterm_actions should contain only one action, but use collections for uniformity
vector<unsigned> shift_actions;
vector<unsigned> reduce_actions;
vector<unsigned> terminate_actions;
vector<unsigned> nonterm_actions;

bool USES_BUFFER_LSTM() {
  return !BERT || BERT_BUFFER_LSTM;
}

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
          // run parameters
          ("model_dir,m", po::value<string>(), "Load saved model from this directory")
          ("text_format", "serialize models in text")
          ("git_state", "print git revision and diff to stderr")

          ("unnormalized", "do not locally normalize score distributions")
          ("bracket_constrained", "uses gold bracketing structure to constrain possible action probability distribution")

          ("spmrl", "Use the SPMRL variant of EVALB")
          ("inorder", "super experimental implementation of Liu and Zhang 2017, breaks many of the other flags")

          ("collapse_unary", "assume that oracle files represent parses with collapsed unary chains. Don't allow producing unary chains, and uncollapse unaries when decoding (split on +)")
          ("max_unary", po::value<unsigned>()->default_value(MAX_UNARY), "maximum size (-1) of a unary chain")
          ("max_cons_nts", po::value<unsigned>(), "maximum number of NTs that can be opened consecutively")
          ("reverse_trees", "assume that oracle files represent travels of mirrored trees (i.e. right-to-left). Mirror the trees when decoding")

          // data
          ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
          ("dev_data,d", po::value<string>(), "Development corpus")
          ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")
          ("gold_training_data", po::value<string>(), "List of Transitions - smaller corpus (e.g. wsj in a wsj+silver experiment)")
          ("test_data,p", po::value<string>(), "Test corpus")
          ("bracketing_test_data", po::value<string>(), "Test bracketed corpus")
          ("max_sentence_length_train", po::value<int>()->default_value(MAX_SENTENCE_LENGTH_TRAIN), "Don't train on sentences longer than this length")
          ("max_sentence_length_eval", po::value<int>()->default_value(MAX_SENTENCE_LENGTH_EVAL), "Don'evaluate on sentences longer than this length")

          ("silver_blocks_per_gold", po::value<int>()->default_value(SILVER_BLOCKS_PER_GOLD), "How many same-sized blocks of the silver data should be sampled and trained, between every train on the entire gold set?")

          // model parameters
          ("use_pos_tags,P", "make POS tags visible to parser")
          ("use_morph_features", "make morphological features visible to parser")
          ("words,w", po::value<string>(), "Pretrained word embeddings")

          ("bert", "use BERT to represent inputs")
          ("bert_large", "use BERT-Large (otherwise use BERT-Base)")
          ("bert_feature_downscale", po::value<float>(), "scale down BERT features by this much")

          ("self_attend_words", "use self-attention over words, using a BERT graph")

          ("bert_model_dir", po::value<string>(), "path to directory containing the BERT model (overrides defaults: bert_models/uncased_L-12_H-768_A-12 or bert_models/uncased_L-24_H-1024_A-16 for BERT-large")
          ("bert_graph_path", po::value<string>(), "path to *_graph.pb file containing the BERT graph (overrides defaults: bert_models/uncased_L-12_H-768_A-12_graph.pb or bert_models/uncased_L-24_H-1024_A-16_graph.pb for BERT-large")

          ("bert_buffer_lstm", "run the regular buffer lstm over the BERT embeddings")
          ("bert_buffer_use_cls", "use the BERT CLS embedding as the buffer guard")
          ("bert_buffer_lstm_use_cls_sep", "use both CLS and SEP in the BERT lstm buffer")

          ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
          ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
          ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
          ("morph_dim", po::value<unsigned>()->default_value(10), "morph features dimension (if use_morph_features is set)")
          ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
          ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
          ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
          ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")

          ("no_stack,S", "Don't encode the stack")
          ("no_action_history", "Don't encode the action history")

          // training
          ("train,t", "Should training be run?")
          ("dropout,D", po::value<float>(), "Dropout rate")
          ("model_output_dir", po::value<string>(), "Override auto-generate name of prefix to save model")
          // ("set_iter", po::value<int>(),  "")
          ("save_frequency_minutes", po::value<int>()->default_value(30),  "save model roughly every this many minutes (if it hasn't been saved by a dev decode)")
          ("dev_check_frequency", po::value<int>()->default_value(9958),  "evaluate on the dev set every this many sentences (9958 = 4 times per epoch on English)")

          ("optimizer", po::value<string>()->default_value("sgd"), "sgd | adam")
          ("sgd_e0", po::value<float>()->default_value(0.1f),  "initial step size for gradient descent")
          ("sgd_eta_decay", po::value<float>()->default_value(0.05),  "learning rate decay for gradient descent (learning rate = sgd_e0 / (1 + epoch * sgd_eta_decay)) ")

          ("adam_lr", po::value<float>()->default_value(0.001f), "initial learning rate for Adam for dynet components")

          ("lr_decay_patience", po::value<int>()->default_value(-1), "if non-negative, if dev accuracy doesn't improve in this many epochs, decrease the lr by lr_decay_factor (for both dynet and BERT)")
          ("lr_decay_factor", po::value<float>()->default_value(0.5), "used with lr_decay_patience")

          ("batch_size", po::value<int>()->default_value(1),  "number of training examples to use to compute each gradient update")
          ("eval_batch_size", po::value<int>()->default_value(8),  "number of examples to process in parallel for evaluation")
          ("subbatch_max_tokens", po::value<int>()->default_value(9999),  "maximum number of sub-word units to process in parallel while training")

          ("bert_lr", po::value<float>()->default_value(BERT_LR), "initial BERT learning rate (after warmup)")
          ("bert_warmup_steps", po::value<int>()->default_value(BERT_WARMUP_STEPS), "number of steps in BERT warmup period")

          ("min_risk_training", "min risk training (default F1)")
          ("min_risk_method", po::value<string>()->default_value("reinforce"), "reinforce | beam | beam_unnormalized")
          ("min_risk_include_gold", "use the true parse in the gradient updates")
          ("min_risk_candidates", po::value<int>()->default_value(10), "min risk number of candidates")

          ("label_smoothing_epsilon", po::value<float>()->default_value(0.0f), "use epsilon interpolation with the uniform distribution in label smoothing")
          ("entropy_regularization_weight", po::value<float>()->default_value(0.0f))

          ("max_margin_training", "")
          ("softmax_margin_training", "")

          ("dynamic_exploration_include_gold", "use the true parse in the gradient updates")
          ("dynamic_exploration_candidates", po::value<int>()->default_value(1))
          ("dynamic_exploration", po::value<string>(), "if passed, should be greedy | sample")
          ("dynamic_exploration_probability", po::value<float>()->default_value(1.0), "with this probability, use the model probabilities to explore (with method given by --dynamic_exploration)")

          ("compute_distribution_stats", "compute entropy and gold probabilities for action distributions")

          // inference
          ("samples,s", po::value<int>(), "Sample N trees for each test sentence instead of greedy max decoding")
          ("output_beam_as_samples", "Print the items in the beam in the same format as samples")
          ("samples_include_gold", "Also include the gold parse in the list of samples output")
          ("bracket_constrained_test_decode", "uses gold bracketing structure to constrain possible actions taken in decode (note: this doesn't affect the possible actions for computing probability distribution supports, and can be set even if --bracket_constrained is unset)")

          ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
          ("max_cons_nt", po::value<unsigned>()->default_value(8), "maximum number of non-terminals that can be opened consecutively")
          ("beam_size,b", po::value<int>()->default_value(1), "beam size")
          ("beam_within_word", "greedy decode within word")
          ("beam_filter_at_word_size", po::value<int>()->default_value(-1), "when using beam_within_word, filter word completions to this size (defaults to decode_beam_size if < 0)")
          ("factored_ensemble_beam", "do beam search in each model in the ensemble separately, then take the union and rescore with the entire ensemble")

          ("ptb_output_file", po::value<string>(), "When outputting parses, use original POS tags and non-unk'ed words")
          ("eval_files_prefix", po::value<string>(), "When testing, output parses and output files to this prefix")

          ("block_count", po::value<int>()->default_value(0), "divide the dev set up into this many blocks and only decode one of them (indexed by block_num)")
          ("block_num", po::value<int>()->default_value(0), "decode only this block (0-indexed), must be used with block_count")
          ("decode_start_index", po::value<int>()->default_value(0), "start here")

          ("interactive", "Run in interactive mode")

          // ensemble inference
          ("models", po::value<vector<string>>()->multitoken(), "Load ensemble of saved models from these files")
          ("combine_type", po::value<string>(), "Decision-level combination type for ensemble (sum or product)")

        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

vector<unsigned>& actions_of_same_structural_type(unsigned action) {
  if (IN_ORDER && action == TERM_ACTION) return terminate_actions;
  if (action == SHIFT_ACTION) return shift_actions;
  if (action == REDUCE_ACTION) return reduce_actions;
  return nonterm_actions;
}

vector<float> action_mask(const vector<unsigned>& valid_actions) {
  vector<float> mask(adict.size(), -numeric_limits<float>::infinity());
  for (auto a: valid_actions) {
    mask[a] = 0;
  }
  return mask;
}

Expression log_softmax_constrained(const Expression& logits, const vector<unsigned>& valid_actions) {
  Expression mask = input(*logits.pg, Dim({adict.size()}), action_mask(valid_actions));
  return log_softmax(logits + mask);
  //return log_softmax(logits, valid_actions);
  //return log(softmax(logits));
}

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Discriminative(
        unsigned action, char prev_a, unsigned bsize, unsigned ssize,
        unsigned nopen_parens, unsigned ncons_nt, unsigned unary,
        const Stack<Bracket>& completed_brackets, const Stack<OpenBracket>& open_brackets,
        unsigned words_shifted
) {
  bool is_shift = action == SHIFT_ACTION;
  bool is_reduce = action == REDUCE_ACTION;
  bool is_term = IN_ORDER ? action == TERM_ACTION : false;
  bool is_nt = !(is_shift | is_reduce | is_term);

  if (IN_ORDER) {
    assert(is_shift || is_reduce || is_nt || is_term) ;
    // TODO(dfried): use this
    static const unsigned MAX_OPEN_NTS = 100;
//  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
    if (is_term){
      if(ssize == 2 && bsize == 1 && prev_a == 'R') return false;
      return true;
    }

    if(ssize == 1){
      if(!is_shift) return true;
      return false;
    }

    if (is_shift){
      if(bsize == 1) return true;
      if(nopen_parens == 0) return true;
      return false;
    }

    if (is_nt) {
      if(bsize == 1 && unary >= MAX_UNARY) return true;
      if(prev_a == 'N') return true;
      return false;
    }

    if (is_reduce){
      if (COLLAPSE_UNARY) {
        // TODO(dfried): in theory we'd be able to check this using open and completed brackets,
        //  in the same way as for top-down, but for in-order these aren't currently
        //  initialized with the left end-point of the span but the split point
        if (unary > 1) {
          cerr << "error: should never have a current unary_count > 1 with COLLAPSE_UNARY (currently " << unary << ")" << endl;
          assert(unary <= 1);
        }
        if (unary == 1 && prev_a == 'N' && bsize != 1) return true;
      }
      if(prev_a == 'N' && unary > MAX_UNARY) return true;
      if(nopen_parens == 0) return true;
      return false;
    }
    //assert(false); // should never reach here
    cerr << "Logic error for in-order action check" << endl;
    abort();

  } else { // standard top-down RNNG
    assert(is_shift || is_reduce || is_nt) ;
    static const unsigned MAX_OPEN_NTS = 100;

    // TODO: check that these are sufficient for other languages

    if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
    if (is_nt && ncons_nt >= MAX_CONS_NT) return true;
    if (ssize == 1) {
      if (!is_nt) return true;
      return false;
    }

    if (COLLAPSE_UNARY && is_reduce) {
      if (!completed_brackets.empty()
          && get<1>(completed_brackets.back()) == get<1>(open_brackets.back()) // this open bracket has the same beginning as the last one completed (which must be contained in it)
          && get<2>(completed_brackets.back()) == words_shifted // and we haven't shifted any additional words, so this would be a unary chain
          ) {
        // TODO: remove this
        //return true;
        // only allow a reduce if there are no words remaining to shift
        return bsize != 1;
        // TODO(dfried): more general version of this that constrains all actions
        //  based on number of brackets opening at the same span beginning and number of words shifted
      }
    }

    // be careful with top-level parens- you can only close them if you
    // have fully processed the buffer
    if (nopen_parens == 1 && bsize > 1) {
      if (is_reduce) return true;
    }

    // you can't reduce after an NT action
    if (is_reduce && prev_a == 'N') return true;
    if (is_nt && bsize == 1) return true;
    if (is_shift && bsize == 1) return true;
    if (is_reduce && ssize < 3) return true;

    // TODO should we control the depth of the parse in some way? i.e., as long as there
    // are items in the buffer, we can do an NT operation, which could cause trouble
    return false;
  }
}

void print_tree(const Tree& tree, const parser::Sentence& sentence, bool output_tags, ostream& out_stream, bool suppress_newline=false) {
  for (auto& tok: linearize_tree(tree, sentence, output_tags, true)) {
    out_stream << tok << " ";
  }
  if (!suppress_newline)
    out_stream << endl;
}

void print_parse(const vector<unsigned>& actions, const parser::Sentence& sentence, bool ptb_output_format, ostream& out_stream, bool suppress_newline = false) {
  Tree tree = to_tree_u(actions, sentence);
  print_tree(tree, sentence, ptb_output_format, out_stream, suppress_newline);
  /*
  if (IN_ORDER) {
    Tree tree = to_tree_u(actions, sentence);
    print_tree(tree, sentence, ptb_output_format, out_stream);
  } else {
    int ti = 0;
    for (auto a : actions) {
      if (adict.Convert(a)[0] == 'N') {
        out_stream << " (" << ntermdict.Convert(action2NTindex.find(a)->second);
      } else if (adict.Convert(a)[0] == 'S') {
        if (ptb_output_format) {
          string preterminal = posdict.Convert(sentence.pos[ti]);
          out_stream << " (" << preterminal << ' ' << non_unked_termdict.Convert(sentence.non_unked_raw[ti])
                     << ")";
          ti++;
        } else { // use this branch to surpress preterminals
          out_stream << ' ' << termdict.Convert(sentence.raw[ti++]);
        }
      } else out_stream << ')';
    }
    out_stream << endl;
  }
   */
}

void print_parse(const vector<int>& actions, const parser::Sentence& sentence, bool ptb_output_format, ostream& out_stream, bool suppress_newline = false) {
  for (auto action : actions)
    assert(action >= 0);
  print_parse(vector<unsigned>(actions.begin(), actions.end()), sentence, ptb_output_format, out_stream, suppress_newline);
}

Expression logsumexp_stable(const vector<Expression>& all_log_probs) {
  // Expression cwise_max = max(all_log_probs); // gives an error despite being correct
  Expression cwise_max = all_log_probs.front();
  for (auto it = all_log_probs.begin() + 1; it != all_log_probs.end(); ++it)
    cwise_max = max(cwise_max, *it);
  vector <Expression> exp_log_probs;
  for (const Expression& log_probs : all_log_probs)
    exp_log_probs.push_back(exp(log_probs - cwise_max));
  return log(sum(exp_log_probs)) + cwise_max;
}

struct AbstractParser {
  virtual std::shared_ptr<AbstractParserState> new_sentence(
          ComputationGraph* hg,
          const parser::Sentence& sent,
          const vector<shared_ptr<Parameters>>& bert_embeddings,
          bool is_evaluation,
          bool build_training_graph,
          bool apply_dropout
  ) = 0;

  pair<vector<unsigned>, vector<unsigned>> get_support_and_can_take(const vector<unsigned>& valid_actions, unsigned action_count, const vector<int>* bracket_constraint_actions) {
    vector<unsigned> actions_in_support = valid_actions;
    vector<unsigned> actions_can_take = actions_in_support;
    if (bracket_constraint_actions) {
      if (action_count >= bracket_constraint_actions->size()) {
        cerr << "not enough bracket constraint actions for current action_count" << endl;
        abort();
      }
      actions_can_take = actions_of_same_structural_type((*bracket_constraint_actions)[action_count]);
      if (BRACKET_CONSTRAINED_TRAINING) {
        if (!bracket_constraint_actions) {
          cerr << "must pass bracket_constraint_actions to all model invocations when bracket_constrained_training is set" << endl;
        }
        actions_in_support = actions_can_take;
      }
    }
    if (actions_in_support.empty()) {
      cerr << "no actions in support from current parser state" << endl;
      abort();
    }
    assert(!actions_in_support.empty());

    if (actions_can_take.empty()) {
      cerr << "no actions can be taken from current parser state" << endl;
      abort();
    }
    assert(!actions_can_take.empty());
    return make_pair(actions_in_support, actions_can_take);
  }

  pair<vector<unsigned>, Expression> abstract_log_prob_parser(
      ComputationGraph* hg,
      const parser::Sentence& sent,
      const vector<shared_ptr<Parameters>>& bert_embeddings,
      const vector<int>& correct_actions,
      double *right,
      bool is_evaluation,
      bool sample = false,
      bool label_smoothing = false,
      float label_smoothing_epsilon = 0.0,
      DynamicOracle* dynamic_oracle = nullptr,
      bool loss_augmented = false,
      bool softmax_margin = false,
      StreamingStatistics* streaming_entropy = nullptr,
      StreamingStatistics* streaming_gold_prob = nullptr,
      vector<Expression>* entropy_accumulator = nullptr,
      const vector<int>* bracket_constraint_actions = nullptr
  ) {
    // can't have both correct actions and an oracle
    assert(correct_actions.empty() || !dynamic_oracle);
    bool build_training_graph = !correct_actions.empty() || dynamic_oracle;
    bool apply_dropout = (DROPOUT && !is_evaluation);

    if (label_smoothing) {
      assert(build_training_graph);
    }

    if (loss_augmented || softmax_margin) {
      assert(build_training_graph);
    }

    std::shared_ptr<AbstractParserState> state = new_sentence(hg, sent, bert_embeddings, is_evaluation, build_training_graph, apply_dropout);

    unsigned action_count = 0;  // incremented at each prediction

    vector<Expression> scores;
    vector<unsigned> results;

    while(!state->is_finished()) {
      vector<unsigned> actions_in_support;
      vector<unsigned> actions_can_take;
      if (state->get_valid_actions().empty()) {
        cerr << "no valid actions:" << endl;
        cerr << "unary: " << state->get_unary() << endl;
        for (unsigned raw: sent.non_unked_raw) {
          cerr << non_unked_termdict.Convert(raw) << " ";
        }
        cerr << endl;
        for (unsigned ix: results) {
          cerr << adict.Convert(ix) << " ";
        }
        cerr << endl;
      }
      std::tie(actions_in_support, actions_can_take) = get_support_and_can_take(
              state->get_valid_actions(), action_count, bracket_constraint_actions
      );

      unsigned correct_action;
      if (build_training_graph) {
        if (!correct_actions.empty()) {
          if (action_count >= correct_actions.size()) {
            cerr << "Correct action list exhausted, but not in final parser state.\n";
            abort();
          }
          correct_action = correct_actions[action_count];
        } else {
          correct_action = dynamic_oracle->oracle_action(*state);
        }
      }
      Expression adiste; // used for exploration
      Expression adiste_augmented; // used for loss

      if (loss_augmented || softmax_margin) {
        vector<float> aug(possible_actions.size(), 1.0f);
        aug[correct_action] = 0.0f;
        std::tie(adiste, adiste_augmented) = state->get_action_log_probs(actions_in_support, &aug);
        //adiste = adiste + input(*hg, Dim({aug.size()}), aug);
      } else {
        std::tie(adiste, adiste_augmented) = state->get_action_log_probs(actions_in_support, nullptr);
      }

      vector<float> adist = as_vector(adiste.value());

      if (build_training_graph) {
        assert(std::find(actions_in_support.begin(), actions_in_support.end(), correct_action) != actions_in_support.end());
        if (streaming_gold_prob) {
          Expression log_gold_prob = pick(adiste, correct_action);
          streaming_gold_prob->standardize_and_update(exp(as_scalar(log_gold_prob.value())));
        }
      }

      if (streaming_entropy) {
        double entropy = 0;
        // doing this with a dot product produces nans due to invalid actions
        for (unsigned ac: actions_in_support) {
          double log_p = adist[ac];
          entropy -= log_p * exp(log_p);
        }
        streaming_entropy->standardize_and_update(entropy);
      }

      if (entropy_accumulator) {
        Expression entropy_e = input(*hg, 0.0);
        for (unsigned ac: actions_in_support) {
          Expression log_prob = pick(adiste, ac);
          entropy_e = entropy_e - log_prob * exp(log_prob);
        }
        entropy_accumulator->push_back(entropy_e);
      }

      unsigned model_action = actions_can_take[0];
      if (sample) {
        double p = rand01();
        assert(actions_can_take.size() > 0);
        vector<float> dist_to_sample;
        if (ALPHA != 1.0f) {
          // Expression r_t_smoothed = r_t * ALPHA;
          // Expression adiste_smoothed = log_softmax(r_t_smoothed, current_valid_actions);
          //Expression adiste_smoothed = log_softmax(adiste * ALPHA, valid_actions);
          Expression adiste_smoothed = log_softmax_constrained(adiste * ALPHA, actions_can_take);
          dist_to_sample = as_vector(hg->incremental_forward());
        } else {
          dist_to_sample = adist;
        }
        unsigned w = 0;
        for (; w < actions_can_take.size(); ++w) {
          p -= exp(dist_to_sample[actions_can_take[w]]);
          if (p < 0.0) break;
        }
        if (w == actions_can_take.size()) w--;
        model_action = actions_can_take[w];
      } else { // max
        double best_score = adist[actions_can_take[0]];
        for (int i = 1; i < actions_can_take.size(); ++i) {
          if (adist[actions_can_take[i]] > best_score) {
            best_score = adist[actions_can_take[i]];
            model_action = actions_can_take[i];
          }
        }
      }

      unsigned action_taken;
      if (loss_augmented) {
        if (model_action == correct_action) {
          (*right)++;
        } else {
          scores.push_back(pick(adiste_augmented, correct_action) - pick(adiste_augmented, model_action));
        }

        if (dynamic_oracle && rand01() < DYNAMIC_EXPLORATION_PROBABILITY) {
          action_taken = model_action;
        } else {
          action_taken = correct_action;
        }
      } else if (build_training_graph) {  // if we have reference actions or an oracle (for training)
        if (model_action == correct_action) { (*right)++; }
        if (label_smoothing) {
          assert(!UNNORMALIZED);
          Expression cross_entropy = pick(adiste_augmented, correct_action) * input(*hg, (1 - label_smoothing_epsilon));
          // add uniform cross entropy
          for (unsigned a: actions_in_support) {
            cross_entropy = cross_entropy + pick(adiste_augmented, a) * input(*hg, label_smoothing_epsilon / actions_in_support.size());
          }
          scores.push_back(cross_entropy);
        } else {
          scores.push_back(pick(adiste_augmented, correct_action));
        }
        if (dynamic_oracle && rand01() < DYNAMIC_EXPLORATION_PROBABILITY) {
          action_taken = model_action;
        } else {
          action_taken = correct_action;
        }
      } else {
        // adiste should be the same as adiste_augmented in this case
        scores.push_back(pick(adiste_augmented, model_action));
        action_taken = model_action;
      }

      ++action_count;
      results.push_back(action_taken);
      state = state->perform_action(action_taken);
      //cerr << action_count << "\t" << as_scalar(log_probs.back().value()) << "\t";
      //print_parse(results, sent, false, cerr);
    }

    if (!correct_actions.empty() && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }

    state->finish_sentence();

    Expression tot_loss = scores.empty() ? input(*hg, 0.0): -sum(scores);
    assert(tot_loss.pg != nullptr);
    return pair<vector<unsigned>, Expression>(results, tot_loss);
  }

  virtual vector<pair<vector<unsigned>, Expression>> abstract_log_prob_parser_beam(
      ComputationGraph* hg,
      const parser::Sentence& sent,
      const vector<shared_ptr<Parameters>>& bert_embeddings,
      int beam_size,
      bool is_evaluation = true,
      const vector<int>* bracket_constraint_actions = nullptr
  ) {
    //ComputationGraph hg;
    struct BeamItem {
      explicit BeamItem(std::shared_ptr<AbstractParserState> state, unsigned last_action, Expression score) :
              state(state), last_action(last_action), score(score)  {}
      std::shared_ptr<AbstractParserState> state;
      unsigned last_action;
      Expression score;
    };

    bool build_training_graph = !is_evaluation;
    bool apply_dropout = (DROPOUT && !is_evaluation);

    std::shared_ptr<AbstractParserState> initial_state = new_sentence(hg, sent, bert_embeddings, is_evaluation, build_training_graph, apply_dropout);

    vector<Expression> log_probs;
    vector<unsigned> results;

    vector<Stack<BeamItem>> completed;
    vector<Stack<BeamItem>> beam;

    beam.push_back(Stack<BeamItem>(BeamItem(initial_state, START_OF_SENTENCE_ACTION, input(*hg, 0.0))));

    int action_count = 0;
    while (completed.size() < beam_size && !beam.empty()) {
      // old beam item, action to be applied, resulting total score
      vector<std::tuple<Stack<BeamItem>, unsigned, Expression>> successors;

      while (!beam.empty()) {
        const Stack<BeamItem> current_stack_item = beam.back();
        beam.pop_back();

        std::shared_ptr<AbstractParserState> current_parser_state = current_stack_item.back().state;

        vector<unsigned> actions_in_support;
        vector<unsigned> actions_can_take;
        std::tie(actions_in_support, actions_can_take) = get_support_and_can_take(
                current_parser_state->get_valid_actions(), action_count, bracket_constraint_actions
        );

        Expression adiste = current_parser_state->get_action_log_probs(actions_in_support, nullptr).first;
        vector<float> adist = as_vector(hg->incremental_forward());

        for (unsigned action: actions_can_take) {
          Expression action_score = pick(adiste, action);
          Expression total_score = current_stack_item.back().score + action_score;
          successors.push_back(
                  std::tuple<Stack<BeamItem>, unsigned, Expression>(current_stack_item, action, total_score)
          );
        }
      }

      action_count += 1;

      int num_pruned_successors = std::min(beam_size, static_cast<int>(successors.size()));
      partial_sort(successors.begin(),
                   successors.begin() + num_pruned_successors,
                   successors.end(),
                   [](const std::tuple<Stack<BeamItem>, unsigned, Expression>& t1, const std::tuple<Stack<BeamItem>, unsigned, Expression>& t2) {
                     return as_scalar(get<2>(t1).value()) > as_scalar(get<2>(t2).value()); // sort in descending order by total score
                   });
      while (successors.size() > num_pruned_successors)
        successors.pop_back();

      for (auto& successor : successors) {
        Stack<BeamItem> current_stack_item = get<0>(successor);
        std::shared_ptr<AbstractParserState> current_parser_state = current_stack_item.back().state;
        unsigned action = get<1>(successor);
        Expression total_score = get<2>(successor);
        std::shared_ptr<AbstractParserState> successor_parser_state = current_parser_state->perform_action(action);
        Stack<BeamItem> successor_stack_item = current_stack_item.push_back(
                BeamItem(successor_parser_state,
                         action,
                         total_score)
        );
        if (successor_parser_state->is_finished())
          completed.push_back(successor_stack_item);
        else
          beam.push_back(successor_stack_item);
      }
    }

    sort(completed.begin(), completed.end(), [](const Stack<BeamItem>& t1, const Stack<BeamItem>& t2) {
                     return as_scalar(t1.back().score.value()) > as_scalar(t2.back().score.value()); // sort in descending order by total score
                   });

    vector<pair<vector<unsigned>, Expression>> completed_actions_and_nlp;
    for (const auto & completed_stack_item: completed) {
      completed_stack_item.back().state->finish_sentence();

      Stack<BeamItem> stack_item = completed_stack_item;
      Expression nlp = -1 *  stack_item.back().score;
      vector<unsigned> actions;
      while (stack_item.back().last_action != START_OF_SENTENCE_ACTION) {
        actions.push_back(stack_item.back().last_action);
        stack_item = stack_item.pop_back();
      }
      reverse(actions.begin(), actions.end());
      completed_actions_and_nlp.push_back(pair<vector<unsigned>, Expression>(actions, nlp));
    }

    return completed_actions_and_nlp;
  }

  virtual vector<pair<vector<unsigned>, Expression>> abstract_log_prob_parser_beam_within_word(
          ComputationGraph* hg,
          const parser::Sentence& sent,
          const vector<shared_ptr<Parameters>>& bert_embeddings,
          int beam_size,
          int beam_filter_at_word_size,
          bool is_evaluation = true
  ) {
    //ComputationGraph hg;
      if (beam_filter_at_word_size < 0)
        beam_filter_at_word_size = beam_size;

    struct BeamItem {
      explicit BeamItem(std::shared_ptr<AbstractParserState> state, unsigned last_action, Expression score) :
              state(state), last_action(last_action), score(score)  {}
      std::shared_ptr<AbstractParserState> state;
      unsigned last_action;
      Expression score;
    };

    bool build_training_graph = !is_evaluation;
    bool apply_dropout = (DROPOUT && !is_evaluation);

    std::shared_ptr<AbstractParserState> initial_state = new_sentence(hg, sent, bert_embeddings, is_evaluation, build_training_graph, apply_dropout);

    vector<unsigned> results;

    vector<Stack<BeamItem>> completed;
    vector<Stack<BeamItem>> beam;

    beam.push_back(Stack<BeamItem>(BeamItem(initial_state, START_OF_SENTENCE_ACTION, input(*hg, 0.0))));

    for (int current_termc = 0; current_termc < sent.size(); current_termc++) {
      completed.clear();
      while (completed.size() < beam_size && !beam.empty()) {
        // old beam item, action to be applied, resulting total score
        vector<std::tuple<Stack<BeamItem>, unsigned, Expression>> successors;

        while (!beam.empty()) {
          const Stack<BeamItem> current_stack_item = beam.back();
          beam.pop_back();

          std::shared_ptr<AbstractParserState> current_parser_state = current_stack_item.back().state;
          vector<unsigned> valid_actions = current_parser_state->get_valid_actions();

          Expression adiste = current_parser_state->get_action_log_probs(valid_actions, nullptr).first;
          vector<float> adist = as_vector(hg->incremental_forward());

          for (unsigned action: valid_actions) {
            Expression action_score = pick(adiste, action);
            Expression total_score = current_stack_item.back().score + action_score;
            successors.push_back(
                    std::tuple<Stack<BeamItem>, unsigned, Expression>(current_stack_item, action, total_score)
            );
          }
        }

        int num_pruned_successors = std::min(beam_size, static_cast<int>(successors.size()));
        partial_sort(successors.begin(),
                     successors.begin() + num_pruned_successors,
                     successors.end(),
                     [](const std::tuple<Stack<BeamItem>, unsigned, Expression> &t1,
                        const std::tuple<Stack<BeamItem>, unsigned, Expression> &t2) {
                         return as_scalar(get<2>(t1).value()) > as_scalar(get<2>(t2).value()); // sort in descending order by total score
                     });
        while (successors.size() > num_pruned_successors)
          successors.pop_back();

        for (auto &successor : successors) {
          Stack<BeamItem> current_stack_item = get<0>(successor);
          std::shared_ptr<AbstractParserState> current_parser_state = current_stack_item.back().state;
          unsigned action = get<1>(successor);
          Expression total_score = get<2>(successor);
          std::shared_ptr<AbstractParserState> successor_parser_state = current_parser_state->perform_action(action);
          Stack<BeamItem> successor_stack_item = current_stack_item.push_back(
                  BeamItem(successor_parser_state,
                           action,
                           total_score)
          );
          if (successor_parser_state->word_completed())
            completed.push_back(successor_stack_item);
          else
            beam.push_back(successor_stack_item);
        }
      }

      sort(completed.begin(), completed.end(), [](const Stack<BeamItem> &t1, const Stack<BeamItem> &t2) {
          return as_scalar(t1.back().score.value()) > as_scalar(t2.back().score.value()); // sort in descending order by total score
      });

      beam.clear();
      // keep around a larger completion list if we're at the end of the sentence
      int num_pruned_completion = std::min(current_termc < sent.size() - 1 ? beam_filter_at_word_size : beam_size, static_cast<int>(completed.size()));
      std::copy(completed.begin(), completed.begin() + std::min(num_pruned_completion, static_cast<int>(completed.size())), std::back_inserter(beam));
    }

    vector<pair<vector<unsigned>, Expression>> completed_actions_and_nlp;
    for (const auto & completed_stack_item: completed) {
      completed_stack_item.back().state->finish_sentence();

      Stack<BeamItem> stack_item = completed_stack_item;
      Expression nlp = -1 * stack_item.back().score;
      vector<unsigned> actions;
      while (stack_item.back().last_action != START_OF_SENTENCE_ACTION) {
        actions.push_back(stack_item.back().last_action);
        stack_item = stack_item.pop_back();
      }
      reverse(actions.begin(), actions.end());
      completed_actions_and_nlp.push_back(pair<vector<unsigned>, Expression>(actions, nlp));
    }

    return completed_actions_and_nlp;
  }


};

std::shared_ptr<SymbolicParserState> initialize_symbolic_parser_state(const parser::Sentence& sent) {
  vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
  for (int i = 0; i < sent.size(); ++i) {
    bufferi[sent.size() - i] = i;
  }
  bufferi[0] = -999;
  vector<int> stacki; // position of words in the sentence of head of subtree
  stacki.push_back(-999); // not used for anything
  vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
  is_open_paren.push_back(-1); // corresponds to dummy symbol

  unsigned nt_count = 0; // number of times an NT has been introduced
  unsigned cons_nt_count = 0; // number of consecutive NTs with no intervening shifts or reduces
  int nopen_parens = 0;
  char prev_a = '0';

  unsigned action_count = 0;
  unsigned action = START_OF_SENTENCE_ACTION;

  bool was_word_completed = false;

  Stack<Bracket> completed_brackets;
  Stack<OpenBracket> open_brackets;

  unsigned words_shifted = 0;

  unsigned unary = 0;

  return std::make_shared<SymbolicParserState>(
          Stack<int>(bufferi),
          Stack<int>(stacki),
          Stack<int>(is_open_paren),
          nt_count,
          cons_nt_count,
          nopen_parens,
          prev_a,
          was_word_completed,
          action_count,
          action,
          completed_brackets,
          open_brackets,
          words_shifted,
          unary
  );

}

struct ParserBuilder : public AbstractParser {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_nt; // nonterminal embeddings
  LookupParameters* p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_pos; // pos embeddings (optional)
  Parameters* p_p2w;  // pos2word mapping (optional)
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack
  Parameters* p_cW;
  Parameters* p_bert_stack_bias;
  Parameters* p_B_bert_buffer;

  std::unordered_map<std::string, LookupParameters*> p_morph_embeddings;
  std::unordered_map<std::string, Parameters*> p_morph_W;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})), // nonterminal embeddings
      p_ntup(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})), // nonterminal embeddings when used in a composed representation
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM}, "pbias")),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM}, "A")),
      p_B(model->add_parameters({HIDDEN_DIM, BERT ? BERT_DIM : HIDDEN_DIM}, "B")),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM}, "S")),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM}, "cbias")),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM}, "p2a")),
      p_action_start(model->add_parameters({ACTION_DIM}, "action_start")),
      p_abias(model->add_parameters({ACTION_SIZE}, "abias")),

      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM}, "stack_guard")),

      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2}, "cW")) {

    if (BERT) {
      p_bert_stack_bias = model->add_parameters({HIDDEN_DIM});
      p_t = nullptr;
      p_t2l = nullptr;
      // initialize buffer_lstm in both BERT and !BERT conditional branches for backward compatibilty re: parameter ordering.
      // note the difference in the input dimensions
      if (USES_BUFFER_LSTM()) {
        buffer_lstm = new LSTMBuilder(LAYERS, BERT_DIM, HIDDEN_DIM, model);
        p_B_bert_buffer = model->add_parameters({HIDDEN_DIM, HIDDEN_DIM}, "B_bert_buffer");
      }
    } else {
      p_w = model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); // word embeddings
      p_w2l = model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM}, "w2l");
      p_ib = model->add_parameters({LSTM_INPUT_DIM}, "ib");
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); // pretrained word embeddings (not updated)
      if (USE_POS) {
        p_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
        p_p2w = model->add_parameters({LSTM_INPUT_DIM, POS_DIM}, "p2w");
      }
      // initialize this in both conditional branches for backward compatibilty re: parameter ordering
      if (USES_BUFFER_LSTM())
        buffer_lstm = new LSTMBuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model);
      p_buffer_guard = model->add_parameters({LSTM_INPUT_DIM}, "buffer_guard");
      if (pretrained.size() > 0) {
        p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
        for (auto it : pretrained)
          p_t->Initialize(it.first, it.second);
        p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM}, "t2l");
      } else {
        p_t = nullptr;
        p_t2l = nullptr;
      }
      if (USE_MORPH_FEATURES) {
        for (int i = 0; i < morphology_classes.size(); i++) {
          const string& _class = morphology_classes.Convert(i);
          auto& morph_dict = morphology_dicts[_class];
          p_morph_embeddings[_class] = model->add_lookup_parameters(morph_dict.size() + 1, {MORPH_DIM}); // + 1 for none
          p_morph_W[_class] = model->add_parameters({LSTM_INPUT_DIM, MORPH_DIM}, "morph_" + _class);
        }
      }
    }

  }

  // instance variables for each sentence
  ComputationGraph* hg;
  bool apply_dropout;
  Expression pbias, S, B, A, ptbias, ptW, p2w, ib, cbias, w2l, t2l, p2a, abias, action_start, cW, bert_stack_bias, B_bert_buffer;
  unordered_map<string, Expression> morph_W;

  std::shared_ptr<AbstractParserState> new_sentence(
          ComputationGraph* hg,
          const parser::Sentence& sent,
          const vector<shared_ptr<Parameters>>& bert_embeddings,
          bool is_evaluation,
          bool build_training_graph,
          bool apply_dropout
  ) override {
    this->hg = hg;
    this->apply_dropout = apply_dropout;

    if (!NO_STACK) stack_lstm.new_graph(*hg);
    if (USES_BUFFER_LSTM()) buffer_lstm->new_graph(*hg);
    if (!NO_ACTION_HISTORY) action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);

    if (!NO_STACK) stack_lstm.start_new_sequence();
    if (USES_BUFFER_LSTM()) buffer_lstm->start_new_sequence();
    if (!NO_ACTION_HISTORY) action_lstm.start_new_sequence();

    if (apply_dropout) {
      if (!NO_STACK) stack_lstm.set_dropout(DROPOUT);
      if (USES_BUFFER_LSTM()) buffer_lstm->set_dropout(DROPOUT);
      if (!NO_ACTION_HISTORY) action_lstm.set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      if (!NO_STACK) stack_lstm.disable_dropout();
      if (USES_BUFFER_LSTM()) buffer_lstm->disable_dropout();
      if (!NO_ACTION_HISTORY) action_lstm.disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }

    // variables in the computation graph representing the parameters
    pbias = parameter(*hg, p_pbias);
    S = parameter(*hg, p_S);
    B = parameter(*hg, p_B);
    A = parameter(*hg, p_A);

    cbias = parameter(*hg, p_cbias);
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    p2a = parameter(*hg, p_p2a);
    abias = parameter(*hg, p_abias);
    action_start = parameter(*hg, p_action_start);
    cW = parameter(*hg, p_cW);
    if (BERT) bert_stack_bias = parameter(*hg, p_bert_stack_bias);
    if (BERT && USES_BUFFER_LSTM()) B_bert_buffer = parameter(*hg, p_B_bert_buffer);
    for (auto& pair: p_morph_W) {
      morph_W[pair.first] = parameter(*hg, pair.second);
    }

    if (!NO_ACTION_HISTORY) action_lstm.add_input(action_start);

    vector<Expression> buffer(BERT_BUFFER_LSTM_USE_CLS_SEP ? sent.size() + 2 : sent.size() + 1);  // variables representing word embeddings
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    if (BERT) {
      if (BERT_BUFFER_LSTM_USE_CLS_SEP)
        // should contain CLS at the beginning and SEP at the end
        assert(bert_embeddings.size() == sent.size() + 2);
      else
        // should contain the SEP or CLS embedding at the end
        assert(bert_embeddings.size() == sent.size() + 1);

      if (REVERSE_TREES) {
        for (int i = 0; i < bert_embeddings.size() - 1; i++) {
          buffer[i + 1] = parameter(*hg, bert_embeddings[i].get());
        }
        buffer[0] = parameter(*hg, bert_embeddings[bert_embeddings.size() - 1].get());
      } else {
        for (int i = 0; i < bert_embeddings.size(); ++i) {
          buffer[bert_embeddings.size() - 1 - i] = parameter(*hg, bert_embeddings[i].get());
        }
      }
      if (BERT_FEATURE_DOWNSCALE != 1.0) {
        for (int i = 0; i < bert_embeddings.size(); i++) {
          buffer[i] = buffer[i] * input(*hg, 1.0 / BERT_FEATURE_DOWNSCALE);
        }
      }
    } else {
      w2l = parameter(*hg, p_w2l);
      if (USE_POS) p2w = parameter(*hg, p_p2w);
      ib = parameter(*hg, p_ib);

      for (int i = 0; i < sent.size(); ++i) {
        int wordid = sent.raw[i]; // this will be equal to unk at dev/test
        if (build_training_graph && !is_evaluation && singletons.size() > wordid && singletons[wordid] &&
            rand01() > 0.5)
          wordid = sent.unk[i];
        Expression w = lookup(*hg, p_w, wordid);
        vector<Expression> args = {ib, w2l, w}; // learn embeddings
        if (p_t && pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
          Expression t = const_lookup(*hg, p_t, sent.lc[i]);
          args.push_back(t2l);
          args.push_back(t);
        }
        if (USE_POS) {
          args.push_back(p2w);
          args.push_back(lookup(*hg, p_pos, sent.pos[i]));
        }
        if (USE_MORPH_FEATURES) {
          auto feat_mapping = sent.morphology_features[i];
          for (int idx = 0; idx < morphology_classes.size(); idx++) {
            const string &class_ = morphology_classes.Convert(idx);
            auto &morph_dict = morphology_dicts[class_];
            const auto &morph_singletons = morphology_singletons[class_];
            unsigned feature_idx = feat_mapping.count(idx) ? feat_mapping[idx] : morph_dict.size();
            if (build_training_graph && !is_evaluation && morph_singletons.size() > feature_idx &&
                morph_singletons[feature_idx] && rand01() > 0.5) {
              int unk_index = morph_dict.Convert(UNK);
              assert(unk_index >= 0);
              feature_idx = (unsigned) unk_index;
            }
            args.push_back(morph_W[class_]);
            args.push_back(lookup(*hg, p_morph_embeddings[class_], feature_idx));
          }
        }
        buffer[REVERSE_TREES ? i + 1 : sent.size() - i] = rectify(affine_transform(args));
      }
      // dummy symbol to represent the empty buffer
      buffer[0] = parameter(*hg, p_buffer_guard);

    }
    if (USES_BUFFER_LSTM()) {
      for (auto& b : buffer)
        buffer_lstm->add_input(b);
    }
    if (BERT_BUFFER_LSTM_USE_CLS_SEP) {
      // remove the CLS token from the front of the buffer so that tokens line up properly
      buffer.erase(buffer.begin(), buffer.begin() + 1);
    }

    assert(buffer.size() == sent.size() + 1);

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    if (!NO_STACK) stack_lstm.add_input(stack.back());

    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol

    std::shared_ptr<SymbolicParserState> symbolic_parser_state = initialize_symbolic_parser_state(sent);

    return std::static_pointer_cast<AbstractParserState>(
        std::make_shared<ParserState>(
            this,
            action_lstm.state(),
            (USES_BUFFER_LSTM() ? buffer_lstm->state() : cnn::NULL_RNN_POINTER),
            stack_lstm.state(),
            Stack<Expression>(buffer),
            Stack<Expression>(stack),
            Stack<int>(is_open_paren),
            symbolic_parser_state
        )
    );
  }

};

struct EnsembledParser : public AbstractParser {
  enum class CombineType { sum, product };

  vector<std::shared_ptr<ParserBuilder>> parsers;
  CombineType combine_type;
  bool factored_beam;

  explicit EnsembledParser(vector<std::shared_ptr<ParserBuilder>> parsers, CombineType combine_type, bool factored_beam) :
      parsers(parsers), combine_type(combine_type), factored_beam(factored_beam) {
    assert(!parsers.empty());
  }

  std::shared_ptr<AbstractParserState> new_sentence(
          ComputationGraph* hg,
          const parser::Sentence& sent,
          const vector<shared_ptr<Parameters>>& bert_embeddings,
          bool is_evaluation,
          bool build_training_graph,
          bool apply_dropout
  ) override {
    vector<std::shared_ptr<AbstractParserState>> states;
    for (const std::shared_ptr<ParserBuilder>& parser : parsers)
      states.push_back(parser->new_sentence(hg, sent, bert_embeddings, is_evaluation, build_training_graph, apply_dropout));
    return std::static_pointer_cast<AbstractParserState>(std::make_shared<EnsembledParserState>(this, states));
  }

  vector<pair<vector<unsigned>, Expression>> abstract_log_prob_parser_beam(
          ComputationGraph* hg,
          const parser::Sentence& sent,
          const vector<shared_ptr<Parameters>>& bert_embeddings,
          int beam_size
  ) {
    if (factored_beam) {
      set<vector<unsigned>> all_candidates;
      for (const std::shared_ptr<ParserBuilder>& parser : parsers) {
        auto this_beam = parser->abstract_log_prob_parser_beam(hg, sent, bert_embeddings, beam_size);
        for (auto results : this_beam) {
          all_candidates.insert(results.first);
        }
      }
      vector<pair<vector<unsigned>, Expression>> candidates_and_nlps;
      for (vector<unsigned> candidate : all_candidates) {
        ComputationGraph hg;
        double right;
        auto candidate_and_ensemble_nlp = abstract_log_prob_parser(
                &hg, sent, bert_embeddings, vector<int>(candidate.begin(), candidate.end()), &right, true, false
        );
        candidates_and_nlps.push_back(candidate_and_ensemble_nlp);
      }
      sort(candidates_and_nlps.begin(), candidates_and_nlps.end(), [](const std::pair<vector<unsigned>, Expression>& t1, const std::pair<vector<unsigned>, Expression>& t2) {
        return as_scalar(t1.second.value()) < as_scalar(t2.second.value()); // sort by ascending nlp
      });
      while (candidates_and_nlps.size() > beam_size)
        candidates_and_nlps.pop_back();
      return candidates_and_nlps;

    } else {
      return AbstractParser::abstract_log_prob_parser_beam(hg, sent, bert_embeddings, beam_size);
    }
  }
};

struct SymbolicParserState: public AbstractParserState {
  SymbolicParserState(
          Stack<int> bufferi,
          Stack<int> stacki,
          Stack<int> is_open_paren,
          unsigned nt_count,
          unsigned cons_nt_count,
          int nopen_parens,
          char prev_a,
          bool was_word_completed,
          unsigned action_count,
          unsigned action,
          Stack<Bracket> completed_brackets,
          Stack<OpenBracket> open_brackets,
          unsigned words_shifted,
          unsigned unary
  ) :
          bufferi(bufferi),
          stacki(stacki),
          is_open_paren(is_open_paren),
          nt_count(nt_count),
          cons_nt_count(cons_nt_count),
          nopen_parens(nopen_parens),
          prev_a(prev_a),
          was_word_completed(was_word_completed),
          action_count(action_count),
          action(action),
          completed_brackets(completed_brackets),
          open_brackets(open_brackets),
          words_shifted(words_shifted),
          unary(unary)
  {};

  const Stack<int> bufferi;
  const Stack<int> stacki;

  bool is_finished() const override {
    if (IN_ORDER) {
      return prev_a == 'T';
    } else {
      return stacki.size() == 2 && bufferi.size() == 1;
    }
  }

  bool word_completed() const override {
    return was_word_completed;
  }

  Stack<Bracket> get_completed_brackets() const override {
    return completed_brackets;
  }

  Stack<OpenBracket> get_open_brackets() const override {
    return open_brackets;
  }

  unsigned int get_words_shifted() const override {
    return words_shifted;
  }

  unsigned get_unary() const override {
    return unary;
  }

  bool action_is_valid(unsigned action) const override {
    return ! IsActionForbidden_Discriminative(
            action, prev_a, bufferi.size(), stacki.size(), nopen_parens, cons_nt_count, unary, completed_brackets, open_brackets, words_shifted
    );
  }

  vector<unsigned> get_valid_actions() const override {
    vector<unsigned> valid_actions;
    for (auto a: possible_actions) {
      if (action_is_valid(a))
        valid_actions.push_back(a);
    }
    return valid_actions;
  }

  std::pair<Expression, Expression> get_action_log_probs(const vector<unsigned>& valid_actions, vector<float>* augmentation) const override {
    throw std::runtime_error("get_action_log_probs not implemented for SymbolicParserState");
  }

  std::shared_ptr<AbstractParserState> perform_action(unsigned action) const override {
    const string& actionString = adict.Convert(action);
    const char ac = actionString[0];
    const char ac2 = actionString[1];

    Stack<int> new_bufferi(bufferi);
    Stack<int> new_stacki(stacki);
    Stack<int> new_is_open_paren(is_open_paren);
    unsigned new_nt_count = nt_count;
    unsigned new_cons_nt_count = cons_nt_count;
    int new_nopen_parens = nopen_parens;

    Stack<Bracket> new_completed_brackets(completed_brackets);
    Stack<OpenBracket> new_open_brackets(open_brackets);

    unsigned new_words_shifted = words_shifted;
    unsigned new_unary = unary;

    bool was_word_completed = false;

    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      new_stacki = new_stacki.push_back(bufferi.back());
      new_bufferi = new_bufferi.pop_back();
      new_is_open_paren = new_is_open_paren.push_back(-1);
      was_word_completed = (new_bufferi.size() > 1);
      if (IN_ORDER) {
        new_unary = 0;
      }
      new_words_shifted += 1;
      new_cons_nt_count = 0;
    } else if (ac == 'N') { // NT
      ++new_nopen_parens;
      if (IN_ORDER) {
        assert(stacki.size() > 1);
      } else {
        assert(bufferi.size() > 1);
      }
      auto it = action2NTindex.find(action);
      assert(it != action2NTindex.end());
      int nt_index = it->second;
      new_nt_count++;
      new_stacki = new_stacki.push_back(-1);
      new_is_open_paren = new_is_open_paren.push_back(nt_index);
      new_cons_nt_count += 1;
      new_open_brackets = new_open_brackets.push_back(OpenBracket(nt_index, words_shifted));
    } else if (ac == 'R') { // REDUCE
      --new_nopen_parens;
      if (IN_ORDER) {
        if (prev_a == 'N') new_unary += 1;
        if (prev_a == 'R') new_unary = 0;
      }
      assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)

      // find what paren we are closing
      int i = is_open_paren.size() - 1;
      // while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
      Stack<int> temp_is_open_paren = is_open_paren;
      while (temp_is_open_paren.back() < 0) {
        --i;
        assert(i >= 0);
        temp_is_open_paren = temp_is_open_paren.pop_back();
      }
      int nchildren = is_open_paren.size() - i - 1;
      if (IN_ORDER) {
        assert(nchildren + 1 > 0);
      } else {
        assert(nchildren > 0);
      }
      //cerr << "  number of children to reduce: " << nchildren << endl;

      // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
      // TO BE COMPOSED INTO A TREE EMBEDDING
      for (i = 0; i < nchildren; ++i) {
        assert(new_stacki.back() != -1);
        new_stacki = new_stacki.pop_back();
        new_is_open_paren = new_is_open_paren.pop_back();
      }
      new_is_open_paren = new_is_open_paren.pop_back(); // nt symbol
      assert(new_stacki.back() == -1);
      new_stacki = new_stacki.pop_back(); // nonterminal dummy

      if (IN_ORDER) {
        new_stacki = new_stacki.pop_back(); // leftmost
        new_is_open_paren = new_is_open_paren.pop_back(); // leftmost
        nchildren++; // not used here, but to parallel the structure in the stack function, where the incremented version is used to build the stack
      }

      new_stacki = new_stacki.push_back(999); // who knows, should get rid of this
      new_is_open_paren = new_is_open_paren.push_back(-1); // we just closed a paren at this position
      if (new_stacki.size() <= 2) {
        was_word_completed = true;
      }
      new_cons_nt_count = 0;
      new_completed_brackets = new_completed_brackets.push_back(close_bracket(new_open_brackets.back(), words_shifted));
      new_open_brackets = new_open_brackets.pop_back();
    } else { // TERM
      assert(IN_ORDER);
    }

    return std::make_shared<SymbolicParserState>(
            new_bufferi,
            new_stacki,
            new_is_open_paren,
            new_nt_count,
            new_cons_nt_count,
            new_nopen_parens,
            ac,
            was_word_completed,
            action_count + 1,
            action,
            new_completed_brackets,
            new_open_brackets,
            new_words_shifted,
            new_unary
    );
  }

  void finish_sentence() const override {
    assert(stacki.size() == 2); // guard symbol, root
    assert(bufferi.size() == 1); // guard symbol
    assert(open_brackets.empty());
  }

  const Stack<int> is_open_paren;

  const unsigned nt_count;
  const unsigned cons_nt_count;
  const int nopen_parens;
  const char prev_a;

  const bool was_word_completed = false;

  const unsigned action_count;
  const unsigned action;

  const Stack<Bracket> completed_brackets;
  const Stack<OpenBracket> open_brackets;

  const unsigned words_shifted;

  const unsigned unary;

};

struct ParserState : public AbstractParserState {
  ParserState(
      ParserBuilder* parser,
      RNNPointer action_state,
      RNNPointer buffer_state,
      RNNPointer stack_state,
      Stack<Expression> buffer,
      Stack<Expression> stack,
      Stack<int> is_open_paren,
      std::shared_ptr<SymbolicParserState> symbolic_parser_state
  ) :
      parser(parser),
      action_state(action_state),
      buffer_state(buffer_state),
      stack_state(stack_state),
      buffer(buffer),
      stack(stack),
      symbolic_parser_state(symbolic_parser_state),
      is_open_paren(is_open_paren)
  {}

  ParserBuilder* parser;

  const RNNPointer action_state;
  const RNNPointer buffer_state;
  const RNNPointer stack_state;

  const Stack<Expression> buffer;

  const Stack<Expression> stack;

  const std::shared_ptr<SymbolicParserState> symbolic_parser_state;

  const Stack<int> is_open_paren;

  bool is_finished() const override {
    if (!IN_ORDER) {
      assert((stack.size() == 2 && buffer.size() == 1) == symbolic_parser_state->is_finished());
    }
    return symbolic_parser_state->is_finished();
  }

  bool word_completed() const override {
    return symbolic_parser_state->word_completed();
  }

  Stack<Bracket> get_completed_brackets() const override {
    return symbolic_parser_state->get_completed_brackets();
  }

  Stack<OpenBracket> get_open_brackets() const override {
    return symbolic_parser_state->get_open_brackets();
  }

  unsigned int get_words_shifted() const override {
    return symbolic_parser_state->get_words_shifted();
  }

  unsigned get_unary() const override {
    return symbolic_parser_state->get_unary();
  }

  bool action_is_valid(unsigned action) const override {
    return symbolic_parser_state->action_is_valid(action);
  }

  vector<unsigned> get_valid_actions() const override {
    return symbolic_parser_state->get_valid_actions();
  }

  std::pair<Expression, Expression> get_action_log_probs(const vector<unsigned>& valid_actions, vector<float>* augmentation) const override {
    Expression stack_summary = NO_STACK ? Expression() : parser->stack_lstm.get_h(stack_state).back();
    Expression action_summary = NO_ACTION_HISTORY ? Expression() : parser->action_lstm.get_h(action_state).back();
    Expression buffer_summary = USES_BUFFER_LSTM() ? parser->buffer_lstm->get_h(buffer_state).back() : buffer.back();

    assert (symbolic_parser_state->bufferi.size() == buffer.size());

    if (parser->apply_dropout) { // TODO: don't the outputs of the LSTMs already have dropout applied?
      if (!NO_STACK) stack_summary = dropout(stack_summary, DROPOUT);
      if (!NO_ACTION_HISTORY) action_summary = dropout(action_summary, DROPOUT);
      if (USES_BUFFER_LSTM()) {
        // TODO(dfried): consider applying dropout to BERT as well
        buffer_summary = dropout(buffer_summary, DROPOUT);
      }
    }

    vector<Expression> elements{parser->pbias};
    if (!NO_STACK) {
      elements.push_back(parser->S);
      elements.push_back(stack_summary);
    }
    // with no-buffer BERT, parser->B was already applied
    elements.push_back(BERT && USES_BUFFER_LSTM() ? parser->B_bert_buffer : parser->B);
    elements.push_back(buffer_summary);
    if (!NO_ACTION_HISTORY) {
      elements.push_back(parser->A);
      elements.push_back(action_summary);
    }

    Expression p_t  = affine_transform(elements);
    Expression nlp_t = rectify(p_t);
    Expression r_t = affine_transform({parser->abias, parser->p2a, nlp_t});
    Expression r_t_aug;
    if (augmentation) {
      r_t_aug = r_t + input(*r_t.pg, Dim({static_cast<unsigned int>(augmentation->size())}), *augmentation);
    } else {
      r_t_aug = r_t;
    }
    if (UNNORMALIZED)
      return std::pair<Expression, Expression>(r_t, r_t_aug);
    else
    //  return log_softmax(r_t, valid_actions);
      return std::pair<Expression,Expression>(log_softmax_constrained(r_t, valid_actions), log_softmax_constrained(r_t_aug, valid_actions));
  }

  std::shared_ptr<AbstractParserState> perform_action(unsigned action) const override {
    std::shared_ptr<SymbolicParserState> new_symbolic_parser_state = std::static_pointer_cast<SymbolicParserState>(
            symbolic_parser_state->perform_action(action)
    );

    const string& actionString = adict.Convert(action);
    const char ac = actionString[0];
    const char ac2 = actionString[1];

    RNNPointer new_action_state = action_state;
    RNNPointer new_buffer_state = buffer_state;
    RNNPointer new_stack_state = stack_state;
    Stack<Expression> new_buffer(buffer);
    Stack<Expression> new_stack(stack);

    Stack<int> new_is_open_paren(is_open_paren);

    // add current action to action LSTM
    Expression actione = lookup(*parser->hg, parser->p_a, action);
    if (!NO_ACTION_HISTORY) {
      parser->action_lstm.add_input(action_state, actione);
      new_action_state = parser->action_lstm.state();
    }

    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)

      if (BERT) {
        new_stack = new_stack.push_back(affine_transform({parser->bert_stack_bias, parser->B, buffer.back()}));
      } else {
        new_stack = new_stack.push_back(buffer.back());
        new_buffer_state = parser->buffer_lstm->head_of(buffer_state);
      }
      if (!NO_STACK) {
        parser->stack_lstm.add_input(stack_state, new_stack.back());
        new_stack_state = parser->stack_lstm.state();
      }
      new_buffer = new_buffer.pop_back();
      new_is_open_paren = new_is_open_paren.push_back(-1);
    }
    else if (ac == 'N') { // NT
      if (IN_ORDER) {
        assert(stack.size() > 1);
      } else {
        assert(buffer.size() > 1);
      }
      auto it = action2NTindex.find(action);
      assert(it != action2NTindex.end());
      int nt_index = it->second;
      Expression nt_embedding = lookup(*parser->hg, parser->p_nt, nt_index);
      new_stack = new_stack.push_back(nt_embedding);
      if (!NO_STACK) {
        parser->stack_lstm.add_input(stack_state, nt_embedding);
        new_stack_state = parser->stack_lstm.state();
      }
      new_is_open_paren = new_is_open_paren.push_back(nt_index);
    } else if (ac == 'R'){ // REDUCE
      assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)

      // find what paren we are closing
      int i = is_open_paren.size() - 1;
      // while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
      Stack<int> temp_is_open_paren = is_open_paren;
      while (temp_is_open_paren.back() < 0) {
        --i;
        assert(i >= 0);
        temp_is_open_paren = temp_is_open_paren.pop_back();
      }
      Expression nonterminal = lookup(*parser->hg, parser->p_ntup, temp_is_open_paren.back());
      int nchildren = is_open_paren.size() - i - 1;
      if (IN_ORDER) {
        assert(nchildren + 1 > 0);
      } else {
        assert(nchildren > 0);
      }
      //cerr << "  number of children to reduce: " << nchildren << endl;
      vector<Expression> children(nchildren);
      parser->const_lstm_fwd.start_new_sequence();
      parser->const_lstm_rev.start_new_sequence();

      // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
      // TO BE COMPOSED INTO A TREE EMBEDDING
      for (i = 0; i < nchildren; ++i) {
        children[i] = new_stack.back();
        new_stack = new_stack.pop_back();
        if (!NO_STACK) new_stack_state = parser->stack_lstm.head_of(new_stack_state);
        new_is_open_paren = new_is_open_paren.pop_back();
      }
      new_is_open_paren = new_is_open_paren.pop_back(); // nt symbol
      new_stack = new_stack.pop_back(); // nonterminal dummy

      if (NO_STACK) {
        new_stack = new_stack.push_back(Expression()); // placeholder since we check size
      } else {
        new_stack_state = parser->stack_lstm.head_of(new_stack_state); // nt symbol

        if (IN_ORDER) {
          children.push_back(new_stack.back()); // leftmost
          new_stack = new_stack.pop_back(); // leftmost
          new_stack_state = parser->stack_lstm.head_of(new_stack_state); // leftmost
          new_is_open_paren = new_is_open_paren.pop_back(); // leftmost
          nchildren++;
        }

        // BUILD TREE EMBEDDING USING BIDIR LSTM
        parser->const_lstm_fwd.add_input(nonterminal);
        parser->const_lstm_rev.add_input(nonterminal);
        for (i = 0; i < nchildren; ++i) {
          parser->const_lstm_fwd.add_input(children[i]);
          parser->const_lstm_rev.add_input(children[nchildren - i - 1]);
        }
        Expression cfwd = parser->const_lstm_fwd.back();
        Expression crev = parser->const_lstm_rev.back();
        if (parser->apply_dropout) {
          cfwd = dropout(cfwd, DROPOUT);
          crev = dropout(crev, DROPOUT);
        }
        Expression c = concatenate({cfwd, crev});
        Expression composed = rectify(affine_transform({parser->cbias, parser->cW, c}));
        parser->stack_lstm.add_input(new_stack_state, composed);
        new_stack_state = parser->stack_lstm.state();
        new_stack = new_stack.push_back(composed);
      }
      new_is_open_paren = new_is_open_paren.push_back(-1); // we just closed a paren at this position
    } else { // TERM
      assert(IN_ORDER);
    }

    return std::make_shared<ParserState>(
        parser,
        new_action_state,
        new_buffer_state,
        new_stack_state,
        new_buffer,
        new_stack,
        new_is_open_paren,
        new_symbolic_parser_state
    );
  }

  void finish_sentence() const override {
    symbolic_parser_state->finish_sentence();
    assert(stack.size() == 2); // guard symbol, root

    assert(buffer.size() == 1); // guard symbol
  }
};

struct EnsembledParserState : public AbstractParserState {
  const EnsembledParser* parser;
  const vector<std::shared_ptr<AbstractParserState>> states;

  explicit EnsembledParserState(const EnsembledParser* parser, vector<std::shared_ptr<AbstractParserState>> states) :
      parser(parser), states(states) {
    assert(!states.empty());
  }

  bool is_finished() const override {
    return states.front()->is_finished();
  }

  bool word_completed() const override {
    return states.front()->word_completed();
  }

  bool action_is_valid(unsigned action) const override {
    return states.front()->action_is_valid(action);
  }

  vector<unsigned> get_valid_actions() const override {
    return states.front()->get_valid_actions();
  }

  Stack<Bracket> get_completed_brackets() const override {
    return states.front()->get_completed_brackets();
  }

  Stack<OpenBracket> get_open_brackets() const override {
    return states.front()->get_open_brackets();
  }

  unsigned int get_words_shifted() const override {
    return states.front()->get_words_shifted();
  }

  unsigned get_unary() const override {
    return states.front()->get_unary();
  }

  std::pair<Expression, Expression> get_action_log_probs(const vector<unsigned>& valid_actions, vector<float>* augmentation) const override {
    vector<Expression> all_log_probs;
    vector<Expression> all_log_probs_aug;
    for (const std::shared_ptr<AbstractParserState>& state : states) {
      Expression adiste;
      Expression adiste_aug;
      std::tie(adiste, adiste_aug) = state->get_action_log_probs(valid_actions, augmentation);
      all_log_probs.push_back(adiste);
      all_log_probs_aug.push_back(adiste_aug);
    }
    Expression combined_log_probs;
    Expression combined_log_probs_aug;
    switch (parser->combine_type) {
      case EnsembledParser::CombineType::sum: {
        // combined_log_probs = logsumexp(all_log_probs); // numerically unstable
        combined_log_probs = logsumexp_stable(all_log_probs);
        combined_log_probs_aug = logsumexp_stable(all_log_probs_aug);
        break;
      }
      case EnsembledParser::CombineType::product:
        combined_log_probs = sum(all_log_probs);
        combined_log_probs_aug = sum(all_log_probs_aug);
        break;
    }
    //return log_softmax(combined_log_probs, valid_actions);
    return pair<Expression, Expression>(log_softmax_constrained(combined_log_probs, valid_actions), log_softmax_constrained(combined_log_probs_aug, valid_actions));
  }

  std::shared_ptr<AbstractParserState> perform_action(unsigned action) const override {
    vector<std::shared_ptr<AbstractParserState>> new_states;
    for (const std::shared_ptr<AbstractParserState>& state : states)
      new_states.push_back(state->perform_action(action));
    return std::make_shared<EnsembledParserState>(parser, new_states);
  }

  void finish_sentence() const override {
    for (const std::shared_ptr<AbstractParserState>& state : states)
      state->finish_sentence();
  }
};

pair<vector<Bracket>, vector<OpenBracket>> actions_to_brackets(const parser::Sentence& sentence, const vector<int>& actions) {
  std::shared_ptr<AbstractParserState> parser_state = std::static_pointer_cast<AbstractParserState>(initialize_symbolic_parser_state(sentence));

  for (auto action: actions) {
    assert(action >= 0);
    parser_state = parser_state->perform_action((unsigned) action);
  }

  return pair<vector<Bracket>, vector<OpenBracket>>(parser_state->get_completed_brackets().values(), parser_state->get_open_brackets().values());
}

vector<Bracket> complete_actions_to_brackets(const parser::Sentence& sentence, const vector<int>& actions) {
  auto brackets = actions_to_brackets(sentence, actions);
  // make sure the stack is empty
  assert(brackets.second.empty());
  return brackets.first;
}

unsigned DynamicOracle::oracle_action(const AbstractParserState& parser_state) {
  unsigned words_shifted = parser_state.get_words_shifted();

  if (parser_state.action_is_valid(REDUCE_ACTION)) {
    map<Bracket, int> remaining_brackets;
    for (const auto& bracket: gold_brackets) {
      if (remaining_brackets.find(bracket) == remaining_brackets.end()) {
        remaining_brackets[bracket] = 0;
      }
      remaining_brackets[bracket] += 1;
    }
    for (const auto& bracket: parser_state.get_completed_brackets().values()) {
      if (remaining_brackets.find(bracket) != remaining_brackets.end()) {
        remaining_brackets[bracket] -= 1;
      }
    }

    assert(!parser_state.get_open_brackets().empty());
    const OpenBracket& stack_bracket = parser_state.get_open_brackets().back();

    bool gold_ahead = false;
    for (const auto& pair: remaining_brackets) {
      if (pair.second <= 0) {
        continue;
      }
      const Bracket& gold_bracket = pair.first;
      if (open_matches_closed(gold_bracket, stack_bracket)) {
        unsigned gold_end = get<2>(gold_bracket);
        if (gold_end == words_shifted) {
          return REDUCE_ACTION;
        } else if (gold_end > words_shifted) {
          gold_ahead = true;
        }
      }
    }
    if (!gold_ahead) {
      return REDUCE_ACTION;
    }
  }

  vector<OpenBracket> part_opened_here;
  for (auto& b: parser_state.get_open_brackets().values()) {
    if (b.second == words_shifted) {
      part_opened_here.push_back(b);
    }
  }

  vector<Bracket> gold_to_open;
  for (auto& b: gold_brackets) {
    if (get<1>(b) == words_shifted) {
      gold_to_open.push_back(b);
    }
  }
  std::reverse(gold_to_open.begin(), gold_to_open.end());

  auto part_it = part_opened_here.begin();
  auto gold_it = gold_to_open.begin();
  while (part_it != part_opened_here.end() && gold_it != gold_to_open.end()) {
    // check if labels match
    if (part_it->first == get<0>(*gold_it)) {
      gold_it++;
    }
    part_it++;
  }

  if (gold_it != gold_to_open.end()) {
    int nt_action = ntIndex2Action[get<0>(*gold_it)];
    assert(nt_action >= 0);
    if (parser_state.action_is_valid((unsigned) nt_action)) {
      return (unsigned) nt_action;
    }
  }

  if (parser_state.action_is_valid(SHIFT_ACTION)) {
    return SHIFT_ACTION;
  }

  throw std::runtime_error("no valid oracle actions");
}

void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

vector<string> string_representation(const vector<int>& actions, const parser::Sentence& sentence) {
  vector<string> string_rep;
  unsigned ti = 0;
  for (int a: actions) {
    string token;
    if (adict.Convert(a)[0] == 'N') {
      string_rep.push_back("(" + ntermdict.Convert(action2NTindex.find(a)->second));
    }
    else if (adict.Convert(a)[0] == 'S') {
      string_rep.push_back(posdict.Convert(sentence.pos[REVERSE_TREES ? sentence.pos.size() - 1 - ti : ti]));
      ti++;
    } else if (adict.Convert(a)[0] == 'T') {
      assert(IN_ORDER);
      // todo: check that we're at the end
      break;
    } else {
      string_rep.push_back(")");
    }
  }
  return string_rep;
}

vector<string> string_representation_u(const vector<unsigned>& actions, const parser::Sentence& sentence) {
  vector<int> converted(actions.begin(), actions.end());
  return string_representation(converted, sentence);
}

Tree to_tree(const vector<int>& actions, const parser::Sentence& sentence) {
  vector<string> string_rep = string_representation(actions, sentence);
  Tree tree = IN_ORDER ? parse_inorder(string_rep) : parse_linearized(string_rep);
  if (COLLAPSE_UNARY) {
    // Assume that the representation has unary chains collapsed, so expand for evaluation
    tree = tree.expand_unary("+");
  }
  if (REVERSE_TREES) {
    tree = tree.reverse();
  }

  return tree;
}

Tree to_tree_u(const vector<unsigned>& actions, const parser::Sentence& sentence) {
  vector<int> converted(actions.begin(), actions.end());
  return to_tree(converted, sentence);
}

void linearize_tree_helper(const Tree& node, const parser::Sentence& sentence, unsigned* words_shifted, vector<string>& linearized, bool include_tags, bool condense_parens) {
  const vector<Tree>& children = node.get_children();
  if (children.empty()) {
    string terminal = non_unked_termdict.Convert(sentence.non_unked_raw[*words_shifted]);
    if (include_tags) {
      string pos = posdict.Convert(sentence.pos[*words_shifted]);
      linearized.push_back("(" + pos);
      if (condense_parens) {
        linearized.push_back(terminal + ")");
      } else {
        linearized.push_back(terminal);
        linearized.push_back(")");
      }
    } else {
      linearized.push_back(terminal);
    }
    (*words_shifted)++;
  } else {
    linearized.push_back(node.get_symbol()); // should include the (
    for (const Tree& child: children) {
        linearize_tree_helper(child, sentence, words_shifted, linearized, include_tags, condense_parens);
    }
    if (condense_parens) {
      string back = linearized.back();
      linearized.pop_back();
      linearized.push_back(back + ")");
    } else {
      linearized.push_back(")");
    }
  }
}

vector<string> linearize_tree(const Tree& tree, const parser::Sentence& sentence, bool include_tags, bool condense_parens) {
  vector<string> linearized;
  unsigned words_shifted = 0;
  linearize_tree_helper(tree, sentence, &words_shifted, linearized, include_tags, condense_parens);
  return linearized;
}

void check_spmrl(const string& path, bool is_spmrl) {
    // TODO: allow other languages
  assert(is_spmrl == boost::starts_with(path, "french_"));
}

void bert_fw(WordFeaturizer* word_featurizer,
             const vector<parser::Sentence>& batch_sentences,
             vector<vector<shared_ptr<Parameters>>>& batch_bert_embeddings,
             TF_Tensor** feats_tensor,
             TF_Tensor** grads_tensor) {
  bool is_training = grads_tensor != nullptr;
  int batch_size = batch_sentences.size();
  vector<vector<int32_t>> batch_input_ids;
  vector<vector<int32_t>> batch_word_end_mask;
  if (SELF_ATTEND_WORDS) {
    for (auto &sent : batch_sentences) {
      vector<int32_t> this_end_mask(sent.size() + 2, 1);
      vector<int32_t> this_input_ids;
      this_input_ids.push_back(0); // CLS
      for (int i = 0; i < sent.size(); i++) {
        int wordid = sent.raw[i]; // this will be equal to unk at dev/test
        if (is_training && singletons.size() > wordid && singletons[wordid] && rand01() > 0.5)
          wordid = sent.unk[i];
        this_input_ids.push_back(wordid+2);
      }
      this_input_ids.push_back(1); // SEP
      batch_input_ids.emplace_back(this_input_ids);
      batch_word_end_mask.emplace_back(this_end_mask);
      assert(this_end_mask.size() == this_input_ids.size());
    }
  } else {
    for (auto &sent : batch_sentences) {
      batch_input_ids.emplace_back(sent.word_piece_ids_flat);
      batch_word_end_mask.emplace_back(sent.word_end_mask);
    }
  }
  assert(batch_input_ids.size() == batch_size);
  assert(batch_bert_embeddings.empty());

  vector<int32_t> flat_input_ids;
  vector<int32_t> flat_word_end_mask;
  const int num_words = word_featurizer->batch_inputs(batch_input_ids, batch_word_end_mask, flat_input_ids, flat_word_end_mask);
  const int num_subwords = flat_input_ids.size() / batch_size;
  //std::cout << "running BERT fw on batch of size " << batch_size << " num_words=" << num_words << std::endl;

  const std::vector<int64_t> dims = {batch_size, num_words, BERT_DIM};
  std::size_t data_size = sizeof(float_t);
  for (auto i : dims) {
    data_size *= i;
  }
  word_featurizer->run_fw(batch_size, num_subwords, flat_input_ids, flat_word_end_mask, feats_tensor, grads_tensor);

  assert(TF_Dim(*feats_tensor, 0) == batch_size);
  assert(TF_Dim(*feats_tensor, 1) == num_words);
  assert(TF_Dim(*feats_tensor, 2) == BERT_DIM);
  assert(TF_TensorType(*feats_tensor) == TF_FLOAT);

  Dim bert_dim{BERT_DIM};

  float* feat_arr = static_cast<float*>(TF_TensorData(*feats_tensor));
  float* grads_arr = nullptr;
  if (grads_tensor) {
    assert(TF_Dim(*grads_tensor, 0) == batch_size);
    assert(TF_Dim(*grads_tensor, 1) == num_words);
    assert(TF_Dim(*grads_tensor, 2) == BERT_DIM);
    assert(TF_TensorType(*grads_tensor) == TF_FLOAT);
    grads_arr = static_cast<float*>(TF_TensorData(*grads_tensor));
  }

  batch_bert_embeddings.resize(batch_size);
  for (int instance_within_batch = 0; instance_within_batch < batch_size; instance_within_batch++) {
    // CLS w1 w2 w3 ... wn SEP
    auto& bert_embeddings = batch_bert_embeddings[instance_within_batch];

    // if BERT_BUFFER_LSTM_USE_CLS_SEP is set, then place both CLS and SEP, otherwise just w1 ... [SEP|CLS], depending on value of BERT_BUFFER_USE_CLS
    // TODO(dfried): if this is actually useful, clean this up
    int max_index = batch_sentences[instance_within_batch].word_piece_ids.size() - (BERT_BUFFER_LSTM_USE_CLS_SEP ? 1 : 2);

    for (int word_index = 0; word_index <= max_index; word_index++) {
      int embedding_index = BERT_BUFFER_LSTM_USE_CLS_SEP ? word_index : word_index + 1;
      if (word_index == max_index && BERT_BUFFER_USE_CLS) {
        // use CLS instead of SEP
        assert(!BERT_BUFFER_LSTM_USE_CLS_SEP);
        embedding_index = 0;
      }
      int64_t feat_index = instance_within_batch * num_words * BERT_DIM + embedding_index * BERT_DIM;
      assert(feat_index >= 0);
      float *grads_location = grads_tensor ? &(grads_arr[feat_index]) : nullptr;
      shared_ptr<Parameters> params = make_shared<Parameters>(bert_dim, &(feat_arr[feat_index]), grads_location);
      bert_embeddings.push_back(params);
    }
  }
}

std::string dynet_param_path(std::string directory) {
  return directory + "/dynet_model.bin";
}

std::string bert_param_path(std::string directory) {
  return directory + "/bert_model.ckpt";
}

void make_corpus_indices(const parser::TopDownOracle &source_corpus, vector<int> &indices, int max_sentence_length) {
    assert(indices.empty());
    if (max_sentence_length <= 0) {
      indices.resize(source_corpus.size());
      std::iota(indices.begin(), indices.end(), 0);
    } else {
      for (int i = 0; i < source_corpus.size(); i++) {
        if (source_corpus.sents[i].size() <= max_sentence_length)
          indices.push_back(i);
      }
    }
}

int main(int argc, char** argv) {
  unsigned random_seed = cnn::Initialize(argc, argv);

  auto random_engine = std::default_random_engine(random_seed);

  cerr << "COMMAND LINE:";
  for (int i = 0; i < static_cast<int>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  int status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  if (conf.count("git_state")) {
    system("git rev-parse HEAD 1>&2");
    system("find . | egrep '*\\.(cc|h)$' | xargs git --no-pager diff -- 1>&2");
  }

  USE_POS = conf.count("use_pos_tags");
  USE_MORPH_FEATURES = conf.count("use_morph_features");
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  MORPH_DIM = conf["morph_dim"].as<unsigned>();
  USE_PRETRAINED = conf.count("words");
  NO_STACK = conf.count("no_stack");
  NO_ACTION_HISTORY = conf.count("no_action_history");

  SILVER_BLOCKS_PER_GOLD = conf["silver_blocks_per_gold"].as<int>();

  MAX_SENTENCE_LENGTH_TRAIN = conf["max_sentence_length_train"].as<int>();
  MAX_SENTENCE_LENGTH_EVAL = conf["max_sentence_length_eval"].as<int>();

  UNNORMALIZED = conf.count("unnormalized");
  BRACKET_CONSTRAINED_TRAINING = conf.count("bracket_constrained");

  IN_ORDER = conf.count("inorder");

  COLLAPSE_UNARY = conf.count("collapse_unary");
  REVERSE_TREES = conf.count("reverse_trees");

  if (conf.count("max_unary")) {
    MAX_UNARY = conf["max_unary"].as<unsigned>();
  }

  if (conf.count("max_cons_nt")) {
    MAX_CONS_NT = conf["max_cons_nt"].as<unsigned>();
  } else {
    if (REVERSE_TREES) {
      MAX_CONS_NT = 38;
    } else {
      MAX_CONS_NT = 8;
    }
  }


  if (BRACKET_CONSTRAINED_TRAINING && !COLLAPSE_UNARY) {
    cerr << "must set --collapse_unary with --bracket_constrained" << endl;
    abort();
  }

  BERT = conf.count("bert");
  BERT_LR = conf["bert_lr"].as<float>();
  BERT_WARMUP_STEPS = conf["bert_warmup_steps"].as<int>();

  SELF_ATTEND_WORDS = conf.count("self_attend_words");

  if (SELF_ATTEND_WORDS) {
    if (!BERT) {
      cerr << "must set --bert with --self_attend_words" << endl;
      abort();
    }
  }

  bool spmrl = conf.count("spmrl");

  if (BERT) {
    BERT_LARGE = conf.count("bert_large") != 0;
    if (BERT_LARGE) {
      BERT_GRAPH_PATH = BERT_LARGE_GRAPH_PATH;
      BERT_MODEL_PATH = BERT_LARGE_MODEL_PATH;
      BERT_DIM = 1024;
    } else {
      BERT_GRAPH_PATH = BERT_BASE_GRAPH_PATH;
      BERT_MODEL_PATH = BERT_BASE_MODEL_PATH;
      BERT_DIM = 768;
    }
    if (conf.count("bert_feature_downscale")) {
      BERT_FEATURE_DOWNSCALE = conf["bert_feature_downscale"].as<float>();
    } else {
      BERT_FEATURE_DOWNSCALE = BERT_LARGE ? 6.0 : 4.0;
    }
    if (conf.count("bert_model_dir")) {
      BERT_MODEL_PATH = conf["bert_model_dir"].as<string>();
      if (!conf.count("bert_graph_path")) {
        cerr << "must set --bert_graph_path if --bert_model_dir is set" << endl;
        abort();
      }
      BERT_GRAPH_PATH = conf["bert_graph_path"].as<string>();
    }
    cerr << "using BERT graph " << BERT_GRAPH_PATH << " with dimension " << BERT_DIM << endl;
    cerr << "BERT_FEATURE_DOWNSCALE: " << BERT_FEATURE_DOWNSCALE << endl;
  }

  while (BERT_MODEL_PATH.back() == '/') {
    BERT_MODEL_PATH.pop_back();
  }

  BERT_BUFFER_LSTM = conf.count("bert_buffer_lstm");
  BERT_BUFFER_USE_CLS = conf.count("bert_buffer_use_cls");
  BERT_BUFFER_LSTM_USE_CLS_SEP = conf.count("bert_buffer_lstm_use_cls_sep");

  if (BERT_BUFFER_LSTM_USE_CLS_SEP && !BERT) {
    cerr << "must set --bert if --bert_buffer_lstm_use_cls_sep is set" << endl;
    return 1;
  }


  if (BERT_BUFFER_LSTM_USE_CLS_SEP && REVERSE_TREES) {
    cerr << "--bert_buffer_lstm_use_cls_sep and --reverse_trees not currently implemented" << endl;
    return 1;
  }

  if (BERT_BUFFER_LSTM_USE_CLS_SEP && ! BERT_BUFFER_LSTM) {
    cerr << "must set --bert_buffer_lstm if --bert_buffer_lstm_use_cls_sep is set" << endl;
    return 1;
  }

  if (BERT_BUFFER_LSTM_USE_CLS_SEP && BERT_BUFFER_USE_CLS) {
    cerr << "cannot set both --bert_buffer_use_cls and --bert_buffer_lstm_use_cls_sep" << endl;
    return 1;
  }

  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }

  if (conf.count("train") && conf.count("test_data")) {
    cerr << "Cannot specify --train with --test-data." << endl;
    return 1;
  }

  if (conf.count("train") && conf.count("models")) {
    cerr << "Cannot specify --train with --models." << endl;
    return 1;
  }

  if (conf.count("model_dir") && conf.count("models")) {
    cerr << "Cannot specify --model_dir and --models." << endl;
    return 1;
  }

  if (conf.count("alpha")) {
    ALPHA = conf["alpha"].as<float>();
    if (ALPHA <= 0.f) { cerr << "--alpha must be between 0 and +infty\n"; abort(); }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<int>();
    //if (N_SAMPLES == 0) { cerr << "Please specify N>0 samples\n"; abort(); }
  }

  if (conf.count("dynamic_exploration_probability")) {
    DYNAMIC_EXPLORATION_PROBABILITY = conf["dynamic_exploration_probability"].as<float>();
    if (DYNAMIC_EXPLORATION_PROBABILITY < 0.0f || DYNAMIC_EXPLORATION_PROBABILITY > 1.0f) {
      cerr << "--dynamic_exploration_probability must be between 0 and 1, inclusive." << endl;
      return 1;
    }
  }

  DynamicOracle::ExplorationType exploration_type = DynamicOracle::ExplorationType::none;
  if (conf.count("dynamic_exploration")) {
    map<string, DynamicOracle::ExplorationType> exploration_types{
            {"greedy", DynamicOracle::ExplorationType::greedy},
            {"sample", DynamicOracle::ExplorationType::sample}
    };
    string exp_type = conf["dynamic_exploration"].as<string>();
    assert(exploration_types.count(exp_type));
    exploration_type = exploration_types.at(exp_type);
  }

  const int exploration_candidates = conf["dynamic_exploration_candidates"].as<int>();
  assert(exploration_candidates > 0);
  const bool exploration_include_gold = conf.count("dynamic_exploration_include_gold") > 0;
  if (exploration_include_gold) {
    assert(exploration_candidates > 1);
  }

  ostringstream os;

  if (!boost::filesystem::exists("models")) {
    boost::filesystem::create_directory("models");
  }

  os << "models/ntparse"
     << (USE_POS ? "_pos" : "")
     << (USE_PRETRAINED ? "_pretrained" : "")
     << (BERT ? "_bert" : "")
     << (IN_ORDER ? "_inorder" : "")
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << (NO_STACK ? "_no-stack" : "")
     << (NO_ACTION_HISTORY ? "_no-action-history" : "")
     << "-seed" << random_seed
     << "-pid" << getpid();

  const string fname = conf.count("model_output_dir") > 0 ? conf["model_output_dir"].as<string>() : os.str();
  cerr << "MODEL OUTPUT DIRECTORY: " << fname << endl;
  //bool softlinkCreated = false;

  bool discard_train_sents = (conf.count("train") == 0);

  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict, &morphology_classes, &morphology_dicts, &morphology_singletons);
  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict, &morphology_classes, &morphology_dicts, &morphology_singletons);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict, &morphology_classes, &morphology_dicts, &morphology_singletons);
  parser::TopDownOracle gold_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict, &morphology_classes, &morphology_dicts, &morphology_singletons);

  check_spmrl(conf["training_data"].as<string>(), spmrl);
  corpus.load_oracle(conf["training_data"].as<string>(), true, discard_train_sents, IN_ORDER, USE_MORPH_FEATURES);

  if (conf.count("bracketing_dev_data")) {
    check_spmrl(conf["bracketing_dev_data"].as<string>(), spmrl);
    corpus.load_bdata(conf["bracketing_dev_data"].as<string>());
  }

  bool has_gold_training_data = false;

  if (conf.count("gold_training_data")) {
    check_spmrl(conf["gold_training_data"].as<string>(), spmrl);
    gold_corpus.load_oracle(conf["gold_training_data"].as<string>(), true, discard_train_sents, IN_ORDER, USE_MORPH_FEATURES);
    if (conf.count("bracketing_dev_data")) {
      gold_corpus.load_bdata(conf["bracketing_dev_data"].as<string>());
    }
    has_gold_training_data = true;
  }

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  bool compute_distribution_stats = conf.count("compute_distribution_stats");

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  termdict.SetUnk(UNK); // we don't actually expect to use this often
     // since the Oracles are required to be "pre-UNKified", but this prevents
     // problems with UNKifying the lowercased data which needs to be loaded
  adict.Freeze();
  ntermdict.Freeze();
  if (USE_POS)
          posdict.Freeze();

  morphology_classes.Freeze();
  assert(morphology_dicts.size() == morphology_classes.size());
  for (auto& pair: morphology_dicts) {
    pair.second.Freeze();
    pair.second.SetUnk(UNK);
  }

  SHIFT_ACTION = adict.Convert("SHIFT");
  shift_actions.push_back(SHIFT_ACTION);
  REDUCE_ACTION = adict.Convert("REDUCE");
  reduce_actions.push_back(REDUCE_ACTION);

  if (IN_ORDER) {
    TERM_ACTION = adict.Convert("TERM");
    terminate_actions.push_back(TERM_ACTION);
  }

  {  // compute the singletons in the parser's training data
    /*
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
      */
    singletons.resize(termdict.size(), false);
    for (auto wc : corpus.raw_term_counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    check_spmrl(conf["dev_data"].as<string>(), spmrl);
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), false, false, IN_ORDER, USE_MORPH_FEATURES);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    check_spmrl(conf["test_data"].as<string>(), spmrl);
    test_corpus.load_oracle(conf["test_data"].as<string>(), false, false, IN_ORDER, USE_MORPH_FEATURES);
    if (conf.count("bracketing_test_data")) {
      test_corpus.load_bdata(conf["bracketing_test_data"].as<string>());
    }

  }

  //non_unked_termdict.Freeze();

  for (int i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
    ntIndex2Action[nt] = i;
    nonterm_actions.push_back(i);
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (int i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  /*
  // TODO(dfried): take this out
  const string dev_output = "/tmp/dev-output.txt";
  cerr << "writing transformed dev corpus to " + dev_output;
  ofstream dev_out(dev_output.c_str());
  for (int i = 0; i < dev_corpus.size(); i++) {
    Tree tree = to_tree(dev_corpus.actions[i], dev_corpus.sents[i]);
    print_tree(tree, dev_corpus.sents[i], true, dev_out);
  }
  dev_out.close();
  abort();
   */

  bool factored_ensemble_beam = conf.count("factored_ensemble_beam") > 0;


  bool sample = conf.count("samples") > 0;
  bool output_beam_as_samples = conf.count("output_beam_as_samples") > 0;
  bool output_candidate_trees = output_beam_as_samples || conf.count("samples_include_gold") || sample;

  int beam_size = conf["beam_size"].as<int>();


  // used for training
  int batch_size = conf["batch_size"].as<int>();
  int subbatch_max_tokens = conf["subbatch_max_tokens"].as<int>();
  // used for decoding
  int eval_batch_size = conf["eval_batch_size"].as<int>();

  auto decode = [&](AbstractParser& parser,
                    const parser::Sentence& sentence,
                    const vector<shared_ptr<Parameters>>& bert_embeddings,
                    StreamingStatistics* streaming_entropy = nullptr,
                    const vector<int>* bracket_constraint_actions = nullptr
  ) {
      double right = 0;
      // get parse and negative log probability
      ComputationGraph hg;
      vector<pair<vector<unsigned>, Expression>> results;
      if (beam_size > 1 || conf.count("beam_within_word")) {
        if (conf.count("beam_within_word")) {
          if (bracket_constraint_actions) {
            cerr << "bracket constraint actions not implemented for beam_within_word" << endl;
            abort();
          }
          results = parser.abstract_log_prob_parser_beam_within_word(&hg, sentence, bert_embeddings, beam_size,
                                                                          conf["beam_filter_at_word_size"].as<int>());
        } else {
          results = parser.abstract_log_prob_parser_beam(&hg, sentence, bert_embeddings, beam_size, true, bracket_constraint_actions);
        }
      } else {
        results.push_back(parser.abstract_log_prob_parser(
                &hg,
                sentence,
                bert_embeddings,
                vector<int>(),
                &right,
                true, // is_evaluation
                false, // sample
                false, // label_smoothing
                0.0, // label_smoothing_epsilon
                nullptr, // dynamic_oracle
                false, // loss_augmented
                false, // softmax_margin
                streaming_entropy,
                nullptr, // streaming_gold_prob
                nullptr, // entropy_accumulator
                bracket_constraint_actions
        ));
      }
      auto parse = results[0].first;
      float neg_log_prob = as_scalar(results[0].second.value());
      return make_pair(parse, neg_log_prob);
  };

  auto get_neg_log_likelihood = [&](
          AbstractParser& parser,
          const parser::Sentence& sentence,
          const vector<shared_ptr<Parameters>>& bert_embeddings,
          const vector<int>& actions,
          double* right,
          StreamingStatistics* streaming_gold_prob = nullptr,
          const vector<int>* bracket_constraint_actions = nullptr
  ) {
    ComputationGraph hg;
    parser.abstract_log_prob_parser(
            &hg,
            sentence,
            bert_embeddings,
            actions,
            right,
            true, // is_evaluation
            false,  // sample
            false, // label_smoothing
            0.0, // label_smoothing_epsilon
            nullptr, // dynamic_oracle
            false, // loss_augmented
            false, // softmax_margin
            nullptr, // streaming_entropy
            streaming_gold_prob,
            nullptr, // entropy_accumulator
            bracket_constraint_actions
    );
    return as_scalar(hg.incremental_forward());
  };

  auto evaluate = [&](const vector<parser::Sentence>& corpus_sentences, const vector<vector<int>>* corpus_gold_parses, const vector<vector<unsigned>>& pred_parses, const string& name, const vector<int>& indices, const string& reference_gold_fname = "") {
      string prefix = conf.count("eval_files_prefix") ? conf["eval_files_prefix"].as<string>() : ("/tmp/parser_" + to_string(getpid()));
      auto make_name = [&](const string& base) {
          ostringstream os;
          os << prefix << "_" << base;
          if (name != "")
            os << "." << name;
          os << ".txt";
          return os.str();
      };
      if (corpus_gold_parses)
        assert(corpus_gold_parses->size() == corpus_sentences.size());

      assert(pred_parses.size() == indices.size());
      if (corpus_gold_parses && pred_parses.size() != corpus_gold_parses->size()) {
        cerr << "WARNING: evaluating " << pred_parses.size() << " sentences, fewer than full corpus size " << corpus_gold_parses->size() << " (max_sentence_length_eval=" << MAX_SENTENCE_LENGTH_EVAL << ")" << endl;
      }

      const string pred_fname = make_name("pred");
      ofstream pred_out(pred_fname.c_str());

      if (!corpus_gold_parses && reference_gold_fname == "") {
        cerr << "must pass gold reference parses or a reference filename" << endl;
        abort();
      }

      ofstream gold_out;
      const string gold_fname = (corpus_gold_parses) ? make_name("gold") : reference_gold_fname;

      if (corpus_gold_parses) {
        gold_out = ofstream(gold_fname.c_str());
      }

      const string evalb_fname = make_name("evalb");

      MatchCounts match_counts;
      vector<MatchCounts> all_match_counts;

      for (int sii = 0; sii < indices.size(); sii++) {
        const auto& pred_parse = pred_parses[sii];

        int corpus_index = indices[sii];
        const auto& sentence = corpus_sentences[corpus_index];
        Tree pred_tree = to_tree(vector<int>(pred_parse.begin(), pred_parse.end()), sentence);
        print_tree(pred_tree, sentence, true, pred_out);

        if (corpus_gold_parses) {
          const auto &gold_parse = (*corpus_gold_parses)[corpus_index];
          Tree gold_tree = to_tree(gold_parse, sentence);

          MatchCounts this_counts = pred_tree.compare(gold_tree, spmrl);
          all_match_counts.push_back(this_counts);
          match_counts += this_counts;
          print_tree(gold_tree, sentence, true, gold_out);
        }

        //print_parse(pred_parse, sentence, true, pred_out);
        //print_parse(gold_parse, sentence, true, gold_out);
      }

      pred_out.close();
      if (corpus_gold_parses) {
        gold_out.close();
      }
      cerr << name << " parses in " << pred_fname << endl;
      cerr << name << " output in " << evalb_fname << endl;

      pair < Metrics, vector<MatchCounts>> results = metrics_from_evalb(gold_fname, pred_fname, evalb_fname, spmrl);

      if (corpus_gold_parses) {
        // ensure that match_counts match the status
        //pair<Metrics, vector<MatchCounts>> corpus_results = metrics_from_evalb(corpus.bracketed_fname, pred_fname, evalb_fname + "_corpus", spmrl);

        // first, check that we computed evalb correctly
        if (abs(match_counts.metrics().f1 - results.first.f1) > 1e-2) {
          cerr << "warning: score mismatch" << endl;
          cerr << "computed\trecall=" << match_counts.metrics().recall << ", precision="
               << match_counts.metrics().precision << ", F1=" << match_counts.metrics().f1 << "\n";
          cerr << "evalb\trecall=" << results.first.recall << ", precision=" << results.first.precision << ", F1="
               << results.first.f1 << "\n";
          if (results.first.recall == 0.0 && results.first.precision == 0.0 && results.first.f1 == 0.0) {
            cerr << "evalb appears to not have run; returning computed score" << endl;
            return match_counts.metrics();
          }
        }
        //cerr << "evalb corpus\trecall=" << corpus_results.first.recall << ", precision=" << corpus_results.first.precision << ", F1=" << corpus_results.first.f1 << "\n";

        for (int sii = 0; sii < all_match_counts.size(); sii++) {
          if (all_match_counts[sii] != results.second[sii]) {
            int corpus_index = indices[sii];
            cerr << "mismatch for " << (sii + 1) << endl;
            cerr << all_match_counts[sii].correct << " " << all_match_counts[sii].gold << " "
                 << all_match_counts[sii].predicted << endl;
            cerr << results.second[sii].correct << " " << results.second[sii].gold << " "
                 << results.second[sii].predicted << endl;
            Tree pred_tree = to_tree(vector<int>(pred_parses[sii].begin(), pred_parses[sii].end()),
                                     corpus_sentences[corpus_index]);
            Tree gold_tree = to_tree((*corpus_gold_parses)[corpus_index], corpus_sentences[corpus_index]);
            pred_tree.compare(gold_tree, spmrl, true);
          }
        }

        // then, if we have a reference file, double-check that scores against the gold parses in it are the same as the ones we dumped
        if (reference_gold_fname != "") {
          cerr << "checking against " << reference_gold_fname << endl;
          pair < Metrics, vector<MatchCounts>>
          reference_results = metrics_from_evalb(reference_gold_fname, pred_fname, evalb_fname, spmrl);
          if (abs(match_counts.metrics().f1 - reference_results.first.f1) > 1e-2) {
            cerr << "warning: score mismatch" << endl;
            cerr << "computed\trecall=" << match_counts.metrics().recall << ", precision="
                 << match_counts.metrics().precision << ", F1=" << match_counts.metrics().f1 << "\n";
            cerr << "evalb\trecall=" << reference_results.first.recall << ", precision="
                 << reference_results.first.precision << ", F1=" << reference_results.first.f1 << "\n";
          }
          return reference_results.first;
        }

      }
      return results.first;
  };

  struct DecodeStats {
    DecodeStats(const Metrics &metrics, double llh, double perplexity, double err) :
            metrics(metrics), llh(llh), perplexity(perplexity), err(err) {}
    const Metrics metrics;
    const double llh;
    const double perplexity;
    const double err;
  };

  vector<int> dev_indices;
  make_corpus_indices(dev_corpus, dev_indices, MAX_SENTENCE_LENGTH_EVAL);

  auto dev_decode = [&](AbstractParser& parser,
                        WordFeaturizer* word_featurizer,
                        StreamingStatistics* streaming_entropy,
                        StreamingStatistics* streaming_gold_prob
          ) {
      // TODO(dfried): support separate BERT word featurizers for ensemble models
      int dev_size = dev_indices.size();
      vector<vector<unsigned>> predicted;

      double trs = 0;
      double right = 0;
      double dwords = 0;
      double llh = 0;

      int sii = 0;

      while (sii < dev_size) {
        int this_batch_size = min(eval_batch_size, dev_size - sii);
        assert(this_batch_size > 0);
        vector<parser::Sentence> batch_sentences;
        for (int batch_sent = 0; batch_sent < this_batch_size; batch_sent++)
          batch_sentences.push_back(dev_corpus.sents[dev_indices[sii + batch_sent]]);

        TF_Tensor* bert_feats = nullptr;
        vector<vector<shared_ptr<Parameters>>> batch_bert_embeddings;

        if (BERT) {
          assert(word_featurizer);
          bert_fw(word_featurizer, batch_sentences, batch_bert_embeddings, &bert_feats, nullptr);
        } else {
          batch_bert_embeddings.resize(batch_sentences.size());
        }

        for (int batch_index = 0; batch_index < this_batch_size; batch_index++) {
          const auto& sentence=dev_corpus.sents[dev_indices[sii]];
          const vector<int>& actions=dev_corpus.actions[dev_indices[sii]];
          const auto& bert_embeddings = batch_bert_embeddings[batch_index];

          dwords += sentence.size();
          llh -= get_neg_log_likelihood(parser, sentence, bert_embeddings, actions, &right, streaming_gold_prob, BRACKET_CONSTRAINED_TRAINING ? &actions : nullptr);
          vector<unsigned> pred = decode(parser, sentence, bert_embeddings, streaming_entropy, BRACKET_CONSTRAINED_TRAINING ? &actions : nullptr).first;

          predicted.push_back(pred);
          trs += actions.size();
          sii++;
        }
        TF_DeleteTensor(bert_feats);
      }

      Metrics metrics = evaluate(dev_corpus.sents, &dev_corpus.actions, predicted, "dev", dev_indices, dev_corpus.bracketed_fname);
      return DecodeStats(metrics, llh, exp(-llh / dwords), (trs - right) / trs);
  };

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);

    string optimizer_name = conf["optimizer"].as<string>();
    assert(optimizer_name == "sgd" || optimizer_name == "adam");


    Model model;
    WordFeaturizer* word_featurizer = nullptr;
    ParserBuilder parser(&model, pretrained);
    unique_ptr<Trainer> optimizer = optimizer_name == "sgd"
                                    ? unique_ptr<Trainer>(new SimpleSGDTrainer(&model, 1e-6, conf["sgd_e0"].as<float>()))
                                    : unique_ptr<Trainer>(new AdamTrainer(&model, 1e-6, conf["adam_lr"].as<float>())); //(&model);
    parser::TrainingPosition training_position;
    StreamingStatistics streaming_f1;


    if (optimizer_name == "sgd") {
      optimizer->eta_decay = conf["sgd_eta_decay"].as<float>();
    }

    int lr_decay_patience = conf["lr_decay_patience"].as<int>();
    float lr_decay_factor = conf["lr_decay_factor"].as<float>();
    if (lr_decay_patience >= 0) {
      if (optimizer->eta_decay != 0.0f) {
        cerr << "disallowing run with both epoch-based lr decay and performance tolerance" << endl;
        abort();
      }
    }

    int dev_check_frequency = conf["dev_check_frequency"].as<int>();

    string bert_model_path = bert_param_path(BERT_MODEL_PATH);

    if (conf.count("model_dir")) {
      string model_dir = conf["model_dir"].as<string>();
      //cerr << "before load model" << endl;
      ifstream in(dynet_param_path(model_dir).c_str());
      if (conf.count("text_format")) {
        boost::archive::text_iarchive ia(in);
        ia >> model >> *optimizer >> training_position >> streaming_f1;
        //ia >> model;
      } else {
        boost::archive::binary_iarchive ia(in);
        ia >> model >> *optimizer >> training_position >> streaming_f1;
        //ia >> model;
      }
      cerr << "streaming f1 mean: " << streaming_f1.mean_value();
      cerr << " streaming f1 std mean: " << streaming_f1.total_standardized / streaming_f1.num_samples;
      cerr << endl;
      cerr << "after load model" << endl;
      // TODO(dfried): does BERT save its optimizer state?
      bert_model_path = bert_param_path(model_dir);
    } else {
      cerr << "using " << optimizer_name << " for training" << endl;
    }

    if (BERT) {
      word_featurizer = new WordFeaturizer(
              BERT_GRAPH_PATH.c_str(),
              bert_model_path,
              BERT_LR,
              BERT_WARMUP_STEPS
      );
    }

    //AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;

    bool min_risk_training = conf.count("min_risk_training") > 0;
    bool max_margin_training = conf.count("max_margin_training") > 0;
    bool softmax_margin_training = conf.count("softmax_margin_training") > 0;
    string min_risk_method = conf["min_risk_method"].as<string>();
    int min_risk_candidates = conf["min_risk_candidates"].as<int>();
    bool min_risk_include_gold = conf.count("min_risk_include_gold") > 0;
    assert(min_risk_candidates > 0);

    float label_smoothing_epsilon = conf["label_smoothing_epsilon"].as<float>();
    float entropy_regularization_weight = conf["entropy_regularization_weight"].as<float>();

    bool label_smoothing = (label_smoothing_epsilon != 0.0f);
    if (label_smoothing) {
      assert(!min_risk_training);
      assert(!max_margin_training);
      assert(label_smoothing_epsilon > 0);
    }

    if (min_risk_training) {
      assert(!max_margin_training);
      assert(exploration_type == DynamicOracle::ExplorationType::none);
    }

    if (max_margin_training) {
      assert(UNNORMALIZED);
      assert(!min_risk_training);
      assert(exploration_type == DynamicOracle::ExplorationType::none || exploration_type == DynamicOracle::ExplorationType::greedy);
    }

    if (softmax_margin_training) {
      assert(!min_risk_training);
      assert(!UNNORMALIZED);
    }

    //double tot_lad_score = 0.0;
    //double tot_gold_score = 0.0;
    //unsigned violations = 0;

    int sents_since_last_status = 0;

    //assert(!conf.count("set_iter"));

    /*
    if (conf.count("set_iter")) {
      assert(!has_gold_training_data);
      // todo: if support this for backward compat, also set the epoch and sentence
      training_position.iter = conf["set_iter"].as<int>();
      training_position.tot_seen = (unsigned) (training_position.iter + 1) * status_every_i_iterations;
      training_position.epoch = training_position.tot_seen / corpus.size();
      training_position.sentence = training_position.tot_seen % corpus.size();
      DecodeStats dev_decode_stats = dev_decode(parser, nullptr, nullptr);
      training_position.best_dev_f1 = dev_decode_stats.metrics.f1;
      training_position.best_dev_error = dev_decode_stats.err;
      cerr << "resuming parser at ";
      cerr << " iter: " << training_position.iter;
      cerr << " tot_seen: " << training_position.tot_seen;
      cerr << " epoch: " << training_position.epoch;
      cerr << " sentence: " << training_position.sentence;
      cerr << " dev f1: " << training_position.best_dev_f1;
      cerr << " dev err: " << training_position.best_dev_error;
    }
     */

    int save_frequency_minutes = conf["save_frequency_minutes"].as<int>();

//    double best_dev_err = 9e99;
//    double bestf1=0.0;


    auto train_sentence = [&](
            ComputationGraph& hg,
            const parser::Sentence& sentence,
            const vector<shared_ptr<Parameters>>& bert_embeddings,
            const vector<int>& actions,
            double* right
    ) -> tuple<Expression, Expression, MatchCounts> {
        Expression loss = input(hg, 0.0);
        Expression llh = input(hg, 0.0);

        MatchCounts sentence_match_counts;

        auto get_f1_and_update_mc = [&](Tree& gold_tree, const parser::Sentence& sentence, const vector<unsigned> actions) {
            Tree pred_tree = to_tree(vector<int>(actions.begin(), actions.end()), sentence);
            MatchCounts match_counts = pred_tree.compare(gold_tree, spmrl, false);
            sentence_match_counts += match_counts;
            return (float) match_counts.metrics().f1 / 100.0f;
        };

        Tree gold_tree = to_tree(actions, sentence);
        if (conf.count("min_risk_training")) {
          if (min_risk_method == "reinforce") {
            /*
            cerr << "gold ";
            print_parse(vector<unsigned>(actions.begin(), actions.end()), sentence, true, cerr);
            cerr << endl;
            */
            vector<Expression>* entropy_accumulator = nullptr;
            if (entropy_regularization_weight != 0.0) {
              entropy_accumulator = new vector<Expression>();
            }
            for (int i = 0; i < min_risk_candidates; i++) {
              double blank;
              pair<vector<unsigned>, Expression> sample_and_nlp;
              if (min_risk_include_gold && i == 0) {
                sample_and_nlp = parser.abstract_log_prob_parser(
                        &hg,
                        sentence,
                        bert_embeddings,
                        actions,
                        &blank,
                        false, // is_evaluation
                        false, // sample
                        false, // label_smoothing
                        0.0, // label_smoothing_epsilon
                        nullptr, // dynamic_oracle
                        false, // loss_augmented
                        false, // softmax_margin
                        nullptr, // streaming_entropy
                        nullptr, // streaming_gold_prob
                        entropy_accumulator,
                        BRACKET_CONSTRAINED_TRAINING ? &actions : nullptr
                );
              } else {
                sample_and_nlp = parser.abstract_log_prob_parser(
                        &hg,
                        sentence,
                        bert_embeddings,
                        vector<int>(),
                        &blank,
                        false, // is_evaluation
                        true, // sample
                        false, // label_smoothing
                        0.0, // label_smoothing_epsilon
                        nullptr, // dynamic_oracle
                        false, // loss_augmented
                        false, // softmax_margin
                        nullptr, // streaming_entropy
                        nullptr, // streaming_gold_prob
                        entropy_accumulator,
                        BRACKET_CONSTRAINED_TRAINING ? &actions : nullptr
                );
              }
              /*
              cerr << " " << i;
              print_parse(sample_and_nlp.first, sentence, true, cerr);
              cerr << endl;
              */
              float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, sample_and_nlp.first);
              double standardized_f1 = streaming_f1.standardize_and_update(scaled_f1);
              llh = llh - sample_and_nlp.second;
              loss = loss + (sample_and_nlp.second * input(hg, standardized_f1));
            }
            if (entropy_accumulator) {
              // TODO(dfried): consider incorporating this into the reward instead
              loss = loss - entropy_regularization_weight * sum(*entropy_accumulator);
              delete entropy_accumulator;
            }
            llh = llh * input(hg, 1.0 / min_risk_candidates);
            loss = loss * input(hg, 1.0 / min_risk_candidates);
          } else if (min_risk_method == "beam" || min_risk_method == "beam_noprobs" ||
                     min_risk_method == "beam_unnormalized" || min_risk_method == "beam_unnormalized_log") {
            if (entropy_regularization_weight != 0) {
              cerr << "entropy regularization not implemented for beam min_risk" << endl;
              abort();
            }

            if (BRACKET_CONSTRAINED_TRAINING) {
              cerr << "bracket constrained training not implemented for beam min_risk" << endl;
              abort();
            }

            auto candidates = parser.abstract_log_prob_parser_beam(
                    &hg,
                    sentence,
                    bert_embeddings,
                    (min_risk_include_gold ? min_risk_candidates - 1 : min_risk_candidates),
                    false // is_evaluation
            );

            if (min_risk_include_gold) {
              double blank;
              candidates.push_back(parser.abstract_log_prob_parser(
                      &hg,
                      sentence,
                      bert_embeddings,
                      actions,
                      &blank,
                      false, // is_evaluation
                      false // sample
              ));
            }

            if (min_risk_method == "beam") {
              // L_risk objective (Eq 4) from Edunov et al 2017: https://arxiv.org/pdf/1711.04956.pdf
                /*
              vector<Expression> log_probs_plus_log_losses;
              vector<Expression> log_probs;
              for (auto &parse_and_loss: candidates) {
                //Expression normed_log_prob = -parse_and_loss.second;
                //Expression normed_log_prob = -parse_and_loss.second - log(input(hg, parse_and_loss.first.size()));
                Expression normed_log_prob = -parse_and_loss.second * input(hg, 1.0 / parse_and_loss.first.size());
                float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, parse_and_loss.first);
                standardize_and_update_f1(scaled_f1);
                if (scaled_f1 < 1.0) {
                  log_probs_plus_log_losses.push_back(normed_log_prob + log(input(hg, 1.0f - scaled_f1)));
                }
                log_probs.push_back(normed_log_prob);
              }
              if (log_probs_plus_log_losses.size() > 0) {
                Expression loss = logsumexp(log_probs_plus_log_losses) - logsumexp(log_probs);
                loss_v = as_scalar(hg.incremental_forward());
              } else {
                Expression loss = input(hg, 0.0f);
                loss_v = as_scalar(hg.incremental_forward());
              }
                 */
              vector<Expression> normed_log_probs;
              for (auto &parse_and_loss: candidates) {
                llh = llh - parse_and_loss.second;
                Expression normed_log_prob = -parse_and_loss.second * input(hg, 1.0 / parse_and_loss.first.size());
                normed_log_probs.push_back(normed_log_prob);
              }
              Expression log_normalizer = logsumexp(normed_log_probs);
              assert(normed_log_probs.size() ==  candidates.size());
              for (int i = 0; i < normed_log_probs.size(); i++) {
                float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, candidates[i].first);
                scaled_f1 = streaming_f1.standardize_and_update(scaled_f1);
                loss = loss + input(hg, -scaled_f1) * exp(normed_log_probs[i] - log_normalizer);
              }
            } else if (min_risk_method == "beam_noprobs") {
              // TODO(dfried): this is hiding outer loss
              Expression loss = input(hg, 0.0f);
              float normalizer = 0.0;
              for (auto &parse_and_loss: candidates) {
                llh = llh - parse_and_loss.second;
                float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, parse_and_loss.first);
                //scaled_f1 = standardize_and_update_f1(scaled_f1);
                loss = loss + (parse_and_loss.second * input(hg, scaled_f1));
              }
              loss = loss * input(hg, 1.0 / candidates.size());
            } else {
              assert(min_risk_method == "beam_unnormalized" || min_risk_method == "beam_unnormalized_log");
              // TODO(dfried): this is hiding outer loss
              Expression loss = input(hg, 0.0f);
              float normalizer = 0.0;
              for (auto &parse_and_loss: candidates) {
                Expression log_prob = -parse_and_loss.second;
                llh = llh - parse_and_loss.second;
                Expression prob = exp(log_prob);
                float prob_v = as_scalar(hg.incremental_forward());
                float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, parse_and_loss.first);
                //scaled_f1 = standardize_and_update_f1(scaled_f1);
                normalizer += scaled_f1 * prob_v;
                loss = loss - log_prob * input(hg, (scaled_f1) * prob_v);
                //loss = loss - log_prob * input(hg, (scaled_f1));
              }
              if (min_risk_method == "beam_unnormalized_log") {
                if (normalizer != 0.0)
                  loss = loss * input(hg, 1.0f / normalizer);
              }
            }
          }  else {
            cerr << "invalid min_risk_method: " << min_risk_method << endl;
            exit(1);
          }
        }
        /* doesn't work; search errors
         else if (max_margin_training) {
          assert(UNNORMALIZED);
          double blank;

          DynamicOracle dynamic_oracle(sentence, actions);
          auto lad_and_neg_score = parser.abstract_log_prob_parser(&hg,
                                                                sentence,
                                                                vector<int>(),
                                                                right,
                                                                false, // is_evaluation
                                                                false, //sample
                                                                label_smoothing,
                                                                label_smoothing_epsilon,
                                                                &dynamic_oracle,
                                                                true // loss augmented
          );
          get_f1_and_update_mc(gold_tree, sentence, lad_and_neg_score.first);
          double lad_score = -as_scalar(lad_and_neg_score.second.value());
          tot_lad_score += lad_score;


          cerr << "lad: " << lad_score;
          print_parse(lad_and_neg_score.first, sentence, false, cerr);

          auto gold_and_neg_score = parser.abstract_log_prob_parser(&hg, sentence, actions, &blank, false, false);
          double gold_score = -as_scalar(gold_and_neg_score.second.value());
          tot_gold_score += gold_score;

          cerr << "gold: " << gold_score;
          print_parse(gold_and_neg_score.first, sentence, false, cerr);

          if (lad_score > gold_score) {
            Expression loss = gold_and_neg_score.second - lad_and_neg_score.second;
            loss_v = as_scalar(hg.incremental_forward());
            cerr << "violation: " << loss_v << endl;
            violations++;
          } else {
            loss_v = 0;
          }
          cerr << endl;
        }
          */
        else { // not min_risk
          bool run_gold = exploration_type == DynamicOracle::ExplorationType::none || exploration_include_gold;
          bool run_explore = exploration_type != DynamicOracle::ExplorationType::none;
          if (!run_explore) {
            assert(exploration_candidates == 1);
          }

          vector<Expression>* entropy_accumulator = nullptr;
          if (entropy_regularization_weight != 0.0) {
            entropy_accumulator = new vector<Expression>();
          }

          if (run_gold) {
            auto result_and_nlp = parser.abstract_log_prob_parser(&hg,
                                                                  sentence,
                                                                  bert_embeddings,
                                                                  actions,
                                                                  right,
                                                                  false, // is_evaluation
                                                                  false, //sample
                                                                  label_smoothing,
                                                                  label_smoothing_epsilon,
                                                                  nullptr, // dynamic_oracle
                                                                  max_margin_training, // loss_augmented
                                                                  softmax_margin_training, // softmax margin
                                                                  nullptr, // streaming_entropy
                                                                  nullptr, // streaming_gold_prob
                                                                  entropy_accumulator,
                                                                  BRACKET_CONSTRAINED_TRAINING ? &actions : nullptr
            );
            get_f1_and_update_mc(gold_tree, sentence, result_and_nlp.first);
            loss = loss + result_and_nlp.second * input(hg, 1.0 / exploration_candidates);
          }
          if (run_explore) {
            DynamicOracle dynamic_oracle(sentence, actions);
            int candidates_to_generate = exploration_include_gold ? exploration_candidates - 1 : exploration_candidates;
            for (int i = 0; i < candidates_to_generate; i++) {
              auto result_and_nlp = parser.abstract_log_prob_parser(&hg,
                                                                    sentence,
                                                                    bert_embeddings,
                                                                    vector<int>(),
                                                                    right,
                                                                    false, // is_evaluation
                                                                    exploration_type ==
                                                                    DynamicOracle::ExplorationType::sample, //sample
                                                                    label_smoothing,
                                                                    label_smoothing_epsilon,
                                                                    &dynamic_oracle,
                                                                    max_margin_training, // loss_augmented
                                                                    softmax_margin_training,
                                                                    nullptr, // streaming_entropy
                                                                    nullptr, // streaming_gold_prob
                                                                    entropy_accumulator,
                                                                    BRACKET_CONSTRAINED_TRAINING ? &actions : nullptr
              );
              if (DYNAMIC_EXPLORATION_PROBABILITY == 0.0f) {
                assert(vector<int>(result_and_nlp.first.begin(),
                                   result_and_nlp.first.end()) == actions);
              }
              get_f1_and_update_mc(gold_tree, sentence, result_and_nlp.first);
              loss = loss + result_and_nlp.second * input(hg, 1.0 / exploration_candidates);
            }
          }
          llh = -1.0 * loss;

          if (entropy_accumulator) {
            // TODO(dfried): consider incorporating this into the reward instead
            loss = loss - (entropy_regularization_weight / exploration_candidates * sum(*entropy_accumulator));
            delete entropy_accumulator;
          }
          //loss_v = as_scalar(result_and_nlp.second.value());
        }
        return tuple<Expression, Expression, MatchCounts>(loss, llh, sentence_match_counts);
    };


    auto save_model = [&](const string& base_filename, const string& save_type, const string& info, bool remove_old) {
        string prefix = base_filename + "_" + save_type;
        assert(prefix.find(' ') == std::string::npos);
        std::vector<boost::filesystem::path> old_files = utils::glob_files(prefix + "_*model");

        string save_dir = prefix;
        if (info != "") save_dir += "_" + info;
        save_dir += "_model";
        boost::filesystem::path file_path(save_dir);
        cerr << "writing model to directory " << save_dir  << "\t";

        if (boost::filesystem::exists(save_dir)) {
          cerr << "[WARNING: already exists]";
        } else {
          boost::filesystem::create_directory(save_dir);
        }

        ofstream out(dynet_param_path(save_dir));
        if (conf.count("text_format")) {
          boost::archive::text_oarchive oa(out);
          oa << model << *optimizer << training_position << streaming_f1;
          oa << termdict << adict << ntermdict << posdict;
        } else {
          boost::archive::binary_oarchive oa(out);
          oa << model << *optimizer << training_position << streaming_f1;
          oa << termdict << adict << ntermdict << posdict;
        }

        if (BERT) {
          word_featurizer->save_checkpoint(bert_param_path(save_dir));
        }

        cerr << "streaming f1 mean: " << streaming_f1.mean_value();
        cerr << " streaming f1 std mean: " << streaming_f1.total_standardized / streaming_f1.num_samples;
        cerr << endl;

        if (remove_old) {
          if (old_files.size() > 1)  {
            cerr << "multiple old directories exist; not removing: ";
            for (auto path: old_files) cerr << path.filename().string() << " ";
          } else if (old_files.size() == 1) {
            if (!equivalent(file_path, old_files[0])) {
              cerr << "removing file " << old_files[0].string();
              if (!boost::filesystem::remove_all(old_files[0])) {
                cerr << "unsuccessful";
              }
            } else {
              cerr << "old file has same name; not removing" << endl;
            }
          }
        }
        cerr << endl;

    };

    auto train_block = [&](const parser::TopDownOracle& corpus, vector<int>::iterator indices_begin, vector<int>::iterator indices_end, int epoch_size, int start_sentence = 0) {
      int sentence_count = std::distance(indices_begin, indices_end);
      int status_every_n_batches = min(status_every_i_iterations, sentence_count) / batch_size;
      cerr << "Number of sentences in current block: " << sentence_count << endl;
        /*
      cerr << "First sentence: ";
      for (auto& str: corpus.sents[*indices_begin].raw) cerr << str << " ";
      cerr << endl;
         */
      int trs = 0;
      int words = 0;
      double right = 0;
      double llh = 0;
      //tot_gold_score = 0.0;
      //tot_lad_score = 0.0;

      int sents = 0;

      //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
      auto time_start = chrono::system_clock::now();
      auto last_save_time = chrono::system_clock::now();

        MatchCounts block_match_counts;

        cerr << "staring at sentence " << start_sentence << endl;
        vector<int>::iterator index_iter = indices_begin + start_sentence;
        training_position.sentence = start_sentence;
        cerr << "first sentence (" << training_position.sentence << "): ";
        for (auto& word_id: corpus.sents[*index_iter].raw) {
          cerr << termdict.Convert(word_id) << " ";
        }
        cerr << endl;
        /*
        if (USE_MORPH_FEATURES) {
          for (auto& feature_map: corpus.sents[*index_iter].morphology_features) {
            for (auto& pair: feature_map) {
              const string& _class = morphology_classes.Convert(pair.first);
              auto& dict = morphology_dicts[_class];
              const string& feature = dict.Convert(pair.second);
              cerr << _class << "=" << feature << "|";
            }
            cerr << " ";
          }
          cerr << endl;
        }
         */
        while (true) {
          {
            int this_batch_size = min(batch_size, static_cast<int>(std::distance(index_iter, indices_end)));
            vector<std::tuple<int, int, int>> batch_length_batch_index_corpus_index;
            for (int i = 0; i < this_batch_size; i++) {
              int corpus_index = *(index_iter + i);
              int length = static_cast<int>(corpus.sents[corpus_index].word_piece_ids_flat.size());
              // put the batch index in so that ties are broken in batch (shuffled) order rather than corpus (invariant) order
              batch_length_batch_index_corpus_index.push_back(std::tuple<int,int,int>(length, i, corpus_index));
            }

            // sort ascending by length (lexicographical comparison)
            std::sort(batch_length_batch_index_corpus_index.begin(), batch_length_batch_index_corpus_index.end());

            int batch_index = 0;
            while (batch_index < this_batch_size) {
              vector<parser::Sentence> subbatch_sentences;
              vector<int> subbatch_corpus_indices;

              int max_length = 0;

              for (int subbatch_index = batch_index; subbatch_index < this_batch_size; subbatch_index++) {
                int length = get<0>(batch_length_batch_index_corpus_index[subbatch_index]);
                int corpus_index = get<2>(batch_length_batch_index_corpus_index[subbatch_index]);

                int new_max_length = max(length, max_length);

                if ((subbatch_index != batch_index)
                    && (new_max_length * (subbatch_sentences.size() + 1) > subbatch_max_tokens)) {
                  // Move to the next subbatch if we've exceeeded the token quota for this one (accounting
                  // for padding), but make sure sentences longer than subbatch_max_tokens still get
                  // processed in a subbatch of their own
                  break;
                } else {
                  subbatch_sentences.push_back(corpus.sents[corpus_index]);
                  subbatch_corpus_indices.push_back(corpus_index);
                }
                max_length = new_max_length;
              }
              assert(subbatch_sentences.size() == subbatch_corpus_indices.size());

              int this_subbatch_size = subbatch_sentences.size();

              batch_index += this_subbatch_size;

              ComputationGraph hg;
              vector<Expression> subbatch_losses;

              TF_Tensor* bert_feats = nullptr;
              TF_Tensor* bert_grads = nullptr;
              vector<vector<shared_ptr<Parameters>>> subbatch_bert_embeddings;

              if (BERT) {
                assert(word_featurizer);
                bert_fw(word_featurizer, subbatch_sentences, subbatch_bert_embeddings, &bert_feats, &bert_grads);
              } else {
                subbatch_bert_embeddings.resize(subbatch_sentences.size());
              }

              // TODO(dfried): this will use a small batch at the end of every epoch if training data isn't evenly divisible by batch size
              for (int subbatch_sent = 0; subbatch_sent < this_subbatch_size; subbatch_sent++) {
                int corpus_index = subbatch_corpus_indices[subbatch_sent];
                auto &sentence = corpus.sents[corpus_index];
                const vector<int> &actions = corpus.actions[corpus_index];
                auto& bert_embeddings = subbatch_bert_embeddings[subbatch_sent];
                auto loss_llh_and_mc = train_sentence(hg, sentence, bert_embeddings, actions, &right);
                subbatch_losses.push_back(get<0>(loss_llh_and_mc));
                double subbatch_llh = as_scalar(get<1>(loss_llh_and_mc).value());
                block_match_counts += get<2>(loss_llh_and_mc);
                if (!min_risk_training && !max_margin_training && (entropy_regularization_weight == 0.0)) {
                  double subbatch_loss = as_scalar(get<0>(loss_llh_and_mc).value());
                  if (subbatch_loss < 0) {
                    cerr << "loss < 0 on sentence " << corpus_index << ": loss=" << subbatch_loss << endl;
                  }
                  //assert(lp >= 0.0)
                }
                llh += subbatch_llh;
                trs += actions.size();
                words += sentence.size();
                sents++;
                training_position.sentence++;
                training_position.tot_seen++;
                sents_since_last_status++;
              }

              // We average the losses across the full batch, so divide by the
              // batch size (not the subbatch size!)
              Expression subbatch_loss = sum(subbatch_losses) / this_batch_size;
              double subbatch_loss_v = as_scalar(subbatch_loss.value());
              //cerr << "subbatch loss: " << subbatch_loss_v << endl;
              hg.backward();

              if (BERT) {
                word_featurizer->run_bw(bert_grads);
                TF_DeleteTensor(bert_feats);
                TF_DeleteTensor(bert_grads);
              }
            }

            // The optimizer only references parameters that are part of the
            // model, which does not include transient "parameters" that point
            // to tensorflow-allocated memory. The fact that transient
            // parameters have had their memory de-allocated shouldn't matter at
            // this point.
            optimizer->update(1.0);
            if (BERT) {
              word_featurizer->run_step();
              word_featurizer->run_zero_grad();
            }
            training_position.batches++;
            index_iter += this_batch_size;

          }

          if (training_position.batches % status_every_n_batches == 0) {
            training_position.iter++;
            optimizer->status();
            auto time_now = chrono::system_clock::now();
            auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
            cerr << "status #" << (training_position.iter + 1) << " batch #" << training_position.batches <<   " (epoch " << (static_cast<double>(training_position.tot_seen) / epoch_size) << ")";
            /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
            /*
          if (max_margin_training) {
            cerr <<
              " mean-gold-score: " << tot_gold_score / sents <<
              " mean-lad-score: " << tot_lad_score / sents <<
              " violations: " <<  violations << "/" << sents;
          } else
             */
            {
              cerr <<
                   " per-action-ppl: " << exp(-llh / trs) <<
                   " per-input-ppl: " << exp(-llh / words) <<
                   " per-sent-ppl: " << exp(-llh / sents_since_last_status) <<
                   " err: " << (trs - right) / trs;
            }
            cerr <<
                 " trace f1: " << block_match_counts.metrics().f1 / 100.f  <<
                 " [" << dur.count() / (double)sents_since_last_status << "ms per instance]";
            if (min_risk_training) {
              //sampled_f1s.clear();
              cerr << " mean sampled f1: " << streaming_f1.mean_value() << " mean standardized f1:" << streaming_f1.mean_standardized_value();
            }

            cerr << endl;
            llh = trs = right = words = sents = sents_since_last_status = 0;
            //tot_gold_score = tot_lad_score = 0.0;
            //violations = 0;

            int tot_seen_last_status = training_position.tot_seen_last_status;
            training_position.tot_seen_last_status = training_position.tot_seen;
            if ((training_position.tot_seen / dev_check_frequency) > (tot_seen_last_status / dev_check_frequency)) { // report on dev set
              auto t_start = chrono::high_resolution_clock::now();
              StreamingStatistics* streaming_entropy = nullptr;
              StreamingStatistics* streaming_gold_prob = nullptr;
              if (compute_distribution_stats) {
                streaming_entropy = new StreamingStatistics();
                streaming_gold_prob = new StreamingStatistics();
              }
              DecodeStats dev_decode_stats = dev_decode(parser, word_featurizer, streaming_entropy, streaming_gold_prob);
              auto t_end = chrono::high_resolution_clock::now();

              cerr << "recall=" << dev_decode_stats.metrics.recall
                   << ", precision=" << dev_decode_stats.metrics.precision
                   << ", F1=" << dev_decode_stats.metrics.f1
                   << ", complete match=" << dev_decode_stats.metrics.complete_match
                   << endl;
              cerr << "  **dev (iter=" << training_position.iter
                   << " epoch=" << (static_cast<double>(training_position.tot_seen) / epoch_size)
                   << ")\tllh=" << dev_decode_stats.llh
                   << " ppl: " << dev_decode_stats.perplexity
                   << " f1: " << dev_decode_stats.metrics.f1
                   << " err: " << dev_decode_stats.err
                   << "\t[" << dev_corpus.size() << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]"
                   << endl;
              if (compute_distribution_stats) {
                cerr << "mean entropy: " << streaming_entropy->mean_value() << " stddev entropy: " << streaming_entropy->std << " mean gold prob: " << streaming_gold_prob->mean_value() << " stddev gold prob: " << streaming_gold_prob->std << endl;
                delete streaming_entropy;
                delete streaming_gold_prob;
              }
              string model_tag = "it-" + to_string(training_position.iter) + "-f1-" + utils::to_string_precision(dev_decode_stats.metrics.f1, 2);
              long minutes_since_save = chrono::duration_cast<chrono::minutes>(chrono::system_clock::now() - last_save_time).count();
              bool print_next = false;
              if (save_frequency_minutes >= 0 && minutes_since_save >= save_frequency_minutes){
                  cerr << "  " << minutes_since_save << " minutes since save... ";
                save_model(fname, "periodic", model_tag , true);
                last_save_time = chrono::system_clock::now();
                print_next = true;
              }
              if (dev_decode_stats.metrics.f1 > training_position.best_dev_f1) {
                training_position.best_dev_error = dev_decode_stats.err;
                training_position.best_dev_f1=dev_decode_stats.metrics.f1;
                training_position.best_dev_f1_or_lr_updated_epoch = training_position.epoch;
                cerr << "  new best... ";
                save_model(fname, "best-epoch-" + to_string(training_position.epoch), model_tag, true);
                last_save_time = chrono::system_clock::now();
                print_next = true;
              }
              if (print_next) {
                auto next_sent = (index_iter == indices_end) ? indices_begin : index_iter;
                if (index_iter != indices_end) {
                  cerr << "next sentence: (" << training_position.sentence << ") ";
                  for (auto &word_id: corpus.sents[*next_sent].raw) {
                    cerr << termdict.Convert(word_id) << " ";
                  }
                } else {
                  cerr << "end of block";
                }
                cerr << endl;
              }
            }
            time_start = chrono::system_clock::now();
          }

          if (index_iter == indices_end) {
            break;
          }
        }
    };

    int start_epoch = training_position.epoch;
    training_position.epoch = 0;
    int shuffle_count = 0;

    while (training_position.epoch < start_epoch) {
      assert(!has_gold_training_data); // not implemented, should also shuffle silver data, possibly one more time depending on whether training_position.sentence is at least the gold_corpus size
      parser::TopDownOracle* main_corpus = &corpus;
      vector<int> main_indices;
      make_corpus_indices(*main_corpus, main_indices, MAX_SENTENCE_LENGTH_TRAIN);
      std::shuffle(main_indices.begin(), main_indices.end(), random_engine);
      training_position.epoch++;
      shuffle_count++;
    }
    cerr << "shuffle count: " << shuffle_count << endl;

    while (!requested_stop) {
      parser::TopDownOracle* main_corpus = &corpus;

      int sentence_count = 0;

      int start_sentence = 0;
      if (training_position.sentence != 0) {
        start_sentence = training_position.sentence;
        training_position.sentence = 0;
      }

      if (has_gold_training_data) {
        assert(start_sentence == 0); // not implemented
        training_position.in_silver_block = true;
        main_corpus = &gold_corpus;
        vector<int> silver_indices;
        make_corpus_indices(corpus, silver_indices, MAX_SENTENCE_LENGTH_TRAIN);
        std::shuffle(silver_indices.begin(), silver_indices.end(), random_engine);
        int offset = std::min(corpus.size(), gold_corpus.size() * SILVER_BLOCKS_PER_GOLD);
        train_block(corpus, silver_indices.begin(), silver_indices.begin() + offset, offset + gold_corpus.size(), start_sentence);
        sentence_count += offset;
        training_position.in_silver_block = false;
      }

      vector<int> main_indices;
      make_corpus_indices(*main_corpus, main_indices, MAX_SENTENCE_LENGTH_TRAIN);
      std::shuffle(main_indices.begin(), main_indices.end(), random_engine);
      sentence_count += main_indices.size();
      train_block(*main_corpus, main_indices.begin(), main_indices.end(), sentence_count, start_sentence);

      if (lr_decay_patience >= 0) {
        int since_last_update = training_position.epoch - training_position.best_dev_f1_or_lr_updated_epoch;
        if (since_last_update > lr_decay_patience) {
          cerr << since_last_update << " epochs have passed since the last new best F1 on dev or the last LR update; updating learning rates" << endl;
          // make sure to set this before calling update_epoch, which will set eta for the current epoch based on eta0.
          optimizer->eta0 *= lr_decay_factor;
          cerr << "new dynet eta0: " << optimizer->eta0 << endl;
          if (BERT) {
            word_featurizer->set_learning_rate(word_featurizer->get_last_set_learning_rate() * lr_decay_factor);
          }
          training_position.best_dev_f1_or_lr_updated_epoch = training_position.epoch;
        }
      }

      optimizer->update_epoch();

      training_position.epoch++;
      training_position.sentence = 0;

      /*
      ostringstream epoch_os;
      epoch_os << fname << "_" << epoch << ".bin";
      const string epoch_fname = epoch_os.str();
      cerr << "epoch " << epoch << " of " << sentence_count << " sentences, writing to "  << epoch_fname << endl;
      ofstream out(epoch_fname);
      boost::archive::binary_oarchive oa(out);
      oa << model << sgd;
      oa << termdict << adict << ntermdict << posdict;
      */

    }
    if (BERT) {
      delete word_featurizer;
    }
  } // should do training?

  bool interactive_mode = conf.count("interactive");

  if (test_corpus.size() > 0 || interactive_mode) { // do inference for test evaluation

    vector<std::shared_ptr<Model>> models;
    vector<std::shared_ptr<ParserBuilder>> parsers;
    std::shared_ptr<EnsembledParser> ensembled_parser;

    AbstractParser* abstract_parser;

    WordFeaturizer* word_featurizer;

    string bert_model_path;

    bool bracket_constrained_decode = conf.count("bracket_constrained_test_decode") || BRACKET_CONSTRAINED_TRAINING;

    if (bracket_constrained_decode && !COLLAPSE_UNARY) {
      cerr << "must set --collapse_unary with --bracket_constrained_test_decode" << endl;
      abort();
    }

    if (conf.count("model_dir")) {
      string model_dir = conf["model_dir"].as<string>();
      models.push_back(std::make_shared<Model>());
      parsers.push_back(std::make_shared<ParserBuilder>(models.back().get(), pretrained));
      cerr << "Loading single parser from " << model_dir << "..." << endl;
      ifstream in(dynet_param_path(model_dir));
      if (conf.count("text_format")) {
        boost::archive::text_iarchive ia(in);
        ia >> *models.back();
      } else {
        boost::archive::binary_iarchive ia(in);
        ia >> *models.back();
      }
      abstract_parser = parsers.back().get();
      if (BERT) {
        bert_model_path = bert_param_path(model_dir);
      }
    }

    else {
      assert(!BERT);
      map<string, EnsembledParser::CombineType> combine_types{
          {"sum", EnsembledParser::CombineType::sum},
          {"product", EnsembledParser::CombineType::product}
      };
      assert(conf.count("combine_type"));
      string combine_type = conf["combine_type"].as<string>();
      assert(combine_types.count(combine_type));

      assert(conf.count("models"));
      vector<string> model_paths = conf["models"].as<vector<string>>();
      assert(!model_paths.empty());
      cerr << "Loading ensembled parser..." << endl;
      for (const string& path : model_paths) {
        models.push_back(std::make_shared<Model>());
        parsers.push_back(std::make_shared<ParserBuilder>(models.back().get(), pretrained));
        cerr << "Loading parser from " << path << "..." << endl;
        ifstream in(path);
        if (conf.count("text_format")) {
            boost::archive::text_iarchive ia(in);
            ia >> *models.back();
        } else {
            boost::archive::binary_iarchive ia(in);
            ia >> *models.back();
        }
      }
      ensembled_parser = std::make_shared<EnsembledParser>(parsers, combine_types.at(combine_type), factored_ensemble_beam);
      cerr << "Loaded ensembled parser with combine_type of " << combine_type << "." << endl;
      abstract_parser = ensembled_parser.get();
    }

    if (BERT) {
      word_featurizer = new WordFeaturizer(
              BERT_GRAPH_PATH.c_str(),
              bert_model_path,
              BERT_LR, // this shouldn't be used
              BERT_WARMUP_STEPS // this shouldn't be used
      );
    }

    if (conf.count("interactive")) {
      boost::asio::io_service io_service;

      boost::asio::posix::stream_descriptor in(io_service, ::dup(STDIN_FILENO));
      //boost::asio::posix::stream_descriptor out(io_service, ::dup(STDOUT_FILENO));

      cout << "READY" << endl;

      int num_failures = 0;
      while (num_failures < 10) {
        TF_Tensor *bert_feats = nullptr;
        try {
          boost::asio::streambuf input_buffer;
          boost::asio::read_until(in, input_buffer, '\n');
          std::istream buffer(&input_buffer);
          std::string line;
          std::getline(buffer, line);
          std::stringstream stream(line);
          boost::property_tree::ptree json_data_in;
          boost::property_tree::read_json(stream, json_data_in);

          if (json_data_in.get<string>("action", "") == "exit") {
            abort();
          }

          for (auto &property: json_data_in) {
            cerr << property.first << " " << json_data_in.get<string>(property.first) << endl;
          }

          parser::Sentence sentence = corpus.sentence_from_json(
                  json_data_in,
                  false, // is_training
                  USE_MORPH_FEATURES, // read_morphology_features
                  BERT // read_bert
          );

          vector<parser::Sentence> batch_sentences { sentence };
          vector<vector<shared_ptr<Parameters>>> batch_bert_embeddings;

          if (BERT) {
            assert(word_featurizer);
            bert_fw(word_featurizer, batch_sentences, batch_bert_embeddings, &bert_feats, nullptr);
          } else {
            batch_bert_embeddings.resize(batch_sentences.size());
          }

          const auto& bert_embeddings = batch_bert_embeddings.back();
          pair <vector<unsigned>, float> result_and_nlp = decode(
                  *abstract_parser, sentence, bert_embeddings
          );

          std::ostringstream parse_out_stream;
          print_parse(result_and_nlp.first, sentence, true, parse_out_stream, true);
          float log_likelihood = -1 * result_and_nlp.second;

          boost::property_tree::ptree json_data_out;
          json_data_out.put("parse", parse_out_stream.str());
          json_data_out.put("llh", log_likelihood);

          boost::property_tree::write_json(cout, json_data_out, false);
          cout << endl;

        } catch (const std::exception& e) {
          cerr << e.what() << endl;
          num_failures++;
          if (BERT && bert_feats) {
            TF_DeleteTensor(bert_feats);
          }
        }

      }

    }


    vector<int> test_indices;
    make_corpus_indices(test_corpus, test_indices, MAX_SENTENCE_LENGTH_EVAL);

    int start_index = 0;
    int stop_index = test_indices.size();
    int block_count = conf["block_count"].as<int>();
    int block_num = conf["block_num"].as<int>();

    if (test_indices.size() != test_corpus.size()) {
      cerr << "WARNING: decoding " << test_indices.size() << " sentences, fewer than full corpus size " << test_corpus.size()
           << " (max_sentence_length_eval=" << MAX_SENTENCE_LENGTH_EVAL << ")" << endl;
    }

    if (block_count > 0) {
      assert(block_num < block_count);
      int q = test_indices.size() / block_count;
      int r = test_indices.size() % block_count;
      start_index = q * block_num + min(block_num, r);
      stop_index = q * (block_num + 1) + min(block_num + 1, r);
    }

    if (output_candidate_trees) {

      if (sample && output_beam_as_samples) {
        cerr << "warning: outputting samples and the contents of the beam\n";
      }

      ostringstream ptb_os;
      if (conf.count("ptb_output_file")) {
        ptb_os << conf["ptb_output_file"].as<string>();
      } else {
        ptb_os << "/tmp/parser_ptb_out." << getpid() << ".txt";
      }

      if (block_count > 0) {
        ptb_os << "_block-" << block_num;
      }

      ofstream ptb_out(ptb_os.str().c_str());

      double right = 0;
      auto t_start = chrono::high_resolution_clock::now();
      const vector<int> actions;
      vector<int> n_distinct_samples;

      int sii = start_index;
      while (sii < stop_index) {
        int this_batch_size = min(eval_batch_size, static_cast<int>(stop_index - sii));

        vector<parser::Sentence> batch_sentences;
        for (int batch_sent = 0; batch_sent < this_batch_size; batch_sent++)
          batch_sentences.push_back(test_corpus.sents[test_indices[sii + batch_sent]]);

        TF_Tensor *bert_feats = nullptr;
        vector<vector<shared_ptr<Parameters>>> batch_bert_embeddings;

        if (BERT) {
          assert(word_featurizer);
          bert_fw(word_featurizer, batch_sentences, batch_bert_embeddings, &bert_feats, nullptr);
        } else {
          batch_bert_embeddings.resize(batch_sentences.size());
        }

        for (int batch_index = 0; batch_index < this_batch_size; batch_index++) {
          const auto &sentence = test_corpus.sents[test_indices[sii]];
          // TODO: this overrides dynet random seed, but should be ok if we're only sampling
          const auto& bert_embeddings = batch_bert_embeddings[batch_index];
          const auto& actions = test_corpus.actions[test_indices[sii]];
          cnn::rndeng->seed(sii);
          set<vector<unsigned>> samples;
          if (conf.count("samples_include_gold")) {
            ComputationGraph hg;
            auto result_and_nlp = abstract_parser->abstract_log_prob_parser(
                    &hg, sentence, bert_embeddings, actions, &right, true
            );
            vector<unsigned> result = result_and_nlp.first;
            double nlp = as_scalar(result_and_nlp.second.value());
            cout << test_indices[sii] << " ||| " << -nlp << " |||";
            vector<unsigned> converted_actions(actions.begin(), actions.end());
            print_parse(converted_actions, sentence, false, cout);
            ptb_out << test_indices[sii] << " ||| " << -nlp << " |||";
            print_parse(converted_actions, sentence, true, ptb_out);
            samples.insert(converted_actions);
          }

          for (int z = 0; z < N_SAMPLES; ++z) {
            ComputationGraph hg;
            // TODO(dfried): should remove actions here?
            pair < vector<unsigned>,
                    Expression > result_and_nlp = abstract_parser->abstract_log_prob_parser(
                            &hg, sentence, bert_embeddings, actions, &right,
                            true, // is_evaluation
                            sample, // sample
                            false, // label_smoothing
                            0.0, // label_smoothing_epsilon
                            nullptr, // dynamic_oracle
                            false, // loss_augmented
                            false, // softmax_margin
                            nullptr, // streaming_entropy
                            nullptr, // streaming_gold_prob
                            nullptr, // entropy_accumulator
                            bracket_constrained_decode ? &actions : nullptr
                    ); // TODO: fix ordering of sample and eval here, produces correct behavior but is confusing
            double lp = as_scalar(result_and_nlp.second.value());
            cout << test_indices[sii] << " ||| " << -lp << " |||";
            print_parse(result_and_nlp.first, sentence, false, cout);
            ptb_out << test_indices[sii] << " ||| " << -lp << " |||";
            print_parse(result_and_nlp.first, sentence, true, ptb_out);
            samples.insert(result_and_nlp.first);
          }

          if (output_beam_as_samples) {
            ComputationGraph hg;
            vector<pair < vector<unsigned>, Expression>> beam_results;
            if (conf.count("beam_within_word")) {
              if (bracket_constrained_decode) {
                cerr << "bracket constraint actions not implemented for beam_within_word" << endl;
                abort();
              }
              beam_results = abstract_parser->abstract_log_prob_parser_beam_within_word(
                      &hg, sentence, bert_embeddings, beam_size, conf["beam_filter_at_word_size"].as<int>()
              );
            } else {
              beam_results = abstract_parser->abstract_log_prob_parser_beam(
                      &hg, sentence, bert_embeddings, beam_size, true,
                      bracket_constrained_decode ? &actions : nullptr
              );
            }
            if (beam_results.size() < beam_size) {
              cerr << "WARNING: only " << beam_results.size() << " parses found by beam search for sent " << test_indices[sii]
                   << endl;
            }
            int num_results = beam_results.size();
            for (int i = 0; i < beam_size; i++) {
              int ix = std::min(i, num_results - 1);
              pair < vector<unsigned>, Expression > result_and_nlp = beam_results[ix];
              double nlp = as_scalar(result_and_nlp.second.value());
              cout << test_indices[sii] << " ||| " << -nlp << " |||";
              print_parse(result_and_nlp.first, sentence, false, cout);
              ptb_out << test_indices[sii] << " ||| " << -nlp << " |||";
              print_parse(result_and_nlp.first, sentence, true, ptb_out);
              samples.insert(result_and_nlp.first);
            }
          }

          n_distinct_samples.push_back(samples.size());
          sii++;
        } // for batch_index
        if (BERT) {
          TF_DeleteTensor(bert_feats);
        }
      } // sii < stop_index
      ptb_out.close();

      double avg_distinct_samples = accumulate(n_distinct_samples.begin(), n_distinct_samples.end(), 0.0) /
                                    (double) n_distinct_samples.size();
      cerr << "avg distinct samples: " << avg_distinct_samples << endl;
    }



    // shortcut: only do a test decode if we aren't outputting any candidate trees
    if (!output_candidate_trees) {
      auto t_start = chrono::high_resolution_clock::now();
      vector<vector<unsigned>> predicted(test_indices.size());
      StreamingStatistics streaming_entropy;
      StreamingStatistics streaming_gold_prob;
      double neg_log_likelihood = 0;
      double actions_correct = 0;
      unsigned long num_actions = 0;

      /*
      // decode longer sentences first to fail faster if OOM
      sort(test_indices.begin(), test_indices.end(),
           [&](int ix1, int ix2) {
               return test_corpus.sents[ix1].size() > test_corpus.sents[ix2].size();
      });
       */

      int sii = 0;
      if (conf.count("decode_start_index")) {
        sii = conf["decode_start_index"].as<int>();
      }
      while (sii < test_indices.size()) {
        int this_batch_size = min(eval_batch_size, stop_index - sii);
        //const vector<parser::Sentence> batch_sentences(test_corpus.sents.begin() + sii, test_corpus.sents.begin() + sii + this_batch_size);
        vector<parser::Sentence> batch_sentences;
        for (int batch_sent = 0; batch_sent < this_batch_size; batch_sent++)
          batch_sentences.push_back(test_corpus.sents[test_indices[sii + batch_sent]]);


        TF_Tensor *bert_feats = nullptr;
        vector<vector<shared_ptr<Parameters>>> batch_bert_embeddings;

        if (BERT) {
          assert(word_featurizer);
          bert_fw(word_featurizer, batch_sentences, batch_bert_embeddings, &bert_feats, nullptr);
        } else {
          batch_bert_embeddings.resize(batch_sentences.size());
        }

        for (int batch_index = 0; batch_index < this_batch_size && sii < stop_index; batch_index++) {
          int corpus_index = test_indices[sii];
          if (sii % 10 == 0) {
            cerr << "\r decoding sent: " << sii;
          }
          const auto& sentence = test_corpus.sents[corpus_index];
          const auto& bert_embeddings = batch_bert_embeddings[batch_index];
          const auto& actions = test_corpus.actions[corpus_index];
          pair <vector<unsigned>, float> result_and_nlp = decode(
                  *abstract_parser, sentence, bert_embeddings, &streaming_entropy,
                  bracket_constrained_decode ? &actions : nullptr
          );
          predicted[corpus_index] = result_and_nlp.first;
          neg_log_likelihood += get_neg_log_likelihood(
                  *abstract_parser, sentence, bert_embeddings, actions, &actions_correct, &streaming_gold_prob,
                  bracket_constrained_decode ? &actions : nullptr
          );
          num_actions += actions.size();
          sii++;
        }
        if (BERT) {
          TF_DeleteTensor(bert_feats);
        }
      }
      cerr << endl;
      auto t_end = chrono::high_resolution_clock::now();
      Metrics metrics = evaluate(test_corpus.sents, bracket_constrained_decode ? nullptr : &test_corpus.actions, predicted, "test", test_indices, test_corpus.bracketed_fname);
      cerr << "recall=" << metrics.recall << ", precision=" << metrics.precision << ", F1=" << metrics.f1 << ", complete match=" << metrics.complete_match << "\n";
      cerr << "decode: mean entropy: " << streaming_entropy.mean_value() << " stddev entropy: " << streaming_entropy.std << endl; //<< " mean gold prob: " << streaming_gold_prob.mean_value();
      cerr << "gold: mean gold probability: " << streaming_gold_prob.mean_value() << " stddev gold probability: " << streaming_entropy.std << " avg log likelihood: " << -neg_log_likelihood / (test_indices.size()) << " actions correct: " << actions_correct / num_actions << endl; //<< " mean gold prob: " << streaming_gold_prob.mean_value();
    }
  }


}
