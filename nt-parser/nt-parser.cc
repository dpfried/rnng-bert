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

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict, non_unked_termdict;

const unsigned START_OF_SENTENCE_ACTION = std::numeric_limits<unsigned>::max();

volatile bool requested_stop = false;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;

unsigned MAX_CONS_NT = 8;

unsigned SHIFT_ACTION = UINT_MAX;
unsigned REDUCE_ACTION = UINT_MAX;

float ALPHA = 1.f;
float DYNAMIC_EXPLORATION_PROBABILITY = 1.f;
unsigned N_SAMPLES = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
unsigned POS_SIZE = 0;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X
std::map<int,int> ntIndex2Action;  // pass in index of X, return index of action NT(X)

bool USE_POS = false;  // in discriminative parser, incorporate POS information in token embedding
bool USE_PRETRAINED = false;  // in discriminative parser, use pretrained word embeddings (not updated)
bool NO_STACK = false;
unsigned SILVER_BLOCKS_PER_GOLD = 10;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;

vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
          ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
          ("explicit_terminal_reduce,x", "[recommended] If set, the parser must explicitly process a REDUCE operation to complete a preterminal constituent")
          ("dev_data,d", po::value<string>(), "Development corpus")
          ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")
          ("gold_training_data", po::value<string>(), "List of Transitions - smaller corpus (e.g. wsj in a wsj+silver experiment)")
          ("silver_blocks_per_gold", po::value<unsigned>()->default_value(10), "How many same-sized blocks of the silver data should be sampled and trained, between every train on the entire gold set?")
          ("test_data,p", po::value<string>(), "Test corpus")
          ("dropout,D", po::value<float>(), "Dropout rate")
          ("samples,s", po::value<unsigned>(), "Sample N trees for each test sentence instead of greedy max decoding")
          ("output_beam_as_samples", "Print the items in the beam in the same format as samples")
          ("samples_include_gold", "Also include the gold parse in the list of samples output")
          ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
          ("model,m", po::value<string>(), "Load saved model from this file")
          ("use_pos_tags,P", "make POS tags visible to parser")
          ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
          ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
          ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
          ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
          ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
          ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
          ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
          ("train,t", "Should training be run?")
          ("words,w", po::value<string>(), "Pretrained word embeddings")
          ("max_cons_nt", po::value<unsigned>()->default_value(8), "maximum number of non-terminals that can be opened consecutively")
          ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
          ("beam_within_word", "greedy decode within word")
          ("beam_filter_at_word_size", po::value<int>()->default_value(-1), "when using beam_within_word, filter word completions to this size (defaults to decode_beam_size if < 0)")
          ("no_stack,S", "Don't encode the stack")
          ("text_format", "serialize models in text")
          ("factored_ensemble_beam", "do beam search in each model in the ensemble separately, then take the union and rescore with the entire ensemble")
          ("ptb_output_file", po::value<string>(), "When outputting parses, use original POS tags and non-unk'ed words")
          ("models", po::value<vector<string>>()->multitoken(), "Load ensemble of saved models from these files")
          ("combine_type", po::value<string>(), "Decision-level combination type for ensemble (sum or product)")
          ("block_count", po::value<unsigned>()->default_value(0), "divide the dev set up into this many blocks and only decode one of them (indexed by block_num)")
          ("block_num", po::value<unsigned>()->default_value(0), "decode only this block (0-indexed), must be used with block_count")
          ("min_risk_training", "min risk training (default F1)")
          ("min_risk_method", po::value<string>()->default_value("reinforce"), "reinforce, beam, or beam_unnormalized")
          ("min_risk_include_gold", "use the true parse in the gradient updates")
          ("min_risk_candidates", po::value<unsigned>()->default_value(10), "min risk number of candidates")
          ("label_smoothing_epsilon", po::value<float>()->default_value(0.0f), "use epsilon interpolation with the uniform distribution in label smoothing")
          ("model_output_file", po::value<string>())
          ("optimizer", po::value<string>()->default_value("sgd"))
          ("dynamic_exploration", po::value<string>(), "if passed, should be greedy | sample")
          ("dynamic_exploration_probability", po::value<float>()->default_value(1.0), "with this probability, use the model probabilities to explore (with method given by --dynamic_exploration)")
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

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Discriminative(unsigned action, char prev_a, unsigned bsize, unsigned ssize, unsigned nopen_parens, unsigned ncons_nt) {
    bool is_shift = action == SHIFT_ACTION;
    bool is_reduce = action == REDUCE_ACTION;
    bool is_nt = !(is_shift | is_reduce);

    static const unsigned MAX_OPEN_NTS = 100;
    if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
    if (is_nt && ncons_nt >= MAX_CONS_NT) return true;
    if (ssize == 1) {
        if (!is_nt) return true;
        return false;
    }

    if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        // if a SHIFT has an implicit REDUCE, then only shift after an NT:
        if (is_shift && prev_a != 'N') return true;
    }

    // be careful with top-level parens- you can only close them if you
    // have fully processed the buffer
    if (nopen_parens == 1 && bsize > 1) {
        if (IMPLICIT_REDUCE_AFTER_SHIFT && is_shift) return true;
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
  virtual std::shared_ptr<AbstractParserState> new_sentence(ComputationGraph* hg, const parser::Sentence& sent, bool is_evaluation, bool build_training_graph, bool apply_dropout) = 0;

  pair<vector<unsigned>, Expression> abstract_log_prob_parser(
      ComputationGraph* hg,
      const parser::Sentence& sent,
      const vector<int>& correct_actions,
      double *right,
      bool is_evaluation,
      bool sample = false,
      bool label_smoothing = false,
      float label_smoothing_epsilon = 0.0,
      DynamicOracle* dynamic_oracle = nullptr
  ) {
    // can't have both correct actions and an oracle
    assert(correct_actions.empty() || !dynamic_oracle);
    bool build_training_graph = !correct_actions.empty() || dynamic_oracle;
    bool apply_dropout = (DROPOUT && !is_evaluation);

    if (label_smoothing) {
      assert(build_training_graph);
    }

    std::shared_ptr<AbstractParserState> state = new_sentence(hg, sent, is_evaluation, build_training_graph, apply_dropout);

    unsigned action_count = 0;  // incremented at each prediction

    vector<Expression> log_probs;
    vector<unsigned> results;

    while(!state->is_finished()) {
      vector<unsigned> valid_actions = state->get_valid_actions();

      Expression adiste = state->get_action_log_probs(valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());

      unsigned model_action = valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(valid_actions.size() > 0);
        vector<float> dist_to_sample;
        if (ALPHA != 1.0f) {
          // Expression r_t_smoothed = r_t * ALPHA;
          // Expression adiste_smoothed = log_softmax(r_t_smoothed, current_valid_actions);
          Expression adiste_smoothed = log_softmax(adiste * ALPHA, valid_actions);
          dist_to_sample = as_vector(hg->incremental_forward());
        } else {
          dist_to_sample = adist;
        }
        unsigned w = 0;
        for (; w < valid_actions.size(); ++w) {
          p -= exp(dist_to_sample[valid_actions[w]]);
          if (p < 0.0) break;
        }
        if (w == valid_actions.size()) w--;
        model_action = valid_actions[w];
      } else { // max
        double best_score = adist[valid_actions[0]];
        for (unsigned i = 1; i < valid_actions.size(); ++i) {
          if (adist[valid_actions[i]] > best_score) {
            best_score = adist[valid_actions[i]];
            model_action = valid_actions[i];
          }
        }
      }

      unsigned action_taken;
      if (build_training_graph) {  // if we have reference actions or an oracle (for training)
        unsigned correct_action;
        if (!correct_actions.empty()) {
          if (action_count >= correct_actions.size()) {
            cerr << "Correct action list exhausted, but not in final parser state.\n";
            abort();
          }
          correct_action = correct_actions[action_count];
        } else {
          correct_action = dynamic_oracle->oracle_action(*state);
        }
        if (model_action == correct_action) { (*right)++; }
        if (label_smoothing) {
          Expression cross_entropy = pick(adiste, correct_action) * input(*hg, (1 - label_smoothing_epsilon));
          // add uniform cross entropy
          for (unsigned a: valid_actions) {
            cross_entropy = cross_entropy + pick(adiste, a) * input(*hg, label_smoothing_epsilon / valid_actions.size());
          }
          log_probs.push_back(cross_entropy);
        } else {
          log_probs.push_back(pick(adiste, correct_action));
        }
        if (dynamic_oracle && rand01() < DYNAMIC_EXPLORATION_PROBABILITY) {
          action_taken = model_action;
        } else {
          action_taken = correct_action;
        }
      } else {
        log_probs.push_back(pick(adiste, model_action));
        action_taken = model_action;
      }

      ++action_count;
      results.push_back(action_taken);
      state = state->perform_action(action_taken);
    }

    if (!correct_actions.empty() && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }

    state->finish_sentence();

    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return pair<vector<unsigned>, Expression>(results, tot_neglogprob);
  }

  virtual vector<pair<vector<unsigned>, Expression>> abstract_log_prob_parser_beam(
      ComputationGraph* hg,
      const parser::Sentence& sent,
      unsigned beam_size,
      bool is_evaluation = true
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

    std::shared_ptr<AbstractParserState> initial_state = new_sentence(hg, sent, is_evaluation, build_training_graph, apply_dropout);

    vector<Expression> log_probs;
    vector<unsigned> results;

    vector<Stack<BeamItem>> completed;
    vector<Stack<BeamItem>> beam;

    beam.push_back(Stack<BeamItem>(BeamItem(initial_state, START_OF_SENTENCE_ACTION, input(*hg, 0.0))));

    unsigned action_count = 0;
    while (completed.size() < beam_size && !beam.empty()) {
      action_count += 1;
      // old beam item, action to be applied, resulting total score
      vector<std::tuple<Stack<BeamItem>, unsigned, Expression>> successors;

      while (!beam.empty()) {
        const Stack<BeamItem>& current_stack_item = beam.back();
        beam.pop_back();

        std::shared_ptr<AbstractParserState> current_parser_state = current_stack_item.back().state;
        vector<unsigned> valid_actions = current_parser_state->get_valid_actions();
        Expression adiste = current_parser_state->get_action_log_probs(valid_actions);
        vector<float> adist = as_vector(hg->incremental_forward());

        for (unsigned action: valid_actions) {
          Expression action_score = pick(adiste, action);
          Expression total_score = current_stack_item.back().score + action_score;
          successors.push_back(
                  std::tuple<Stack<BeamItem>, unsigned, Expression>(current_stack_item, action, total_score)
          );
        }
      }

      unsigned num_pruned_successors = std::min(beam_size, static_cast<unsigned>(successors.size()));
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
          unsigned beam_size,
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

    std::shared_ptr<AbstractParserState> initial_state = new_sentence(hg, sent, is_evaluation, build_training_graph, apply_dropout);

    vector<unsigned> results;

    vector<Stack<BeamItem>> completed;
    vector<Stack<BeamItem>> beam;

    beam.push_back(Stack<BeamItem>(BeamItem(initial_state, START_OF_SENTENCE_ACTION, input(*hg, 0.0))));

    for (unsigned current_termc = 0; current_termc < sent.size(); current_termc++) {
      completed.clear();
      while (completed.size() < beam_size && !beam.empty()) {
        // old beam item, action to be applied, resulting total score
        vector<std::tuple<Stack<BeamItem>, unsigned, Expression>> successors;

        while (!beam.empty()) {
          const Stack<BeamItem>& current_stack_item = beam.back();
          beam.pop_back();

          std::shared_ptr<AbstractParserState> current_parser_state = current_stack_item.back().state;
          vector<unsigned> valid_actions = current_parser_state->get_valid_actions();
          Expression adiste = current_parser_state->get_action_log_probs(valid_actions);
          vector<float> adist = as_vector(hg->incremental_forward());

          for (unsigned action: valid_actions) {
            Expression action_score = pick(adiste, action);
            Expression total_score = current_stack_item.back().score + action_score;
            successors.push_back(
                    std::tuple<Stack<BeamItem>, unsigned, Expression>(current_stack_item, action, total_score)
            );
          }
        }

        unsigned num_pruned_successors = std::min(beam_size, static_cast<unsigned>(successors.size()));
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
      unsigned num_pruned_completion = std::min(current_termc < sent.size() - 1 ? beam_filter_at_word_size : beam_size, static_cast<unsigned>(completed.size()));
      std::copy(completed.begin(), completed.begin() + std::min(num_pruned_completion, (unsigned) completed.size()), std::back_inserter(beam));
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
  for (unsigned i = 0; i < sent.size(); ++i) {
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
          words_shifted
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
  Parameters* p_ptbias; // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_ptW;    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
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

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})), // word embeddings
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})), // pretrained word embeddings (not updated)
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})), // nonterminal embeddings
      p_ntup(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})), // nonterminal embeddings when used in a composed representation
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),

      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})) {
    if (IMPLICIT_REDUCE_AFTER_SHIFT) {
      p_ptbias = model->add_parameters({LSTM_INPUT_DIM}); // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
      p_ptW = model->add_parameters({LSTM_INPUT_DIM, 2*LSTM_INPUT_DIM});    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
    }
    if (USE_POS) {
      p_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2w = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    buffer_lstm = new LSTMBuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
    }
  }

  // instance variables for each sentence
  ComputationGraph* hg;
  bool apply_dropout;
  Expression pbias, S, B, A, ptbias, ptW, p2w, ib, cbias, w2l, t2l, p2a, abias, action_start, cW;

  std::shared_ptr<AbstractParserState> new_sentence(ComputationGraph* hg, const parser::Sentence& sent, bool is_evaluation, bool build_training_graph, bool apply_dropout) override {
    this->hg = hg;
    this->apply_dropout = apply_dropout;

    if (!NO_STACK) stack_lstm.new_graph(*hg);
    buffer_lstm->new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);

    if (!NO_STACK) stack_lstm.start_new_sequence();
    buffer_lstm->start_new_sequence();
    action_lstm.start_new_sequence();

    if (apply_dropout) {
      if (!NO_STACK) stack_lstm.set_dropout(DROPOUT);
      buffer_lstm->set_dropout(DROPOUT);
      action_lstm.set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      if (!NO_STACK) stack_lstm.disable_dropout();
      buffer_lstm->disable_dropout();
      action_lstm.disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }

    // variables in the computation graph representing the parameters
    pbias = parameter(*hg, p_pbias);
    S = parameter(*hg, p_S);
    B = parameter(*hg, p_B);
    A = parameter(*hg, p_A);
    if (IMPLICIT_REDUCE_AFTER_SHIFT) {
      ptbias = parameter(*hg, p_ptbias);
      ptW = parameter(*hg, p_ptW);
    }
    if (USE_POS) p2w = parameter(*hg, p_p2w);

    ib = parameter(*hg, p_ib);
    cbias = parameter(*hg, p_cbias);
    w2l = parameter(*hg, p_w2l);
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    p2a = parameter(*hg, p_p2a);
    abias = parameter(*hg, p_abias);
    action_start = parameter(*hg, p_action_start);
    cW = parameter(*hg, p_cW);

    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i]; // this will be equal to unk at dev/test
      if (build_training_graph && !is_evaluation && singletons.size() > wordid && singletons[wordid] && rand01() > 0.5)
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
      buffer[sent.size() - i] = rectify(affine_transform(args));
    }

    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    for (auto& b : buffer)
      buffer_lstm->add_input(b);

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
            buffer_lstm->state(),
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

  std::shared_ptr<AbstractParserState> new_sentence(ComputationGraph* hg, const parser::Sentence& sent, bool is_evaluation, bool build_training_graph, bool apply_dropout) override {
    vector<std::shared_ptr<AbstractParserState>> states;
    for (const std::shared_ptr<ParserBuilder>& parser : parsers)
      states.push_back(parser->new_sentence(hg, sent, is_evaluation, build_training_graph, apply_dropout));
    return std::static_pointer_cast<AbstractParserState>(std::make_shared<EnsembledParserState>(this, states));
  }

  vector<pair<vector<unsigned>, Expression>> abstract_log_prob_parser_beam(
          ComputationGraph* hg,
          const parser::Sentence& sent,
          unsigned beam_size
  ) {
    if (factored_beam) {
      set<vector<unsigned>> all_candidates;
      for (const std::shared_ptr<ParserBuilder>& parser : parsers) {
        auto this_beam = parser->abstract_log_prob_parser_beam(hg, sent, beam_size);
        for (auto results : this_beam) {
          all_candidates.insert(results.first);
        }
      }
      vector<pair<vector<unsigned>, Expression>> candidates_and_nlps;
      for (vector<unsigned> candidate : all_candidates) {
        ComputationGraph hg;
        double right;
        auto candidate_and_ensemble_nlp = abstract_log_prob_parser(&hg, sent, vector<int>(candidate.begin(), candidate.end()), &right, true, false);
        candidates_and_nlps.push_back(candidate_and_ensemble_nlp);
      }
      sort(candidates_and_nlps.begin(), candidates_and_nlps.end(), [](const std::pair<vector<unsigned>, Expression>& t1, const std::pair<vector<unsigned>, Expression>& t2) {
        return as_scalar(t1.second.value()) < as_scalar(t2.second.value()); // sort by ascending nlp
      });
      while (candidates_and_nlps.size() > beam_size)
        candidates_and_nlps.pop_back();
      return candidates_and_nlps;

    } else {
      return AbstractParser::abstract_log_prob_parser_beam(hg, sent, beam_size);
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
          unsigned words_shifted
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
          words_shifted(words_shifted)
  {};

  const Stack<int> bufferi;
  const Stack<int> stacki;

  bool is_finished() const override {
    return stacki.size() == 2 && bufferi.size() == 1;
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

  bool action_is_valid(unsigned action) const override {
    return not IsActionForbidden_Discriminative(action, prev_a, bufferi.size(), stacki.size(), nopen_parens, cons_nt_count);
  }

  vector<unsigned> get_valid_actions() const override {
    vector<unsigned> valid_actions;
    for (auto a: possible_actions) {
      if (action_is_valid(a))
        valid_actions.push_back(a);
    }
    return valid_actions;
  }

  Expression get_action_log_probs(const vector<unsigned>& valid_actions) const override {
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

    bool was_word_completed = false;

    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        assert(false); // for purposes of word-level beam debugging
        --new_nopen_parens;
        assert(is_open_paren.back() >= 0);

        new_stacki = new_stacki.pop_back();
        new_bufferi = new_bufferi.pop_back();
        new_is_open_paren = new_is_open_paren.pop_back();
        new_stacki = new_stacki.push_back(999);
        new_is_open_paren = new_is_open_paren.push_back(-1);
      } else {
        new_stacki = new_stacki.push_back(bufferi.back());
        new_bufferi = new_bufferi.pop_back();
        new_is_open_paren = new_is_open_paren.push_back(-1);
        was_word_completed = (new_bufferi.size() > 1);
      }
      new_words_shifted += 1;
      new_cons_nt_count = 0;
    }

    else if (ac == 'N') { // NT
      ++new_nopen_parens;
      assert(bufferi.size() > 1);
      auto it = action2NTindex.find(action);
      assert(it != action2NTindex.end());
      int nt_index = it->second;
      new_nt_count++;
      new_stacki = new_stacki.push_back(-1);
      new_is_open_paren = new_is_open_paren.push_back(nt_index);
      new_cons_nt_count += 1;
      new_open_brackets = new_open_brackets.push_back(OpenBracket(nt_index, words_shifted));
    }

    else { // REDUCE
      --new_nopen_parens;
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
      assert(nchildren > 0);
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
      new_stacki = new_stacki.push_back(999); // who knows, should get rid of this
      new_is_open_paren = new_is_open_paren.push_back(-1); // we just closed a paren at this position
      if (new_stacki.size() <= 2) {
        was_word_completed = true;
      }
      new_cons_nt_count = 0;
      new_completed_brackets = new_completed_brackets.push_back(close_bracket(new_open_brackets.back(), words_shifted));
      new_open_brackets = new_open_brackets.pop_back();
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
            new_words_shifted
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
    return stack.size() == 2 && buffer.size() == 1;
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

  bool action_is_valid(unsigned action) const override {
    return symbolic_parser_state->action_is_valid(action);
  }

  vector<unsigned> get_valid_actions() const override {
    return symbolic_parser_state->get_valid_actions();
  }

  Expression get_action_log_probs(const vector<unsigned>& valid_actions) const override {
    Expression stack_summary = NO_STACK ? Expression() : parser->stack_lstm.get_h(stack_state).back();
    Expression action_summary = parser->action_lstm.get_h(action_state).back();
    Expression buffer_summary = parser->buffer_lstm->get_h(buffer_state).back();
    if (parser->apply_dropout) { // TODO: don't the outputs of the LSTMs already have dropout applied?
      if (!NO_STACK) stack_summary = dropout(stack_summary, DROPOUT);
      action_summary = dropout(action_summary, DROPOUT);
      buffer_summary = dropout(buffer_summary, DROPOUT);
    }
    Expression p_t = NO_STACK ?
                     affine_transform({parser->pbias, parser->B, buffer_summary, parser->A, action_summary}) :
                     affine_transform({parser->pbias, parser->S, stack_summary, parser->B, buffer_summary, parser->A, action_summary});
    Expression nlp_t = rectify(p_t);
    Expression r_t = affine_transform({parser->abias, parser->p2a, nlp_t});
    return log_softmax(r_t, valid_actions);
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
    parser->action_lstm.add_input(action_state, actione);
    new_action_state = parser->action_lstm.state();

    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
          assert(false); // for purposes of word-level beam debugging
        Expression nonterminal = lookup(*parser->hg, parser->p_ntup, is_open_paren.back());
        Expression terminal = buffer.back();
        Expression c = concatenate({nonterminal, terminal});
        Expression pt = rectify(affine_transform({parser->ptbias, parser->ptW, c}));
        new_stack = new_stack.pop_back();
        if (!NO_STACK) new_stack_state = parser->stack_lstm.head_of(stack_state);
        new_buffer = new_buffer.pop_back();
        new_buffer_state = parser->buffer_lstm->head_of(buffer_state);
        new_is_open_paren = new_is_open_paren.pop_back();
        if (!NO_STACK) {
          parser->stack_lstm.add_input(new_stack_state, pt);
          new_stack_state = parser->stack_lstm.state();
        }
        new_stack = new_stack.push_back(pt);
        new_is_open_paren = new_is_open_paren.push_back(-1);
      } else {
        new_stack = new_stack.push_back(buffer.back());
        if (!NO_STACK) {
          parser->stack_lstm.add_input(stack_state, buffer.back());
          new_stack_state = parser->stack_lstm.state();
        }
        new_buffer = new_buffer.pop_back();
        new_buffer_state = parser->buffer_lstm->head_of(buffer_state);
        new_is_open_paren = new_is_open_paren.push_back(-1);
      }
    }

    else if (ac == 'N') { // NT
      assert(buffer.size() > 1);
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
    }

    else { // REDUCE
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
      assert(nchildren > 0);
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

  Expression get_action_log_probs(const vector<unsigned>& valid_actions) const override {
    vector<Expression> all_log_probs;
    for (const std::shared_ptr<AbstractParserState>& state : states)
      all_log_probs.push_back(state->get_action_log_probs(valid_actions));
    Expression combined_log_probs;
    switch (parser->combine_type) {
      case EnsembledParser::CombineType::sum: {
        // combined_log_probs = logsumexp(all_log_probs); // numerically unstable
        combined_log_probs = logsumexp_stable(all_log_probs);
        break;
      }
      case EnsembledParser::CombineType::product:
        combined_log_probs = sum(all_log_probs);
        break;
    }
    return log_softmax(combined_log_probs, valid_actions);
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

void print_parse(const vector<unsigned>& actions, const parser::Sentence& sentence, bool ptb_output_format, ostream& out_stream) {
  int ti = 0;
  for (auto a : actions) {
    if (adict.Convert(a)[0] == 'N') {
      out_stream << " (" << ntermdict.Convert(action2NTindex.find(a)->second);
    } else if (adict.Convert(a)[0] == 'S') {
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        out_stream << termdict.Convert(sentence.raw[ti++]) << ")";
      } else {
        if (ptb_output_format) {
          string preterminal = posdict.Convert(sentence.pos[ti]);
          out_stream << " (" << preterminal << ' ' << non_unked_termdict.Convert(sentence.non_unked_raw[ti]) << ")";
          ti++;
        } else { // use this branch to surpress preterminals
          out_stream << ' ' << termdict.Convert(sentence.raw[ti++]);
        }
      }
    } else out_stream << ')';
  }
  out_stream << endl;
}

void print_parse(const vector<int>& actions, const parser::Sentence& sentence, bool ptb_output_format, ostream& out_stream) {
  for (auto action : actions)
    assert(action >= 0);
  print_parse(vector<unsigned>(actions.begin(), actions.end()), sentence, ptb_output_format, out_stream);
}

Tree to_tree(const vector<int>& actions, const parser::Sentence& sentence) {
  vector<string> linearized;
  unsigned ti = 0;
  for (int a: actions) {
    string token;
    if (adict.Convert(a)[0] == 'N') {
      linearized.push_back("(" + ntermdict.Convert(action2NTindex.find(a)->second));
    }
    else if (adict.Convert(a)[0] == 'S') {
      linearized.push_back(posdict.Convert(sentence.pos[ti++]));
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        linearized.push_back(")");
      }
    } else {
      linearized.push_back(")");
    }
  }
  /*
for (auto& sym: linearized) cout << sym << " ";
cout << endl;
   */
  return parse_linearized(linearized);
}

int main(int argc, char** argv) {
  unsigned random_seed = cnn::Initialize(argc, argv);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  IMPLICIT_REDUCE_AFTER_SHIFT = conf.count("explicit_terminal_reduce") == 0;
  USE_POS = conf.count("use_pos_tags");
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  USE_PRETRAINED = conf.count("words");
  NO_STACK = conf.count("no_stack");

  MAX_CONS_NT = conf["max_cons_nt"].as<unsigned>();

  SILVER_BLOCKS_PER_GOLD = conf["silver_blocks_per_gold"].as<unsigned>();

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

  if (conf.count("model") && conf.count("models")) {
    cerr << "Cannot specify --model and --models." << endl;
    return 1;
  }

  if (conf.count("alpha")) {
    ALPHA = conf["alpha"].as<float>();
    if (ALPHA <= 0.f) { cerr << "--alpha must be between 0 and +infty\n"; abort(); }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<unsigned>();
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

  ostringstream os;
  os << "ntparse"
     << (USE_POS ? "_pos" : "")
     << (USE_PRETRAINED ? "_pretrained" : "")
     << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << (NO_STACK ? "_no-stack" : "")
     << "-seed" << random_seed
     << "-pid" << getpid() << ".params";

  const string fname = conf.count("model_output_file") > 0 ? conf["model_output_file"].as<string>() : os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  //bool softlinkCreated = false;

  bool discard_train_sents = (conf.count("train") == 0);

  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracle gold_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>(), true, discard_train_sents);
  corpus.load_bdata(conf["bracketing_dev_data"].as<string>());

  bool has_gold_training_data = false;

  if (conf.count("gold_training_data")) {
    gold_corpus.load_oracle(conf["gold_training_data"].as<string>(), true, discard_train_sents);
    gold_corpus.load_bdata(conf["bracketing_dev_data"].as<string>());
    has_gold_training_data = true;
  }

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  termdict.SetUnk("UNK"); // we don't actually expect to use this often
     // since the Oracles are required to be "pre-UNKified", but this prevents
     // problems with UNKifying the lowercased data which needs to be loaded
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze();

  SHIFT_ACTION = adict.Convert("SHIFT");
  REDUCE_ACTION = adict.Convert("REDUCE");

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
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), false, false);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>(), false, false);
  }

  non_unked_termdict.Freeze();

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
    ntIndex2Action[nt] = i;
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  bool factored_ensemble_beam = conf.count("factored_ensemble_beam") > 0;


  bool sample = conf.count("samples") > 0;
  bool output_beam_as_samples = conf.count("output_beam_as_samples") > 0;
  bool output_candidate_trees = output_beam_as_samples || conf.count("samples_include_gold") || sample;

  unsigned beam_size = conf["beam_size"].as<unsigned>();
  unsigned test_size = test_corpus.size();

  unsigned start_index = 0;
  unsigned stop_index = test_corpus.size();
  unsigned block_count = conf["block_count"].as<unsigned>();
  unsigned block_num = conf["block_num"].as<unsigned>();


  auto decode = [&](AbstractParser& parser, const parser::Sentence& sentence) {
      double right = 0;
      // get parse and negative log probability
      ComputationGraph hg;
      if (beam_size > 1 || conf.count("beam_within_word")) {
        vector<pair<vector<unsigned>, Expression>> beam_results;
        if (conf.count("beam_within_word")) {
          beam_results = parser.abstract_log_prob_parser_beam_within_word(&hg, sentence,
                                                                          beam_size,
                                                                          conf["beam_filter_at_word_size"].as<int>());
        } else {
          beam_results = parser.abstract_log_prob_parser_beam(&hg, sentence, beam_size);
        }
        return beam_results[0];
      } else {
        return parser.abstract_log_prob_parser(&hg, sentence, vector<int>(), &right, true);
      }
  };

  auto evaluate = [&](const vector<parser::Sentence>& sentences, const vector<vector<int>>& gold_parses, const vector<vector<unsigned>>& pred_parses, const string& name) {
      auto make_name = [&](const string& base) {
          ostringstream os;
          os << "/tmp/parser_" << base << "." << getpid();
          if (name != "")
            os << "." << name;
          os << ".txt";
          return os.str();
      };
      unsigned size = sentences.size();

      assert(pred_parses.size() == size);
      assert(gold_parses.size() == size);

      const string pred_fname = make_name("pred");
      ofstream pred_out(pred_fname.c_str());

      const string gold_fname = make_name("gold");
      ofstream gold_out(gold_fname.c_str());

      const string evalb_fname = make_name("evalb");

      MatchCounts match_counts;
      vector<MatchCounts> all_match_counts;

      for (unsigned sii = 0; sii < size; sii++) {
        const auto& sentence = sentences[sii];
        const auto& pred_parse = pred_parses[sii];
        const auto& gold_parse = gold_parses[sii];
        Tree pred_tree = to_tree(vector<int>(pred_parse.begin(), pred_parse.end()), sentence);
        Tree gold_tree = to_tree(gold_parse, sentence);

        MatchCounts this_counts = pred_tree.compare(gold_tree, true);
        all_match_counts.push_back(this_counts);
        match_counts += this_counts;

        print_parse(pred_parse, sentence, true, pred_out);
        print_parse(gold_parse, sentence, true, gold_out);
      }

      pred_out.close();
      gold_out.close();
      cerr << name << " output in " << pred_fname << endl;

      pair<Metrics, vector<MatchCounts>> results = metrics_from_evalb(gold_fname, pred_fname, evalb_fname);
      //pair<Metrics, vector<MatchCounts>> corpus_results = metrics_from_evalb(corpus.devdata, pred_fname, evalb_fname + "_corpus");

      if (abs(match_counts.metrics().f1 - results.first.f1) > 1e-2) {
        cerr << "warning: score mismatch" << endl;
        cerr << "computed\trecall=" << match_counts.metrics().recall << ", precision=" << match_counts.metrics().precision << ", F1=" << match_counts.metrics().f1 << "\n";
        cerr << "evalb\trecall=" << results.first.recall << ", precision=" << results.first.precision << ", F1=" << results.first.f1 << "\n";
      }
      //cerr << "evalb corpus\trecall=" << corpus_results.first.recall << ", precision=" << corpus_results.first.precision << ", F1=" << corpus_results.first.f1 << "\n";

      /*
      for (unsigned i = 0; i < all_match_counts.size(); i++) {
        if (all_match_counts[i] != results.second[i]) {
          cout << "mismatch for " << (i+1) << endl;
          cout << all_match_counts[i].correct << " " << all_match_counts[i].gold << " " << all_match_counts[i].predicted << endl;
          cout << results.second[i].correct << " " << results.second[i].gold << " " << results.second[i].predicted << endl;
          Tree pred_tree = to_tree(vector<int>(pred_parses[i].begin(), pred_parses[i].end()), sentences[i]);
          Tree gold_tree = to_tree(gold_parses[i], sentences[i]);
          pred_tree.compare(gold_tree, true, true);
        }
      }
      */
      return results.first;
  };


  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);

    string optimizer_name = conf["optimizer"].as<string>();
    assert(optimizer_name == "sgd" || optimizer_name == "adam");


    Model model;
    ParserBuilder parser(&model, pretrained);
    unique_ptr<Trainer> optimizer = optimizer_name == "sgd" ? unique_ptr<Trainer>(new SimpleSGDTrainer(&model)) : unique_ptr<Trainer>(new AdamTrainer(&model)); //(&model);

    if (optimizer_name == "sgd") {
      optimizer->eta_decay = 0.05;
    }


    if (conf.count("model")) {
      cerr << "before load model" << endl;
      ifstream in(conf["model"].as<string>().c_str());
      if (conf.count("text_format")) {
        boost::archive::text_iarchive ia(in);
        ia >> model >> *optimizer;
        //ia >> model;
      } else {
        boost::archive::binary_iarchive ia(in);
        ia >> model >> *optimizer;
        //ia >> model;
      }
      cerr << "after load model" << endl;
    } else {
      cerr << "using " << optimizer_name << " for training" << endl;
    }

    //AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;

    unsigned tot_seen = 0;
    int iter = -1;

    double best_dev_err = 9e99;
    double bestf1=0.0;

    bool min_risk_training = conf.count("min_risk_training") > 0;
    string min_risk_method = conf["min_risk_method"].as<string>();
    unsigned min_risk_candidates = conf["min_risk_candidates"].as<unsigned>();
    bool min_risk_include_gold = conf.count("min_risk_include_gold") > 0;
    assert(min_risk_candidates > 0);

    float label_smoothing_epsilon = conf["label_smoothing_epsilon"].as<float>();
    bool label_smoothing = (label_smoothing_epsilon != 0.0f);
    if (label_smoothing) {
      assert(!min_risk_training);
      assert(label_smoothing_epsilon > 0);
    }

    //vector<double> sampled_f1s;
    double total_f1s = 0.0;
    double total_standardized_f1s = 0.0;
    double m2_f1 = 0.0;
    double mean_f1 = 0.0;
    double std_f1 = 0.0;
    unsigned num_samples = 0;

    auto standardize_and_update_f1 = [&](double scaled_f1)  {
        num_samples++;
        total_f1s += scaled_f1;
        double delta = scaled_f1 - mean_f1;
        mean_f1 += delta / num_samples;
        m2_f1 += delta * (scaled_f1 - mean_f1);
        std_f1 = sqrt(num_samples > 1 ? m2_f1 / (num_samples - 1) : 1.0);
        double standardized_f1 = std_f1 > 0 ? (scaled_f1 - mean_f1) / std_f1 : 0;
        total_standardized_f1s += standardized_f1;
        return standardized_f1;
    };

    auto train_sentence = [&](const parser::Sentence& sentence, const vector<int>& actions, double* right) -> pair<double, MatchCounts> {
        ComputationGraph hg;
        double loss_v;

        MatchCounts sentence_match_counts;

        auto get_f1_and_update_mc = [&](Tree& gold_tree, const parser::Sentence& sentence, const vector<unsigned> actions) {
            Tree pred_tree = to_tree(vector<int>(actions.begin(), actions.end()), sentence);
            MatchCounts match_counts = pred_tree.compare(gold_tree, true);
            sentence_match_counts += match_counts;
            return (float) match_counts.metrics().f1 / 100.0f;
        };

        Tree gold_tree = to_tree(actions, sentence);
        if (conf.count("min_risk_training")) {
          if (min_risk_method == "reinforce") {
            Expression loss = input(hg, 0.0);
            /*
            cerr << "gold ";
            print_parse(vector<unsigned>(actions.begin(), actions.end()), sentence, true, cerr);
            cerr << endl;
            */
            for (int i = 0; i < min_risk_candidates; i++) {
              double blank;
              pair<vector<unsigned>, Expression> sample_and_nlp;
              if (min_risk_include_gold && i == 0) {
                sample_and_nlp = parser.abstract_log_prob_parser(&hg, sentence, actions, &blank, false, false);
              } else {
                sample_and_nlp = parser.abstract_log_prob_parser(&hg, sentence, vector<int>(), &blank, false, true);
              }
              /*
              cerr << " " << i;
              print_parse(sample_and_nlp.first, sentence, true, cerr);
              cerr << endl;
              */
              float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, sample_and_nlp.first);
              // Welford online variance, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
              double standardized_f1 = standardize_and_update_f1(scaled_f1);
              loss = loss + (sample_and_nlp.second * input(hg, standardized_f1));
            }
            loss = loss * input(hg, 1.0 / min_risk_candidates);
            loss_v = as_scalar(hg.incremental_forward());
          } else if (min_risk_method == "beam" || min_risk_method == "beam_noprobs" || min_risk_method == "beam_unnormalized" || min_risk_method == "beam_unnormalized_log") {
            auto candidates = parser.abstract_log_prob_parser_beam(&hg, sentence, min_risk_include_gold ? min_risk_candidates - 1: min_risk_candidates, false);

            if (min_risk_include_gold) {
              double blank;
              candidates.push_back(parser.abstract_log_prob_parser(&hg, sentence, actions, &blank, false, false));
            }

            if (min_risk_method == "beam") {
              // L_risk objective (Eq 4) from Edunov et al 2017: https://arxiv.org/pdf/1711.04956.pdf
              vector<Expression> log_probs_plus_log_losses;
              vector<Expression> log_probs;
              for (auto &parse_and_loss: candidates) {
                //Expression log_prob = -parse_and_loss.second;
                Expression normed_log_prob = -parse_and_loss.second - log(input(hg, parse_and_loss.first.size()));
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
            } else if (min_risk_method == "beam_noprobs") {
              Expression loss = input(hg, 0.0f);
              float normalizer = 0.0;
              for (auto &parse_and_loss: candidates) {
                float scaled_f1 = get_f1_and_update_mc(gold_tree, sentence, parse_and_loss.first);
                //scaled_f1 = standardize_and_update_f1(scaled_f1);
                loss = loss + (parse_and_loss.second * input(hg, scaled_f1));
              }
              loss = loss * input(hg, 1.0 / candidates.size());
              loss_v = as_scalar(hg.incremental_forward());
            } else {
              assert(min_risk_method == "beam_unnormalized" || min_risk_method == "beam_unnormalized_log");
              Expression loss = input(hg, 0.0f);
              float normalizer = 0.0;
              for (auto &parse_and_loss: candidates) {
                Expression log_prob = -parse_and_loss.second;
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
              loss_v = as_scalar(hg.incremental_forward());
            }
          } else {
            cerr << "invalid min_risk_method: " << min_risk_method << endl;
            exit(1);
          }
          hg.backward();
        } else {
          if (exploration_type == DynamicOracle::ExplorationType::none) {
            auto result_and_nlp = parser.abstract_log_prob_parser(&hg,
                                            sentence,
                                            actions,
                                            right,
                                            false, // is_evaluation
                                            false, //sample
                                            label_smoothing,
                                            label_smoothing_epsilon);
            get_f1_and_update_mc(gold_tree, sentence, result_and_nlp.first);
          } else {
            DynamicOracle dynamic_oracle(sentence, actions);
            auto result_and_nlp = parser.abstract_log_prob_parser(&hg,
                                            sentence,
                                            vector<int>(),
                                            right,
                                            false, // is_evaluation
                                            exploration_type == DynamicOracle::ExplorationType::sample, //sample
                                            label_smoothing,
                                            label_smoothing_epsilon,
                                            &dynamic_oracle);
            if (DYNAMIC_EXPLORATION_PROBABILITY == 0.0f) {
              assert(vector<int>(result_and_nlp.first.begin(),
                    result_and_nlp.first.end()) == actions);
            }
            get_f1_and_update_mc(gold_tree, sentence, result_and_nlp.first);
          }
          loss_v = as_scalar(hg.incremental_forward());
          hg.backward();
          //loss_v = as_scalar(result_and_nlp.second.value());
        }
        optimizer->update(1.0);
        return pair<double, MatchCounts>(loss_v, sentence_match_counts);
    };

    auto train_block = [&](const parser::TopDownOracle& corpus, vector<unsigned>::iterator indices_begin, vector<unsigned>::iterator indices_end, int epoch_size) {
      unsigned sentence_count = std::distance(indices_begin, indices_end);
      status_every_i_iterations = min(status_every_i_iterations, sentence_count);
      cerr << "Number of sentences in current block: " << sentence_count << endl;
      unsigned trs = 0;
      unsigned words = 0;
      double right = 0;
      double llh = 0;
      //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
      auto time_start = chrono::system_clock::now();

        MatchCounts block_match_counts;

      for (vector<unsigned>::iterator index_iter = indices_begin; index_iter != indices_end; ++index_iter) {
        tot_seen += 1;
        auto& sentence = corpus.sents[*index_iter];
        const vector<int>& actions=corpus.actions[*index_iter];
        {
          auto loss_and_mc = train_sentence(sentence, actions, &right);
          double loss = loss_and_mc.first;
          block_match_counts += loss_and_mc.second;
          if (!min_risk_training && loss < 0) {
            cerr << "loss < 0 on sentence " << *index_iter << ": loss=" << loss << endl;
            //assert(lp >= 0.0)
          }
          llh += loss;
        }
        trs += actions.size();
        words += sentence.size();

        if (tot_seen % status_every_i_iterations == 0) {
          ++iter;
          optimizer->status();
          auto time_now = chrono::system_clock::now();
          auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
          cerr << "update #" << iter << " (epoch " << (static_cast<double>(tot_seen) / epoch_size) <<
               /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
               ") per-action-ppl: " << exp(llh / trs) <<
               " per-input-ppl: " << exp(llh / words) <<
               " per-sent-ppl: " << exp(llh / status_every_i_iterations) <<
               " err: " << (trs - right) / trs <<
               " trace f1: " << block_match_counts.metrics().f1 / 100.f  <<
               " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]";

          if (min_risk_training) {
            //sampled_f1s.clear();
            cerr << " mean sampled f1: " << mean_f1 << " mean standardized f1:" << total_standardized_f1s / num_samples;
          }
          cerr << endl;
          llh = trs = right = words = 0;

          if (iter % 25 == 0) { // report on dev set
            unsigned dev_size = dev_corpus.size();
            double llh = 0;
            double trs = 0;
            double right = 0;
            double dwords = 0;
            auto t_start = chrono::high_resolution_clock::now();
            vector<vector<unsigned>> predicted;
            for (unsigned sii = 0; sii < dev_size; ++sii) {
              const auto& sentence=dev_corpus.sents[sii];
              const vector<int>& actions=dev_corpus.actions[sii];

//              cerr << "checking symbolic parser via actions_to_brackets" << endl;
              DynamicOracle oracle(dev_corpus.sents[sii], dev_corpus.actions[sii]);

              dwords += sentence.size();
              {
                ComputationGraph hg;
                parser.abstract_log_prob_parser(&hg, sentence, actions, &right, true);
                double lp = as_scalar(hg.incremental_forward());
                llh += lp;
              }
              //ComputationGraph hg;
              vector<unsigned> pred = decode(parser, sentence).first;
              predicted.push_back(pred);
              //vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
              trs += actions.size();
            }
            auto t_end = chrono::high_resolution_clock::now();
            double err = (trs - right) / trs;

            Metrics metrics = evaluate(dev_corpus.sents, dev_corpus.actions, predicted, "dev");
            cerr << "recall=" << metrics.recall << ", precision=" << metrics.precision << ", F1=" << metrics.f1 << "\n";
            cerr << "  **dev (iter=" << iter << " epoch=" << (static_cast<double>(tot_seen) / epoch_size) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << metrics.f1 << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
            if (metrics.f1>bestf1) {
              cerr << "  new best...writing model to " << fname << ".bin ...\n";
              best_dev_err = err;
              bestf1=metrics.f1;
              ofstream out(fname + ".bin");
              if (conf.count("text_format")) {
                  boost::archive::text_oarchive oa(out);
                  oa << model << *optimizer;
                  oa << termdict << adict << ntermdict << posdict;
              } else {
                  boost::archive::binary_oarchive oa(out);
                  oa << model << *optimizer;
                  oa << termdict << adict << ntermdict << posdict;
              }
              // system((string("cp ") + pfx + string(" ") + pfx + string(".best")).c_str());
              // Create a soft link to the most recent model in order to make it
              // easier to refer to it in a shell script.
              /*
              if (!softlinkCreated) {
                string softlink = " latest_model";
                if (system((string("rm -f ") + softlink).c_str()) == 0 &&
                    system((string("ln -s ") + fname + softlink).c_str()) == 0) {
                  cerr << "Created " << softlink << " as a soft link to " << fname
                       << " for convenience." << endl;
                }
                softlinkCreated = true;
              }
              */
            }
          }
          time_start = chrono::system_clock::now();
        }
      }

    };

    int epoch = 0;

    while (!requested_stop) {
      parser::TopDownOracle* main_corpus = &corpus;

      int sentence_count = 0;

      if (has_gold_training_data) {
        main_corpus = &gold_corpus;
        vector<unsigned> silver_indices(corpus.size());
        std::iota(silver_indices.begin(), silver_indices.end(), 0);
        std::random_shuffle(silver_indices.begin(), silver_indices.end());
        unsigned offset = std::min(corpus.size(), gold_corpus.size() * SILVER_BLOCKS_PER_GOLD);
        train_block(corpus, silver_indices.begin(), silver_indices.begin() + offset, offset + gold_corpus.size());
        sentence_count += offset;
      }

      vector<unsigned> main_indices(main_corpus->size());
      std::iota(main_indices.begin(), main_indices.end(), 0);
      std::random_shuffle(main_indices.begin(), main_indices.end());
      sentence_count += main_indices.size();
      train_block(*main_corpus, main_indices.begin(), main_indices.end(), sentence_count);

      optimizer->update_epoch();

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

      epoch++;
    }
  } // should do training?

  if (test_corpus.size() > 0) { // do inference for test evaluation
    vector<std::shared_ptr<Model>> models;
    vector<std::shared_ptr<ParserBuilder>> parsers;
    std::shared_ptr<EnsembledParser> ensembled_parser;

    AbstractParser* abstract_parser;

    if (conf.count("model")) {
      models.push_back(std::make_shared<Model>());
      parsers.push_back(std::make_shared<ParserBuilder>(models.back().get(), pretrained));
      string path(conf["model"].as<string>());
      cerr << "Loading single parser from " << path << "..." << endl;
      ifstream in(path);
      if (conf.count("text_format")) {
        boost::archive::text_iarchive ia(in);
        ia >> *models.back();
      } else {
        boost::archive::binary_iarchive ia(in);
        ia >> *models.back();
      }
      abstract_parser = parsers.back().get();
    }

    else {
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

    if (block_count > 0) {
      assert(block_num < block_count);
      unsigned q = test_corpus.size() / block_count;
      unsigned r = test_corpus.size() % block_count;
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
      vector<unsigned> n_distinct_samples;
      for (unsigned sii = start_index; sii < stop_index; ++sii) {
        const auto &sentence = test_corpus.sents[sii];
        // TODO: this overrides dynet random seed, but should be ok if we're only sampling
        cnn::rndeng->seed(sii);
        set<vector<unsigned>> samples;
        if (conf.count("samples_include_gold")) {
          ComputationGraph hg;
          auto result_and_nlp = abstract_parser->abstract_log_prob_parser(&hg, sentence, test_corpus.actions[sii],
                                                                              &right, true);
          vector<unsigned> result = result_and_nlp.first;
          double nlp = as_scalar(result_and_nlp.second.value());
          cout << sii << " ||| " << -nlp << " |||";
          vector<unsigned> converted_actions(test_corpus.actions[sii].begin(), test_corpus.actions[sii].end());
          print_parse(converted_actions, sentence, false, cout);
          ptb_out << sii << " ||| " << -nlp << " |||";
          print_parse(converted_actions, sentence, true, ptb_out);
          samples.insert(converted_actions);
        }

        for (unsigned z = 0; z < N_SAMPLES; ++z) {
          ComputationGraph hg;
          pair<vector<unsigned>, Expression> result_and_nlp = abstract_parser->abstract_log_prob_parser(&hg, sentence, actions, &right, sample,
                                                                              true); // TODO: fix ordering of sample and eval here
          double lp = as_scalar(result_and_nlp.second.value());
          cout << sii << " ||| " << -lp << " |||";
          print_parse(result_and_nlp.first, sentence, false, cout);
          ptb_out << sii << " ||| " << -lp << " |||";
          print_parse(result_and_nlp.first, sentence, true, ptb_out);
          samples.insert(result_and_nlp.first);
        }

        if (output_beam_as_samples) {
          ComputationGraph hg;
          vector<pair<vector<unsigned>, Expression>> beam_results;
          if (conf.count("beam_within_word")) {
            beam_results = abstract_parser->abstract_log_prob_parser_beam_within_word(&hg, sentence,
                                                                                      beam_size,
                                                                                      conf["beam_filter_at_word_size"].as<int>());
          } else {
            beam_results = abstract_parser->abstract_log_prob_parser_beam(&hg, sentence, beam_size);
          }
          if (beam_results.size() < beam_size) {
            cerr << "warning: only " << beam_results.size() << " parses found by beam search for sent " << sii << endl;
          }
          unsigned long num_results = beam_results.size();
          for (unsigned long i = 0; i < beam_size; i++) {
            unsigned long ix = std::min(i, num_results - 1);
            pair<vector<unsigned>, Expression> result_and_nlp = beam_results[ix];
            double nlp = as_scalar(result_and_nlp.second.value());
            cout << sii << " ||| " << -nlp << " |||";
            print_parse(result_and_nlp.first, sentence, false, cout);
            ptb_out << sii << " ||| " << -nlp << " |||";
            print_parse(result_and_nlp.first, sentence, true, ptb_out);
            samples.insert(result_and_nlp.first);
          }
        }

        n_distinct_samples.push_back(samples.size());
      }
      ptb_out.close();

      double avg_distinct_samples = accumulate(n_distinct_samples.begin(), n_distinct_samples.end(), 0.0) /
                                    (double) n_distinct_samples.size();
      cerr << "avg distinct samples: " << avg_distinct_samples << endl;
    }



    // shortcut: only do a test decode if we aren't outputting any candidate trees
    if (!output_candidate_trees) {
      auto t_start = chrono::high_resolution_clock::now();
      vector<vector<unsigned>> predicted;
      for (unsigned sii = 0; sii < test_size; ++sii) {
        const auto &sentence = test_corpus.sents[sii];
        pair<vector<unsigned>, Expression> result_and_nlp = decode(*abstract_parser, sentence);
        predicted.push_back(result_and_nlp.first);
      }
      auto t_end = chrono::high_resolution_clock::now();
      Metrics metrics = evaluate(test_corpus.sents, test_corpus.actions, predicted, "test");
      cerr << "recall=" << metrics.recall << ", precision=" << metrics.precision << ", F1=" << metrics.f1 << "\n";
    }
  }
}
