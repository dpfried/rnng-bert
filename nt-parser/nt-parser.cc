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

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
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

#include "nt-parser/oracle.h"
#include "nt-parser/pretrained.h"
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/eval.h"
#include "nt-parser/stack.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict, non_unked_termdict;

volatile bool requested_stop = false;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;

float ALPHA = 1.f;
unsigned N_SAMPLES = 0;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
unsigned POS_SIZE = 0;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X

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
        ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
        ("no_stack,S", "Don't encode the stack")
        ("ptb_output_file", po::value<string>(), "When outputting parses, use original POS tags and non-unk'ed words")
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

struct AbstractParserState {
  virtual bool is_finished() = 0;
  virtual vector<unsigned> get_valid_actions() = 0;
  virtual Expression get_action_log_probs(const vector<unsigned>& valid_actions) = 0;
  virtual std::shared_ptr<AbstractParserState> perform_action(const unsigned action) = 0;
  virtual void finish_sentence() = 0;
};

struct AbstractParser {
  virtual std::shared_ptr<AbstractParserState> new_sentence(ComputationGraph* hg, const parser::Sentence& sent, bool is_evaluation, bool build_training_graph, bool apply_dropout) = 0;

  vector<unsigned> abstract_log_prob_parser(
      ComputationGraph* hg,
      const parser::Sentence& sent,
      const vector<int>& correct_actions,
      double *right,
      bool is_evaluation,
      bool sample = false
  ) {
    bool build_training_graph = correct_actions.size() > 0;
    bool apply_dropout = (DROPOUT && !is_evaluation);

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

      unsigned action = model_action;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        if (model_action == action) { (*right)++; }
      }

      ++action_count;
      log_probs.push_back(pick(adiste, action));
      results.push_back(action);

      state = state->perform_action(action);
    }

    if (build_training_graph && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }

    state->finish_sentence();

    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return results;
  }

  vector<pair<vector<unsigned>, double>> abstract_log_prob_parser_beam(
      ComputationGraph* hg,
      const parser::Sentence& sent,
      unsigned beam_size
  ) {
    struct BeamItem {
      std::shared_ptr<AbstractParserState> state;
      double score;
      bool operator<(const BeamItem &other) const {
        return score < other.score;
      }
    };

    bool is_evaluation = false;
    bool build_training_graph = false;
    bool apply_dropout = false;

    std::shared_ptr<AbstractParserState> initial_state = new_sentence(hg, sent, is_evaluation, build_training_graph, apply_dropout);

    throw std::runtime_error("Not yet implemented");
  }
};

struct ParserState;

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

  // checks to see if a proposed action is valid in discriminative models
  static bool IsActionForbidden_Discriminative(const string& a, char prev_a, unsigned bsize, unsigned ssize, unsigned nopen_parens) {
    bool is_shift = (a[0] == 'S' && a[1]=='H');
    bool is_reduce = (a[0] == 'R' && a[1]=='E');
    bool is_nt = (a[0] == 'N');
    assert(is_shift || is_reduce || is_nt);
    static const unsigned MAX_OPEN_NTS = 100;
    if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
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
    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
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
      bufferi[sent.size() - i] = i;
    }

    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    bufferi[0] = -999;
    for (auto& b : buffer)
      buffer_lstm->add_input(b);

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree
    stack.push_back(parameter(*hg, p_stack_guard));
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    if (!NO_STACK) stack_lstm.add_input(stack.back());

    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol

    unsigned nt_count = 0; // number of times an NT has been introduced
    int nopen_parens = 0;
    char prev_a = '0';

    unsigned action_count = 0;
    unsigned action = std::numeric_limits<unsigned>::max();

    return std::static_pointer_cast<AbstractParserState>(
        std::make_shared<ParserState>(
            this,
            action_lstm.state(),
            buffer_lstm->state(),
            stack_lstm.state(),
            Stack<Expression>(buffer),
            Stack<int>(bufferi),
            Stack<Expression>(stack),
            Stack<int>(stacki),
            Stack<int>(is_open_paren),
            nt_count,
            nopen_parens,
            prev_a,
            action_count,
            action
        )
    );
  }

  // *** if correct_actions is empty, this runs greedy decoding ***
  // returns parse actions for input sentence (in training just returns the reference)
  // this lets us use pretrained embeddings, when available, for words that were OOV in the
  // parser training data
  // set sample=true to sample rather than max
  vector<unsigned> log_prob_parser(ComputationGraph* hg,
                       const parser::Sentence& sent,
                       const vector<int>& correct_actions,
                       double *right,
                       bool is_evaluation,
                       bool sample = false) {
      vector<unsigned> results;
      const bool build_training_graph = correct_actions.size() > 0;
      bool apply_dropout = (DROPOUT && !is_evaluation);
      if (!NO_STACK) stack_lstm.new_graph(*hg);
      action_lstm.new_graph(*hg);
      const_lstm_fwd.new_graph(*hg);
      const_lstm_rev.new_graph(*hg);
      if (!NO_STACK) stack_lstm.start_new_sequence();
      buffer_lstm->new_graph(*hg);
      buffer_lstm->start_new_sequence();
      action_lstm.start_new_sequence();
      if (apply_dropout) {
        if (!NO_STACK) stack_lstm.set_dropout(DROPOUT);
        action_lstm.set_dropout(DROPOUT);
        buffer_lstm->set_dropout(DROPOUT);
        const_lstm_fwd.set_dropout(DROPOUT);
        const_lstm_rev.set_dropout(DROPOUT);
      } else {
        if (!NO_STACK) stack_lstm.disable_dropout();
        action_lstm.disable_dropout();
        buffer_lstm->disable_dropout();
        const_lstm_fwd.disable_dropout();
        const_lstm_rev.disable_dropout();
      }
      // variables in the computation graph representing the parameters
      Expression pbias = parameter(*hg, p_pbias);
      Expression S = parameter(*hg, p_S);
      Expression B = parameter(*hg, p_B);
      Expression A = parameter(*hg, p_A);
      Expression ptbias, ptW;
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        ptbias = parameter(*hg, p_ptbias);
        ptW = parameter(*hg, p_ptW);
      }
      Expression p2w;
      if (USE_POS) {
        p2w = parameter(*hg, p_p2w);
      }

      Expression ib = parameter(*hg, p_ib);
      Expression cbias = parameter(*hg, p_cbias);
      Expression w2l = parameter(*hg, p_w2l);
      Expression t2l;
      if (p_t2l)
        t2l = parameter(*hg, p_t2l);
      Expression p2a = parameter(*hg, p_p2a);
      Expression abias = parameter(*hg, p_abias);
      Expression action_start = parameter(*hg, p_action_start);
      Expression cW = parameter(*hg, p_cW);

      action_lstm.add_input(action_start);

      vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings
      vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
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
          bufferi[sent.size() - i] = i;
      }
      // dummy symbol to represent the empty buffer
      buffer[0] = parameter(*hg, p_buffer_guard);
      bufferi[0] = -999;
      for (auto& b : buffer)
        buffer_lstm->add_input(b);

      vector<Expression> stack;  // variables representing subtree embeddings
      vector<int> stacki; // position of words in the sentence of head of subtree
      stack.push_back(parameter(*hg, p_stack_guard));
      stacki.push_back(-999); // not used for anything
      // drive dummy symbol on stack through LSTM
      if (!NO_STACK) stack_lstm.add_input(stack.back());
      vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
      is_open_paren.push_back(-1); // corresponds to dummy symbol

      vector<Expression> log_probs;
      // string rootword; // not used
      unsigned action_count = 0;  // incremented at each prediction
      unsigned nt_count = 0; // number of times an NT has been introduced
      vector<unsigned> current_valid_actions;
      int nopen_parens = 0;
      char prev_a = '0';
      while(stack.size() > 2 || buffer.size() > 1) {
        // get list of possible actions for the current parser state
        current_valid_actions.clear();
        for (auto a: possible_actions) {
          if (IsActionForbidden_Discriminative(adict.Convert(a), prev_a, buffer.size(), stack.size(), nopen_parens))
            continue;
          current_valid_actions.push_back(a);
        }
        //cerr << "valid actions = " << current_valid_actions.size() << endl;

        // p_t = pbias + S * slstm + B * blstm + A * almst
        Expression stack_summary = NO_STACK ? Expression() : stack_lstm.back();
        Expression action_summary = action_lstm.back();
        Expression buffer_summary = buffer_lstm->back();
        if (apply_dropout) {
          if (!NO_STACK) stack_summary = dropout(stack_summary, DROPOUT);
          action_summary = dropout(action_summary, DROPOUT);
          buffer_summary = dropout(buffer_summary, DROPOUT);
        }
        Expression p_t = NO_STACK ?
                         affine_transform({pbias, B, buffer_summary, A, action_summary}) :
                         affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
        Expression nlp_t = rectify(p_t);
        //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
        // r_t = abias + p2a * nlp
        Expression r_t = affine_transform({abias, p2a, nlp_t});
        //if (sample && ALPHA != 1.0f) r_t = r_t * ALPHA;
        // adist = log_softmax(r_t, current_valid_actions)
        Expression adiste = log_softmax(r_t, current_valid_actions);
        vector<float> adist = as_vector(hg->incremental_forward());
        double best_score = adist[current_valid_actions[0]];
        unsigned model_action = current_valid_actions[0];
        if (sample) {
          double p = rand01();
          assert(current_valid_actions.size() > 0);
          vector<float> dist_to_sample;
          if (ALPHA != 1.0f) {
            Expression r_t_smoothed = r_t * ALPHA;
            Expression adiste_smoothed = log_softmax(r_t_smoothed, current_valid_actions);
            dist_to_sample = as_vector(hg->incremental_forward());
          } else {
            dist_to_sample = adist;
          }
          unsigned w = 0;
          for (; w < current_valid_actions.size(); ++w) {
            p -= exp(dist_to_sample[current_valid_actions[w]]);
            if (p < 0.0) { break; }
          }
          if (w == current_valid_actions.size()) w--;
          model_action = current_valid_actions[w];
        } else { // max
          for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
            if (adist[current_valid_actions[i]] > best_score) {
              best_score = adist[current_valid_actions[i]];
              model_action = current_valid_actions[i];
            }
          }
        }
        unsigned action = model_action;
        if (build_training_graph) {  // if we have reference actions (for training) use the reference action
          if (action_count >= correct_actions.size()) {
            cerr << "Correct action list exhausted, but not in final parser state.\n";
            abort();
          }
          action = correct_actions[action_count];
          if (model_action == action) { (*right)++; }
        } else {
          //cerr << "Chosen action: " << adict.Convert(action) << endl;
        }
        //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
        //cerr << endl;
        ++action_count;
        log_probs.push_back(pick(adiste, action));
        results.push_back(action);

        // add current action to action LSTM
        Expression actione = lookup(*hg, p_a, action);
        action_lstm.add_input(actione);

        // do action
        const string& actionString=adict.Convert(action);
        //cerr << "ACT: " << actionString << endl;
        const char ac = actionString[0];
        const char ac2 = actionString[1];
        prev_a = ac;

        if (ac =='S' && ac2=='H') {  // SHIFT
          assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
          if (IMPLICIT_REDUCE_AFTER_SHIFT) {
            --nopen_parens;
            int i = is_open_paren.size() - 1;
            assert(is_open_paren[i] >= 0);
            Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
            Expression terminal = buffer.back();
            Expression c = concatenate({nonterminal, terminal});
            Expression pt = rectify(affine_transform({ptbias, ptW, c}));
            stack.pop_back();
            stacki.pop_back();
            if (!NO_STACK) stack_lstm.rewind_one_step();
            buffer.pop_back();
            bufferi.pop_back();
            buffer_lstm->rewind_one_step();
            is_open_paren.pop_back();
            if (!NO_STACK) stack_lstm.add_input(pt);
            stack.push_back(pt);
            stacki.push_back(999);
            is_open_paren.push_back(-1);
          } else {
            stack.push_back(buffer.back());
            if (!NO_STACK) stack_lstm.add_input(buffer.back());
            stacki.push_back(bufferi.back());
            buffer.pop_back();
            buffer_lstm->rewind_one_step();
            bufferi.pop_back();
            is_open_paren.push_back(-1);
          }
        } else if (ac == 'N') { // NT
          ++nopen_parens;
          assert(buffer.size() > 1);
          auto it = action2NTindex.find(action);
          assert(it != action2NTindex.end());
          int nt_index = it->second;
          nt_count++;
          Expression nt_embedding = lookup(*hg, p_nt, nt_index);
          stack.push_back(nt_embedding);
          if (!NO_STACK) stack_lstm.add_input(nt_embedding);
          stacki.push_back(-1);
          is_open_paren.push_back(nt_index);
        } else { // REDUCE
          --nopen_parens;
          assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
          // find what paren we are closing
          int i = is_open_paren.size() - 1;
          while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
          Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
          int nchildren = is_open_paren.size() - i - 1;
          assert(nchildren > 0);
          //cerr << "  number of children to reduce: " << nchildren << endl;
          vector<Expression> children(nchildren);
          const_lstm_fwd.start_new_sequence();
          const_lstm_rev.start_new_sequence();

          // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
          // TO BE COMPOSED INTO A TREE EMBEDDING
          for (i = 0; i < nchildren; ++i) {
            children[i] = stack.back();
            assert (stacki.back() != -1);
            stacki.pop_back();
            stack.pop_back();
            if (!NO_STACK) stack_lstm.rewind_one_step();
            is_open_paren.pop_back();
          }
          is_open_paren.pop_back(); // nt symbol
          assert (stacki.back() == -1);
          stacki.pop_back(); // nonterminal dummy
          stack.pop_back(); // nonterminal dummy
          if (NO_STACK) {
            stack.push_back(Expression()); // placeholder since we check size
          } else {
            stack_lstm.rewind_one_step(); // nt symbol

            // BUILD TREE EMBEDDING USING BIDIR LSTM
            const_lstm_fwd.add_input(nonterminal);
            const_lstm_rev.add_input(nonterminal);
            for (i = 0; i < nchildren; ++i) {
              const_lstm_fwd.add_input(children[i]);
              const_lstm_rev.add_input(children[nchildren - i - 1]);
            }
            Expression cfwd = const_lstm_fwd.back();
            Expression crev = const_lstm_rev.back();
            if (apply_dropout) {
              cfwd = dropout(cfwd, DROPOUT);
              crev = dropout(crev, DROPOUT);
            }
            Expression c = concatenate({cfwd, crev});
            Expression composed = rectify(affine_transform({cbias, cW, c}));
            stack_lstm.add_input(composed);
            stack.push_back(composed);
          }
          stacki.push_back(999); // who knows, should get rid of this
          is_open_paren.push_back(-1); // we just closed a paren at this position
        }
      }
      if (build_training_graph && action_count != correct_actions.size()) {
        cerr << "Unexecuted actions remain but final state reached!\n";
        abort();
      }
      assert(stack.size() == 2); // guard symbol, root
      assert(stacki.size() == 2);
      assert(buffer.size() == 1); // guard symbol
      assert(bufferi.size() == 1);
      Expression tot_neglogprob = -sum(log_probs);
      assert(tot_neglogprob.pg != nullptr);
      return results;
    }

  struct BeamState {
    RNNPointer stack_position;
    RNNPointer action_position;
    RNNPointer buffer_position;

    vector<Expression> buffer;
    vector<int> bufferi;

    vector<Expression> stack;  // variables representing subtree embeddings
    vector<int> stacki; // position of words in the sentence of head of subtree

    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT

    vector<unsigned> results;  // sequence of predicted actions

    vector<Expression> log_probs;
    double score;
    int action_count;
    int nopen_parens;
    unsigned nt_count;
    char prev_a;

    unsigned action;
  };

  struct BeamStateCompare {
    // sort descending
    bool operator()(const BeamState& a, const BeamState& b) const {
      return a.score > b.score;
    }
  };

  static void prune(vector<BeamState>& pq, unsigned k) {
    if (pq.size() == 1) return;
    if (k > pq.size()) k = pq.size();
    partial_sort(pq.begin(), pq.begin() + k, pq.end(), BeamStateCompare());
    pq.resize(k);
    //reverse(pq.begin(), pq.end()); // shouldn't need to reverse
    //cerr << "PRUNE\n";
    //for (unsigned i = 0; i < pq.size(); ++i) {
    //  cerr << pq[i].score << endl;
    //}
  }

  // log_prob_parser with beam-search
  // *** if correct_actions is empty, this runs greedy decoding ***
  // returns parse actions for input sentence (in training just returns the reference)
  // this lets us use pretrained embeddings, when available, for words that were OOV in the
  // parser training data
  // set sample=true to sample rather than max
  vector<pair<vector<unsigned>, double>> log_prob_parser_beam(ComputationGraph* hg,
                                                              const parser::Sentence& sent,
                                                              unsigned beam_size) {
      // cout << "dropout: " << apply_dropout << endl;
      if (!NO_STACK) stack_lstm.new_graph(*hg);
      action_lstm.new_graph(*hg);
      const_lstm_fwd.new_graph(*hg);
      const_lstm_rev.new_graph(*hg);
      if (!NO_STACK) stack_lstm.start_new_sequence();
      buffer_lstm->new_graph(*hg);
      buffer_lstm->start_new_sequence();
      action_lstm.start_new_sequence();
      if (!NO_STACK) stack_lstm.disable_dropout();
      action_lstm.disable_dropout();
      buffer_lstm->disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
      // variables in the computation graph representing the parameters
      Expression pbias = parameter(*hg, p_pbias);
      Expression S = parameter(*hg, p_S);
      Expression B = parameter(*hg, p_B);
      Expression A = parameter(*hg, p_A);
      Expression ptbias, ptW;
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        ptbias = parameter(*hg, p_ptbias);
        ptW = parameter(*hg, p_ptW);
      }
      Expression p2w;
      if (USE_POS) {
        p2w = parameter(*hg, p_p2w);
      }

      Expression ib = parameter(*hg, p_ib);
      Expression cbias = parameter(*hg, p_cbias);
      Expression w2l = parameter(*hg, p_w2l);
      Expression t2l;
      if (p_t2l)
        t2l = parameter(*hg, p_t2l);
      Expression p2a = parameter(*hg, p_p2a);
      Expression abias = parameter(*hg, p_abias);
      Expression action_start = parameter(*hg, p_action_start);
      Expression cW = parameter(*hg, p_cW);

      BeamState initial_state;

      action_lstm.add_input(action_start);
      initial_state.action_position = action_lstm.state();

      initial_state.buffer = vector<Expression>(sent.size() + 1);  // variables representing word embeddings
      initial_state.bufferi = vector<int>(sent.size() + 1);  // position of the words in the sentence
      // precompute buffer representation from left to right

      // in the discriminative model, here we set up the buffer contents
      for (unsigned i = 0; i < sent.size(); ++i) {
        int wordid = sent.raw[i]; // this will be equal to unk at dev/test
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
        initial_state.buffer[sent.size() - i] = rectify(affine_transform(args));
        initial_state.bufferi[sent.size() - i] = i;
      }
      // dummy symbol to represent the empty buffer
      initial_state.buffer[0] = parameter(*hg, p_buffer_guard);
      initial_state.bufferi[0] = -999;
      for (auto &b : initial_state.buffer)
        buffer_lstm->add_input(b);
      initial_state.buffer_position = buffer_lstm->state();

      initial_state.stack.push_back(parameter(*hg, p_stack_guard));
      initial_state.stacki.push_back(-999); // not used for anything
      // drive dummy symbol on stack through LSTM
      if (!NO_STACK) {
        stack_lstm.add_input(initial_state.stack.back());
        initial_state.stack_position = stack_lstm.state();
      }

      initial_state.is_open_paren.push_back(-1); // corresponds to dummy symbol
      initial_state.action_count = 0;  // incremented at each prediction
      initial_state.nt_count = 0; // number of times an NT has been introduced
      initial_state.nopen_parens = 0;
      initial_state.prev_a = '0';
      initial_state.score = 0.0;
      initial_state.action = 0;

      vector<unsigned> current_valid_actions;

      vector<BeamState> completed;

      vector<BeamState> beam;
      beam.push_back(initial_state);

      while (completed.size() < beam_size && !beam.empty()) {
        vector<BeamState> successors;

        while (!beam.empty()) {

          BeamState current = beam.back();
          beam.pop_back();

          // get list of possible actions for the current parser state
          current_valid_actions.clear();
          for (auto a: possible_actions) {
            if (IsActionForbidden_Discriminative(adict.Convert(a), current.prev_a, current.buffer.size(), current.stack.size(),
                                                 current.nopen_parens))
              continue;
            current_valid_actions.push_back(a);
          }
          //cerr << "valid actions = " << current_valid_actions.size() << endl;

          // p_t = pbias + S * slstm + B * blstm + A * almst
          Expression stack_summary = NO_STACK ? Expression() : stack_lstm.get_h(current.stack_position).back();
          Expression action_summary = action_lstm.get_h(current.action_position).back();
          Expression buffer_summary = buffer_lstm->get_h(current.buffer_position).back();
          Expression p_t = NO_STACK ?
                           affine_transform({pbias, B, buffer_summary, A, action_summary}) :
                           affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
          Expression nlp_t = rectify(p_t);
          //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
          // r_t = abias + p2a * nlp
          Expression r_t = affine_transform({abias, p2a, nlp_t});
          //if (sample && ALPHA != 1.0f) r_t = r_t * ALPHA;
          // adist = log_softmax(r_t, current_valid_actions)
          Expression adiste = log_softmax(r_t, current_valid_actions);
          vector<float> adist = as_vector(hg->incremental_forward());
          /*
          for (auto v : adist) {
            cout << v << "\t";
          }
          cout << endl;
           */

          assert(!current_valid_actions.empty());
          for (unsigned i = 0; i < current_valid_actions.size(); i++) {
            unsigned possible_action = current_valid_actions[i];
            double score = adist[possible_action];

            // update only score, log_prob, and action for now
            // other state vars will be updated after pruning beam for efficiency
            BeamState successor = current;
            successor.log_probs.push_back(pick(adiste, possible_action));

            successor.results.push_back(possible_action);
            successor.action = possible_action;


            successor.score += score;
            ++successor.action_count;

            successors.push_back(successor);
          }
        }

        // cut down to beam size
        prune(successors, beam_size);

        // update state variables for top K
        // check if any of the successors are complete; add others back to the beam
        for (unsigned i = 0; i < successors.size(); i++) {

          BeamState successor = successors[i];
          unsigned action = successor.action;

          // add current action to action LSTM
          Expression actione = lookup(*hg, p_a, action);
          action_lstm.add_input(successor.action_position, actione);
          successor.action_position = action_lstm.state();

          // do action
          const string &actionString = adict.Convert(action);
          //cerr << "ACT: " << actionString << endl;
          const char ac = actionString[0];
          const char ac2 = actionString[1];
          successor.prev_a = ac;

          if (ac == 'S' && ac2 == 'H') {  // SHIFT
            assert(successor.buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
            if (IMPLICIT_REDUCE_AFTER_SHIFT) {
              --successor.nopen_parens;
              int i = successor.is_open_paren.size() - 1;
              assert(successor.is_open_paren[i] >= 0);
              Expression nonterminal = lookup(*hg, p_ntup, successor.is_open_paren[i]);
              Expression terminal = successor.buffer.back();
              Expression c = concatenate({nonterminal, terminal});
              Expression pt = rectify(affine_transform({ptbias, ptW, c}));
              successor.stack.pop_back();
              successor.stacki.pop_back();
              if (!NO_STACK) {
                successor.stack_position = stack_lstm.head_of(successor.stack_position);
              }
              successor.buffer.pop_back();
              successor.bufferi.pop_back();
              successor.buffer_position = buffer_lstm->head_of(successor.buffer_position);
              successor.is_open_paren.pop_back();
              if (!NO_STACK) {
                stack_lstm.add_input(successor.stack_position, pt);
                successor.stack_position = stack_lstm.state();
              }
              successor.stack.push_back(pt);
              successor.stacki.push_back(999);
              successor.is_open_paren.push_back(-1);
            } else {
              successor.stack.push_back(successor.buffer.back());
              if (!NO_STACK) {
                stack_lstm.add_input(successor.stack_position, successor.buffer.back());
                successor.stack_position = stack_lstm.state();
              }
              successor.stacki.push_back(successor.bufferi.back());
              successor.buffer.pop_back();
              successor.buffer_position = buffer_lstm->head_of(successor.buffer_position);
              successor.bufferi.pop_back();
              successor.is_open_paren.push_back(-1);
            }
          } else if (ac == 'N') { // NT
            ++successor.nopen_parens;
            assert(successor.buffer.size() > 1);
            auto it = action2NTindex.find(action);
            assert(it != action2NTindex.end());
            int nt_index = it->second;
            successor.nt_count++;
            Expression nt_embedding = lookup(*hg, p_nt, nt_index);
            successor.stack.push_back(nt_embedding);
            if (!NO_STACK) {
              stack_lstm.add_input(successor.stack_position, nt_embedding);
              successor.stack_position = stack_lstm.state();
            }
            successor.stacki.push_back(-1);
            successor.is_open_paren.push_back(nt_index);
          } else { // REDUCE
            --successor.nopen_parens;
            assert(successor.stack.size() > 2); // dummy symbol means > 2 (not >= 2)
            // find what paren we are closing
            int i = successor.is_open_paren.size() - 1;
            while (successor.is_open_paren[i] < 0) {
              --i;
              assert(i >= 0);
            }
            Expression nonterminal = lookup(*hg, p_ntup, successor.is_open_paren[i]);
            int nchildren = successor.is_open_paren.size() - i - 1;
            assert(nchildren > 0);
            //cerr << "  number of children to reduce: " << nchildren << endl;
            vector<Expression> children(nchildren);
            const_lstm_fwd.start_new_sequence();
            const_lstm_rev.start_new_sequence();

            // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
            // TO BE COMPOSED INTO A TREE EMBEDDING
            for (i = 0; i < nchildren; ++i) {
              children[i] = successor.stack.back();
              assert (successor.stacki.back() != -1);
              successor.stacki.pop_back();
              successor.stack.pop_back();
              if (!NO_STACK) {
                successor.stack_position = stack_lstm.head_of(successor.stack_position);
              }
              successor.is_open_paren.pop_back();
            }
            successor.is_open_paren.pop_back(); // nt symbol
            assert (successor.stacki.back() == -1);
            successor.stacki.pop_back(); // nonterminal dummy
            successor.stack.pop_back(); // nonterminal dummy
            if (NO_STACK) {
              successor.stack.push_back(Expression()); // placeholder since we check size
            } else {
              successor.stack_position = stack_lstm.head_of(successor.stack_position); // nt symbol

              // BUILD TREE EMBEDDING USING BIDIR LSTM
              const_lstm_fwd.add_input(nonterminal);
              const_lstm_rev.add_input(nonterminal);
              for (i = 0; i < nchildren; ++i) {
                const_lstm_fwd.add_input(children[i]);
                const_lstm_rev.add_input(children[nchildren - i - 1]);
              }
              Expression cfwd = const_lstm_fwd.back();
              Expression crev = const_lstm_rev.back();
              Expression c = concatenate({cfwd, crev});
              Expression composed = rectify(affine_transform({cbias, cW, c}));
              stack_lstm.add_input(successor.stack_position, composed);
              successor.stack_position = stack_lstm.state();
              successor.stack.push_back(composed);
            }
            successor.stacki.push_back(999); // who knows, should get rid of this
            successor.is_open_paren.push_back(-1); // we just closed a paren at this position
          } // end REDUCE

          if (successor.stack.size() <= 2 && successor.buffer.size() <= 1) {
            completed.push_back(successor);
          } else {
            beam.push_back(successor);
          }
        } // end successor iteration
      } // end build completed

      sort(completed.begin(), completed.end(), BeamStateCompare());

      vector<pair<vector<unsigned>, double>> completed_actions_and_scores;
      for (const auto & beam_state : completed) {
        assert(beam_state.stack.size() == 2); // guard symbol, root
        assert(beam_state.stacki.size() == 2);
        assert(beam_state.buffer.size() == 1); // guard symbol
        assert(beam_state.bufferi.size() == 1);
        Expression tot_neglogprob = -sum(beam_state.log_probs);
        assert(tot_neglogprob.pg != nullptr);
        double tnlp = as_scalar(hg->incremental_forward());
        completed_actions_and_scores.push_back(pair<vector<unsigned>, double>(beam_state.results, tnlp));
      }
      return completed_actions_and_scores;
    }
};

struct EnsembledParserState;

struct EnsembledParser : public AbstractParser {
  enum class CombineType { sum, product };

  vector<std::shared_ptr<ParserBuilder>> parsers;
  CombineType combine_type;

  explicit EnsembledParser(vector<std::shared_ptr<ParserBuilder>> parsers, CombineType combine_type) :
      parsers(parsers), combine_type(combine_type) {
    assert(!parsers.empty());
  }

  std::shared_ptr<AbstractParserState> new_sentence(ComputationGraph* hg, const parser::Sentence& sent, bool is_evaluation, bool build_training_graph, bool apply_dropout) override {
    vector<std::shared_ptr<AbstractParserState>> states;
    for (const std::shared_ptr<ParserBuilder>& parser : parsers)
      states.push_back(parser->new_sentence(hg, sent, is_evaluation, build_training_graph, apply_dropout));
    return std::static_pointer_cast<AbstractParserState>(std::make_shared<EnsembledParserState>(this, states));
  }
};

struct ParserState : public AbstractParserState {
  ParserState(
      ParserBuilder* parser,
      RNNPointer action_state,
      RNNPointer buffer_state,
      RNNPointer stack_state,
      Stack<Expression> buffer,
      Stack<int> bufferi,
      Stack<Expression> stack,
      Stack<int> stacki,
      Stack<int> is_open_paren,
      unsigned nt_count,
      int nopen_parens,
      char prev_a,
      unsigned action_count,
      unsigned action
  ) :
      parser(parser),
      action_state(action_state),
      buffer_state(buffer_state),
      stack_state(stack_state),
      buffer(buffer),
      bufferi(bufferi),
      stack(stack),
      stacki(stacki),
      is_open_paren(is_open_paren),
      nt_count(nt_count),
      nopen_parens(nopen_parens),
      prev_a(prev_a),
      action_count(action_count),
      action(action)
  {}

  ParserBuilder* parser;

  const RNNPointer action_state;
  const RNNPointer buffer_state;
  const RNNPointer stack_state;

  const Stack<Expression> buffer;
  const Stack<int> bufferi;

  const Stack<Expression> stack;
  const Stack<int> stacki;

  const Stack<int> is_open_paren;

  const unsigned nt_count;
  const int nopen_parens;
  const char prev_a;

  const unsigned action_count;
  const unsigned action;

  bool is_finished() override {
    return stack.size() == 2 && buffer.size() == 1;
  }

  vector<unsigned> get_valid_actions() override {
    vector<unsigned> valid_actions;
    for (auto a: possible_actions) {
      if (ParserBuilder::IsActionForbidden_Discriminative(adict.Convert(a), prev_a, buffer.size(), stack.size(), nopen_parens))
        continue;
      valid_actions.push_back(a);
    }
    return valid_actions;
  }

  Expression get_action_log_probs(const vector<unsigned>& valid_actions) override {
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

  std::shared_ptr<AbstractParserState> perform_action(const unsigned action) override {
    const string& actionString = adict.Convert(action);
    const char ac = actionString[0];
    const char ac2 = actionString[1];

    RNNPointer new_action_state = action_state;
    RNNPointer new_buffer_state = buffer_state;
    RNNPointer new_stack_state = stack_state;
    Stack<Expression> new_buffer(buffer);
    Stack<int> new_bufferi(bufferi);
    Stack<Expression> new_stack(stack);
    Stack<int> new_stacki(stacki);
    Stack<int> new_is_open_paren(is_open_paren);
    unsigned new_nt_count = nt_count;
    int new_nopen_parens = nopen_parens;

    // add current action to action LSTM
    Expression actione = lookup(*parser->hg, parser->p_a, action);
    parser->action_lstm.add_input(action_state, actione);
    new_action_state = parser->action_lstm.state();

    if (ac =='S' && ac2=='H') {  // SHIFT
      assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        --new_nopen_parens;
        assert(is_open_paren.back() >= 0);
        Expression nonterminal = lookup(*parser->hg, parser->p_ntup, is_open_paren.back());
        Expression terminal = buffer.back();
        Expression c = concatenate({nonterminal, terminal});
        Expression pt = rectify(affine_transform({parser->ptbias, parser->ptW, c}));
        new_stack = new_stack.pop_back();
        new_stacki = new_stacki.pop_back();
        if (!NO_STACK) new_stack_state = parser->stack_lstm.head_of(stack_state);
        new_buffer = new_buffer.pop_back();
        new_bufferi = new_bufferi.pop_back();
        new_buffer_state = parser->buffer_lstm->head_of(buffer_state);
        new_is_open_paren = new_is_open_paren.pop_back();
        if (!NO_STACK) {
          parser->stack_lstm.add_input(new_stack_state, pt);
          new_stack_state = parser->stack_lstm.state();
        }
        new_stack = new_stack.push_back(pt);
        new_stacki = new_stacki.push_back(999);
        new_is_open_paren = new_is_open_paren.push_back(-1);
      } else {
        new_stack = new_stack.push_back(buffer.back());
        if (!NO_STACK) {
          parser->stack_lstm.add_input(stack_state, buffer.back());
          new_stack_state = parser->stack_lstm.state();
        }
        new_stacki = new_stacki.push_back(bufferi.back());
        new_buffer = new_buffer.pop_back();
        new_buffer_state = parser->buffer_lstm->head_of(buffer_state);
        new_bufferi = new_bufferi.pop_back();
        new_is_open_paren = new_is_open_paren.push_back(-1);
      }
    }

    else if (ac == 'N') { // NT
      ++new_nopen_parens;
      assert(buffer.size() > 1);
      auto it = action2NTindex.find(action);
      assert(it != action2NTindex.end());
      int nt_index = it->second;
      new_nt_count++;
      Expression nt_embedding = lookup(*parser->hg, parser->p_nt, nt_index);
      new_stack = new_stack.push_back(nt_embedding);
      if (!NO_STACK) {
        parser->stack_lstm.add_input(stack_state, nt_embedding);
        new_stack_state = parser->stack_lstm.state();
      }
      new_stacki = new_stacki.push_back(-1);
      new_is_open_paren = new_is_open_paren.push_back(nt_index);
    }

    else { // REDUCE
      --new_nopen_parens;
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
        assert(new_stacki.back() != -1);
        new_stacki = new_stacki.pop_back();
        new_stack = new_stack.pop_back();
        if (!NO_STACK) new_stack_state = parser->stack_lstm.head_of(new_stack_state);
        new_is_open_paren = new_is_open_paren.pop_back();
      }
      new_is_open_paren = new_is_open_paren.pop_back(); // nt symbol
      assert(new_stacki.back() == -1);
      new_stacki = new_stacki.pop_back(); // nonterminal dummy
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
      new_stacki = new_stacki.push_back(999); // who knows, should get rid of this
      new_is_open_paren = new_is_open_paren.push_back(-1); // we just closed a paren at this position
    }

    return std::make_shared<ParserState>(
        parser,
        new_action_state,
        new_buffer_state,
        new_stack_state,
        new_buffer,
        new_bufferi,
        new_stack,
        new_stacki,
        new_is_open_paren,
        new_nt_count,
        new_nopen_parens,
        ac,
        action_count + 1,
        action
    );
  }

  void finish_sentence() override {
    assert(stack.size() == 2); // guard symbol, root
    assert(stacki.size() == 2);
    assert(buffer.size() == 1); // guard symbol
    assert(bufferi.size() == 1);
  }
};

struct EnsembledParserState : public AbstractParserState {
  const EnsembledParser* parser;
  const vector<std::shared_ptr<AbstractParserState>> states;

  explicit EnsembledParserState(const EnsembledParser* parser, vector<std::shared_ptr<AbstractParserState>> states) :
      parser(parser), states(states) {
    assert(!states.empty());
  }

  bool is_finished() override {
    return states.front()->is_finished();
  }

  vector<unsigned> get_valid_actions() override {
    return states.front()->get_valid_actions();
  }

  Expression get_action_log_probs(const vector<unsigned>& valid_actions) override {
    vector<Expression> all_log_probs;
    for (const std::shared_ptr<AbstractParserState>& state : states)
      all_log_probs.push_back(state->get_action_log_probs(valid_actions));
    Expression combined_log_probs;
    switch (parser->combine_type) {
      case EnsembledParser::CombineType::sum: {
        // combined_log_probs = logsumexp(all_log_probs); // numerically unstable
        // Expression cwise_max = max(all_log_probs); // gives an error despite being correct
        Expression cwise_max = all_log_probs.front();
        for (auto it = all_log_probs.begin() + 1; it != all_log_probs.end(); ++it)
          cwise_max = max(cwise_max, *it);
        vector <Expression> exp_log_probs;
        for (const Expression &log_probs : all_log_probs)
          exp_log_probs.push_back(exp(log_probs) - cwise_max);
        combined_log_probs = log(sum(exp_log_probs)) + cwise_max;
        break;
      }
      case EnsembledParser::CombineType::product:
        combined_log_probs = sum(all_log_probs);
        break;
    }
    return log_softmax(combined_log_probs, valid_actions);
  }

  std::shared_ptr<AbstractParserState> perform_action(const unsigned action) override {
    vector<std::shared_ptr<AbstractParserState>> new_states;
    for (const std::shared_ptr<AbstractParserState>& state : states)
      new_states.push_back(state->perform_action(action));
    return std::make_shared<EnsembledParserState>(parser, new_states);
  }

  void finish_sentence() override {
    for (const std::shared_ptr<AbstractParserState>& state : states)
      state->finish_sentence();
  }
};

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
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  //bool softlinkCreated = false;

  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracle gold_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>(), true);	
  corpus.load_bdata(conf["bracketing_dev_data"].as<string>());

  bool has_gold_training_data = false;

  if (conf.count("gold_training_data")) {
    gold_corpus.load_oracle(conf["gold_training_data"].as<string>(), true);
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

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), false);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>(), false);
  }

  non_unked_termdict.Freeze();

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);

    Model model;
    ParserBuilder parser(&model, pretrained);
    SimpleSGDTrainer sgd(&model);
    cerr << "using sgd for training" << endl;

    if (conf.count("model")) {
      ifstream in(conf["model"].as<string>().c_str());
      boost::archive::binary_iarchive ia(in);
      ia >> model >> sgd;
    }

    //AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;
    sgd.eta_decay = 0.05;

    unsigned tot_seen = 0;
    int iter = -1;

    double best_dev_err = 9e99;
    double bestf1=0.0;

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

      for (vector<unsigned>::iterator index_iter = indices_begin; index_iter != indices_end; ++index_iter) {
        tot_seen += 1;
        auto& sentence = corpus.sents[*index_iter];
        const vector<int>& actions=corpus.actions[*index_iter];
        {
          ComputationGraph hg;
          parser.log_prob_parser(&hg, sentence, actions, &right, false);
          double lp = as_scalar(hg.incremental_forward());
          if (lp < 0) {
            cerr << "Log prob < 0 on sentence " << *index_iter << ": lp=" << lp << endl;
            assert(lp >= 0.0);
          }
          hg.backward();
          sgd.update(1.0);
          llh += lp;
        }
        trs += actions.size();
        words += sentence.size();

        if (tot_seen % status_every_i_iterations == 0) {
          ++iter;
          sgd.status();
          auto time_now = chrono::system_clock::now();
          auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
          cerr << "update #" << iter << " (epoch " << (static_cast<double>(tot_seen) / epoch_size) <<
               /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
               ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
          llh = trs = right = words = 0;

          if (iter % 25 == 0) { // report on dev set
            unsigned dev_size = dev_corpus.size();
            double llh = 0;
            double trs = 0;
            double right = 0;
            double dwords = 0;
            ostringstream os;
            os << "/tmp/parser_dev_eval." << getpid() << ".txt";
            const string pfx = os.str();
            ofstream out(pfx.c_str());
            auto t_start = chrono::high_resolution_clock::now();
            for (unsigned sii = 0; sii < dev_size; ++sii) {
              const auto& sentence=dev_corpus.sents[sii];
              const vector<int>& actions=dev_corpus.actions[sii];
              dwords += sentence.size();
              {  ComputationGraph hg;
                parser.log_prob_parser(&hg,sentence,actions,&right,true);
                double lp = as_scalar(hg.incremental_forward());
                llh += lp;
              }
              ComputationGraph hg;
              vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
              int ti = 0;
              for (auto a : pred) {
                if (adict.Convert(a)[0] == 'N') {
                  out << '(' << ntermdict.Convert(action2NTindex.find(a)->second) << ' ';
                } else if (adict.Convert(a)[0] == 'S') {
                  if (IMPLICIT_REDUCE_AFTER_SHIFT) {
                    out << termdict.Convert(sentence.raw[ti++]) << ") ";
                  } else {
                    if (true) {
                      string preterminal = "XX";
                      out << '(' << preterminal << ' ' << termdict.Convert(sentence.raw[ti++]) << ") ";
                    } else { // use this branch to surpress preterminals
                      out << termdict.Convert(sentence.raw[ti++]) << ' ';
                    }
                  }
                } else out << ") ";
              }
              out << endl;
              double lp = 0;
              trs += actions.size();
            }
            auto t_end = chrono::high_resolution_clock::now();
            out.close();
            double err = (trs - right) / trs;
            cerr << "Dev output in " << pfx << endl;
            //parser::EvalBResults res = parser::Evaluate("foo", pfx);
            std::string evaluable_fname = pfx + "_evaluable.txt";
            std::string evalbout_fname = pfx + "_evalbout.txt";
            std::string command="python remove_dev_unk.py "+ corpus.devdata +" "+pfx+" > " + evaluable_fname;
            const char* cmd=command.c_str();
            system(cmd);

            std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" " + evaluable_fname + " > " + evalbout_fname;
            const char* cmd2=command2.c_str();

            system(cmd2);

            std::ifstream evalfile(evalbout_fname);
            std::string lineS;
            std::string brackstr="Bracketing FMeasure";
            double newfmeasure=0.0;
            std::string strfmeasure="";
            bool found=0;
            while (getline(evalfile, lineS) && !newfmeasure){
              if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                //std::cout<<lineS<<"\n";
                strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                std::string::size_type sz;     // alias of size_t

                newfmeasure = std::stod (strfmeasure,&sz);
                //std::cout<<strfmeasure<<"\n";
              }
            }



            cerr << "  **dev (iter=" << iter << " epoch=" << (static_cast<double>(tot_seen) / epoch_size) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << newfmeasure << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
            if (newfmeasure>bestf1) {
              cerr << "  new best...writing model to " << fname << ".bin ...\n";
              best_dev_err = err;
              bestf1=newfmeasure;
              ofstream out(fname + ".bin");
              boost::archive::binary_oarchive oa(out);
              // oa << model;
              oa << model << sgd;
              oa << termdict << adict << ntermdict << posdict;
              system((string("cp ") + pfx + string(" ") + pfx + string(".best")).c_str());
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

      sgd.update_epoch();

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
      boost::archive::binary_iarchive ia(in);
      ia >> *models.back();
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
        boost::archive::binary_iarchive ia(in);
        ia >> *models.back();
      }
      ensembled_parser = std::make_shared<EnsembledParser>(parsers, combine_types.at(combine_type));
      cerr << "Loaded ensembled parser with combine_type of " << combine_type << "." << endl;
      abstract_parser = ensembled_parser.get();
    }

    bool sample = conf.count("samples") > 0;
    ostringstream ptb_os;
    if (conf.count("ptb_output_file")) {
      ptb_os << conf["ptb_output_file"].as<string>();
    } else {
      ptb_os << "/tmp/parser_ptb_out." << getpid() << ".txt";
    }
    ofstream ptb_out(ptb_os.str().c_str());

    bool output_beam_as_samples = conf.count("output_beam_as_samples");
    if (sample && output_beam_as_samples) {
      cerr << "warning: outputting samples and the contents of the beam\n";
    }
    unsigned beam_size = conf["beam_size"].as<unsigned>();
    unsigned test_size = test_corpus.size();
    double right = 0;
    auto t_start = chrono::high_resolution_clock::now();
    const vector<int> actions;
    vector<unsigned> n_distinct_samples;
    for (unsigned sii = 0; sii < test_size; ++sii) {
      const auto& sentence=test_corpus.sents[sii];
      // TODO: this overrides dynet random seed, but should be ok if we're only sampling
      cnn::rndeng->seed(sii);
      set<vector<unsigned>> samples;
      if (conf.count("samples_include_gold")) {
        ComputationGraph hg;
        vector<unsigned> result = abstract_parser->abstract_log_prob_parser(&hg,sentence, test_corpus.actions[sii],&right,true);
        double lp = as_scalar(hg.incremental_forward());
        cout << sii << " ||| " << -lp << " |||";
        vector<unsigned> converted_actions(test_corpus.actions[sii].begin(), test_corpus.actions[sii].end());
        print_parse(converted_actions, sentence, false, cout);
        ptb_out << sii << " ||| " << -lp << " |||";
        print_parse(converted_actions, sentence, true, ptb_out);
        samples.insert(converted_actions);
      }

      for (unsigned z = 0; z < N_SAMPLES; ++z) {
        ComputationGraph hg;
        vector<unsigned> result = abstract_parser->abstract_log_prob_parser(&hg,sentence,actions,&right,sample,true); // TODO: fix ordering of sample and eval here
        double lp = as_scalar(hg.incremental_forward());
        cout << sii << " ||| " << -lp << " |||";
        print_parse(result, sentence, false, cout);
        ptb_out << sii << " ||| " << -lp << " |||";
        print_parse(result, sentence, true, ptb_out);
        samples.insert(result);
      }

      /*
      if (output_beam_as_samples) {
        ComputationGraph hg;
        auto beam_results = parser.log_prob_parser_beam(&hg, sentence, beam_size);
        if (beam_results.size() < beam_size) {
          cerr << "warning: only " << beam_results.size() << " parses found by beam search for sent " << sii << endl;
        }
        unsigned long num_results = beam_results.size();
        for (unsigned long i = 0; i < beam_size; i++) {
          unsigned long ix = std::min(i, num_results - 1);
          pair<vector<unsigned>, double> result_and_nlp = beam_results[ix];
          double lp = result_and_nlp.second;
          cout << sii << " ||| " << -lp << " |||";
          print_parse(result_and_nlp.first, sentence, false, cout);
          ptb_out << sii << " ||| " << -lp << " |||";
          print_parse(result_and_nlp.first, sentence, true, ptb_out);
          samples.insert(result_and_nlp.first);
        }
      }
      */

      n_distinct_samples.push_back(samples.size());
    }
    double avg_distinct_samples = accumulate(n_distinct_samples.begin(), n_distinct_samples.end(), 0.0) / (double)  n_distinct_samples.size();
    cerr << "avg distinct samples: " << avg_distinct_samples << endl;
    ostringstream os;
    os << "/tmp/parser_test_eval." << getpid() << ".txt";
    const string pfx = os.str();
    ofstream out(pfx.c_str());
    t_start = chrono::high_resolution_clock::now();
    for (unsigned sii = 0; sii < test_size; ++sii) {
      const auto& sentence=test_corpus.sents[sii];
      const vector<int>& actions=test_corpus.actions[sii];
      ComputationGraph hg;
      // greedy predict
      pair<vector<unsigned>, double> result_and_nlp;
      /* if (beam_size > 1) {
        auto beam_results = parser.log_prob_parser_beam(&hg, sentence, beam_size);
        result_and_nlp = beam_results[0];
      } else */ {
        vector<unsigned> result = abstract_parser->abstract_log_prob_parser(&hg, sentence, vector<int>(), &right, true);
        double nlp = as_scalar(hg.incremental_forward());
        result_and_nlp = pair<vector<unsigned>, double>(result, nlp);
      }
      int ti = 0;
      // TODO: convert to use print_parse
      for (auto a : result_and_nlp.first) {
        if (adict.Convert(a)[0] == 'N') {
          out << '(' << ntermdict.Convert(action2NTindex.find(a)->second) << ' ';
        } else if (adict.Convert(a)[0] == 'S') {
          if (IMPLICIT_REDUCE_AFTER_SHIFT) {
            out << termdict.Convert(sentence.raw[ti++]) << ") ";
          } else {
            if (true) {
              string preterminal = "XX";
              out << '(' << preterminal << ' ' << termdict.Convert(sentence.raw[ti++]) << ") ";
            } else { // use this branch to surpress preterminals
              out << termdict.Convert(sentence.raw[ti++]) << ' ';
            }
          }
        } else out << ") ";
      }
      out << endl;
    }
    auto t_end = chrono::high_resolution_clock::now();
    out.close();
    cerr << "Test output in " << pfx << endl;
    //parser::EvalBResults res = parser::Evaluate("foo", pfx);
    std::string evaluable_fname = pfx + "_evaluable.txt";
    std::string evalbout_fname = pfx + "_evalbout.txt";
	  std::string command="python remove_dev_unk.py "+ corpus.devdata +" "+pfx+" > " + evaluable_fname;
    const char* cmd=command.c_str();
    system(cmd);

    std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" " + evaluable_fname + " > " + evalbout_fname; 
    const char* cmd2=command2.c_str();

    system(cmd2);

    std::ifstream evalfile(evalbout_fname);
    std::string lineS;
    std::string brackstr="Bracketing FMeasure";
    double newfmeasure=0.0;
    std::string strfmeasure="";
    bool found=0;
    while (getline(evalfile, lineS) && !newfmeasure){
      if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
        //std::cout<<lineS<<"\n";
        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
        std::string::size_type sz;
        newfmeasure = std::stod (strfmeasure,&sz);
        //std::cout<<strfmeasure<<"\n";
      }
    }

    cerr<<"F1score: "<<newfmeasure<<"\n";

  }
}
