#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/functional/hash.hpp>
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

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict, non_unked_termdict;

volatile bool requested_stop = false;
unsigned kSOS = 0;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;

unsigned MAX_CONS_NT = 8;

unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X

bool IGNORE_WORD_IN_GREEDY = false;

bool WORD_COMPLETION_IS_SHIFT = false;

bool NO_BUFFER = false;
bool NO_HISTORY = false;

unsigned SILVER_BLOCKS_PER_GOLD = 10;

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;

ClassFactoredSoftmaxBuilder *cfsm = nullptr;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("explicit_terminal_reduce,x", "[not recommended] If set, the parser must explicitly process a REDUCE operation to complete a preterminal constituent")
        ("dropout,D", po::value<float>(), "Use dropout")
        ("clusters,c", po::value<string>(), "Clusters word clusters file")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")
        ("gold_training_data", po::value<string>(), "List of Transitions - smaller corpus (e.g. wsj in a wsj+silver experiment)")
        ("silver_blocks_per_gold", po::value<unsigned>()->default_value(10), "How many same-sized blocks of the silver data should be sampled and trained, between every train on the entire gold set?")
        ("test_data,p", po::value<string>(), "Test corpus")
        ("eta_decay,e", po::value<float>(), "Start decaying eta after this many epochs")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("greedy_decode_dev,g", "greedy decode")
        ("dev_output_file,O", po::value<string>(), "write decoded parse trees to this file")
        ("beam_within_word", "greedy decode within word")
        ("ignore_word_in_greedy,i", "greedy decode")
        ("word_completion_is_shift,s", "consider a word completed when it's shifted for beaming and print purposes")
        ("decode_beam_size,b", po::value<unsigned>()->default_value(1), "size of beam to use in decode")
        ("decode_beam_filter_at_word_size", po::value<int>()->default_value(-1), "when using beam_within_word, filter word completions to this size (defaults to decode_beam_size if < 0)")
        ("max_cons_nt", po::value<unsigned>()->default_value(8), "maximum number of non-terminals that can be opened consecutively")
        ("no_history", "Don't encode the history")
        ("no_buffer", "Don't encode the buffer")
        ("block_count", po::value<unsigned>()->default_value(0), "divide the dev set up into this many blocks and only decode one of them (indexed by block_num)")
        ("block_num", po::value<unsigned>()->default_value(0), "decode only this block (0-indexed), must be used with block_count")
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

struct BeamState {
  RNNPointer stack_position;
  RNNPointer action_position;
  RNNPointer term_position;

  vector<Expression> terms;

  vector<Expression> stack;  // variables representing subtree embeddings
  vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT

  vector<unsigned> results;

  vector<Expression> log_probs;

  unsigned action_count;
  unsigned nt_count;

  unsigned cons_nt;

  int nopen_parens;

  double score;

  char prev_a;

  unsigned termc;

  unsigned action;

};

struct BeamStateCompare {
  // sort descending
  bool operator()(const BeamState& a, const BeamState& b) const {
    return a.score > b.score;
  }
};

static void prune(vector<BeamState>& pq, unsigned k) {
  if (pq.size() <= 1) return;
  if (k > pq.size()) k = pq.size();
  // sort descending
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), BeamStateCompare());
  // keep the top k
  pq.resize(k);
  // reverse(pq.begin(), pq.end()); // shouldn't need to reverse
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

void print_parse(const vector<unsigned>& actions, const parser::Sentence& sentence, bool include_preterms, ostream& out_stream) {
  int ti = 0;
  for (auto a : actions) {
    if (adict.Convert(a)[0] == 'N') {
      out_stream << " (" << ntermdict.Convert(action2NTindex.find(a)->second);
    } else if (adict.Convert(a)[0] == 'S') {
      if (IMPLICIT_REDUCE_AFTER_SHIFT) {
        out_stream << termdict.Convert(sentence.raw[ti++]) << ")";
      } else {
        if (include_preterms) {
          string preterminal = "XX";
          out_stream << " (" << preterminal << ' ' << termdict.Convert(sentence.raw[ti]) << ")";
          ti++;
        } else { // use this branch to surpress preterminals
          out_stream << ' ' << termdict.Convert(sentence.raw[ti++]);
        }
      }
    } else out_stream << ')';
  }
  out_stream << endl;
}

struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder term_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_nt; // nonterminal embeddings
  LookupParameters* p_ntup; // nonterminal embeddings when used in a composed representation
  LookupParameters* p_a; // input action embeddings
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_T; // term lstm to parser state
  //Parameters* p_pbias2; // parser state bias
  //Parameters* p_A2; // action lstm to parser state
  //Parameters* p_S2; // stack lstm to parser state
  //Parameters* p_w2l; // word to LSTM input
  //Parameters* p_t2l; // pretrained word embeddings to LSTM input
  //Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_stack_guard;  // end of stack

  Parameters* p_cW;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      term_lstm(LAYERS, INPUT_DIM, HIDDEN_DIM, model),  // sequence of generated terminals
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(1, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_ntup(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_T(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      //p_pbias2(model->add_parameters({HIDDEN_DIM})),
      //p_A2(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      //p_S2(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      //p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      //p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),

      p_cW(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM * 2})) {
    if (pretrained.size() > 0) {
      cerr << "Pretrained embeddings not implemented\n";
      abort();
    }
  }

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Generative(const string& a, char prev_a, unsigned tsize, unsigned ssize, unsigned nopen_parens) {
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
  // you can't reduce after an NT action
  if (is_reduce && prev_a == 'N') return true;
  return false;
}

  /*
void print_parse(const vector<unsigned>& actions, const parser::Sentence& sent) {
  unsigned termc = 0;
  for (unsigned action : actions) {
    const string& a = adict.Convert(action);
    if (a[0] == 'R') {
      cerr << ") ";
    } else if (a[0] == 'N') {
      int nt = action2NTindex[action];
      cerr << " (" << ntermdict.Convert(nt) << " ";
    } else if (a[0] =='S') {
      string word = termdict.Convert(sent.raw[termc]);
      cerr << word << " ";
      termc++;
    }
  }
  cerr << endl;
}
   */

// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// if sent is empty, generate a sentence
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
                     bool is_evaluation) {
    vector<unsigned> results;
    vector<string> stack_content;
    stack_content.push_back("ROOT_GUARD");
    const bool sample = sent.size() == 0;
    const bool build_training_graph = correct_actions.size() > 0;
    assert(sample || build_training_graph);
    bool apply_dropout = (DROPOUT && !is_evaluation);
    if (sample) apply_dropout = false;

    if (apply_dropout) {
      stack_lstm.set_dropout(DROPOUT);
      if (!NO_BUFFER) term_lstm.set_dropout(DROPOUT);
      if (!NO_HISTORY) action_lstm.set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      stack_lstm.disable_dropout();
      if (!NO_BUFFER) term_lstm.disable_dropout();
      if (!NO_HISTORY) action_lstm.disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }
    if (!NO_BUFFER) term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    if (!NO_HISTORY) action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    if (!NO_BUFFER) term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    if (!NO_HISTORY) action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    //Expression pbias2 = parameter(*hg, p_pbias2);
    //Expression S2 = parameter(*hg, p_S2);
    //Expression A2 = parameter(*hg, p_A2);

    //Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    //Expression w2l = parameter(*hg, p_w2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    if (!NO_HISTORY) action_lstm.add_input(action_start);

    vector<Expression> terms(1, lookup(*hg, p_w, kSOS));
    if (!NO_BUFFER) term_lstm.add_input(terms.back());

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';
    unsigned termc = 0;
    while(stack.size() > 2 || termc == 0) {
      assert (stack.size() == stack_content.size());
      // get list of possible actions for the current parser state
      current_valid_actions.clear();
      for (auto a: possible_actions) {
        if (IsActionForbidden_Generative(adict.Convert(a), prev_a, terms.size(), stack.size(), nopen_parens))
          continue;
        current_valid_actions.push_back(a);
      }
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      //onerep
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = (NO_HISTORY) ? Expression() : action_lstm.back();
      Expression term_summary = (NO_BUFFER) ? Expression() : term_lstm.back();
      if (apply_dropout) {
        stack_summary = dropout(stack_summary, DROPOUT);
        if (!NO_HISTORY) action_summary = dropout(action_summary, DROPOUT);
        if (!NO_BUFFER) term_summary = dropout(term_summary, DROPOUT);
      }
      Expression p_t;
      if (NO_BUFFER && NO_HISTORY) {
        p_t = affine_transform({pbias, S, stack_summary});
      } else if (NO_BUFFER && !NO_HISTORY) {
        p_t = affine_transform({pbias, S, stack_summary, A, action_summary});
      } else if (!NO_BUFFER && NO_HISTORY) {
        p_t = affine_transform({pbias, S, stack_summary, T, term_summary});
      } else {
        p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
      }

      Expression nlp_t = rectify(p_t);
      //tworep*
      //Expression p_t = affine_transform({pbias, S, stack_lstm.back(), A, action_lstm.back()});
      //Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});

      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      unsigned action = 0;
      if (sample) {
        auto dist = as_vector(hg->incremental_forward());
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(dist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        action = current_valid_actions[w];
        const string& a = adict.Convert(action);
        if (a[0] == 'R') cerr << ")";
        if (a[0] == 'N') {
          int nt = action2NTindex[action];
          cerr << " (" << ntermdict.Convert(nt);
        }
      } else {
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
        //cerr << endl;
        ++action_count;
        log_probs.push_back(pick(adiste, action));
      }
      results.push_back(action);

      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      if (!NO_HISTORY) action_lstm.add_input(actione);

      // do action
      const string& actionString=adict.Convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac2 = actionString[1];

      bool was_word_completed = false;
      unsigned word_completed = 0;

      if (ac =='S' && ac2=='H') {  // SHIFT
        unsigned wordid = 0;
        //tworep
        //Expression p_t = affine_transform({pbias2, S2, stack_lstm.back(), A2, action_lstm.back(), T, term_lstm.back()});
        //Expression nlp_t = rectify(p_t);
        //tworep-oneact:
        //Expression p_t = affine_transform({pbias2, S2, stack_lstm.back(), T, term_lstm.back()});
        //Expression nlp_t = rectify(p_t);
        if (sample) {
          wordid = cfsm->sample(nlp_t);
          cerr << " " << termdict.Convert(wordid);
        } else {
          assert(termc < sent.size());
          wordid = sent.raw[termc];
          log_probs.push_back(-cfsm->neg_log_softmax(nlp_t, wordid));
        }
        assert (wordid != 0);
        stack_content.push_back(termdict.Convert(wordid)); //add the string of the word to the stack
        ++termc;
        Expression word = lookup(*hg, p_w, wordid);
        terms.push_back(word);
        if (!NO_BUFFER) term_lstm.add_input(word);
        stack.push_back(word);
        stack_lstm.add_input(word);
        is_open_paren.push_back(-1);
        if (WORD_COMPLETION_IS_SHIFT) {
          was_word_completed = (termc < sent.size());
          word_completed = termc;
        } else if (prev_a == 'S' || prev_a == 'R') {
          was_word_completed = true;
          word_completed = termc - 1;
        }
      } else if (ac == 'N') { // NT
        ++nopen_parens;
        auto it = action2NTindex.find(action);
        assert(it != action2NTindex.end());
        int nt_index = it->second;
        nt_count++;
        stack_content.push_back(ntermdict.Convert(nt_index));
        Expression nt_embedding = lookup(*hg, p_nt, nt_index);
        stack.push_back(nt_embedding);
        stack_lstm.add_input(nt_embedding);
        is_open_paren.push_back(nt_index);
        if (!WORD_COMPLETION_IS_SHIFT && (prev_a == 'S' || prev_a == 'R')) {
          was_word_completed = true;
          word_completed = termc;
        }
      } else { // REDUCE
        --nopen_parens;
        assert(stack.size() > 2); // dummy symbol means > 2 (not >= 2)
        assert(stack_content.size() > 2 && stack.size() == stack_content.size());
        // find what paren we are closing
        int i = is_open_paren.size() - 1; //get the last thing on the stack
        while(is_open_paren[i] < 0) { --i; assert(i >= 0); } //iteratively decide whether or not it's a non-terminal
        Expression nonterminal = lookup(*hg, p_ntup, is_open_paren[i]);
        int nchildren = is_open_paren.size() - i - 1;
        assert(nchildren > 0);
        //cerr << "  number of children to reduce: " << nchildren << endl;
        vector<Expression> children(nchildren);
        const_lstm_fwd.start_new_sequence();
        const_lstm_rev.start_new_sequence();

        // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
        // TO BE COMPOSED INTO A TREE EMBEDDING
        string curr_word;
        //cerr << "--------------------------------" << endl;
        //cerr << "Now printing the children" << endl;
        //cerr << "--------------------------------" << endl;
        for (i = 0; i < nchildren; ++i) {
          assert (stack_content.size() == stack.size());
          children[i] = stack.back();
          stack.pop_back();
          stack_lstm.rewind_one_step();
          is_open_paren.pop_back();
          curr_word = stack_content.back();
          //cerr << "At the back of the stack (supposed to be one of the children): " << curr_word << endl;
          stack_content.pop_back();
        }
        assert (stack_content.size() == stack.size());
        //cerr << "Doing REDUCE operation" << endl;
        is_open_paren.pop_back(); // nt symbol
        stack.pop_back(); // nonterminal dummy
        stack_lstm.rewind_one_step(); // nt symbol
        curr_word = stack_content.back();
        //cerr << "--------------------------------" << endl;
        //cerr << "At the back of the stack (supposed to be the non-terminal symbol) : " << curr_word << endl;
        stack_content.pop_back();
        assert (stack.size() == stack_content.size());
        //cerr << "Done reducing" << endl;

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
        stack_content.push_back(curr_word);
        //cerr << curr_word << endl;
        is_open_paren.push_back(-1); // we just closed a paren at this position
        if (stack.size() <= 2) {
          was_word_completed = true;
          word_completed = termc;
        }
      }

      prev_a = ac;

      /*
      if (was_word_completed) {
        Expression e_cum_neglogprob = -sum(log_probs);
        double cum_neglogprob = as_scalar(e_cum_neglogprob.value());
        cerr << "gold nlp after " << word_completed << "[" << log_probs.size() << "]: \t" << cum_neglogprob << endl;
        for (unsigned i = 0; i < log_probs.size(); i++) {
          cerr << as_scalar(log_probs[i].value()) << " ";
        }
        cerr << endl;
        print_parse(results, sent);
      }
      */


    }
    if (action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stack.size() == 2); // guard symbol, root
    if (!sample) {
      Expression tot_neglogprob = -sum(log_probs);
      assert(tot_neglogprob.pg != nullptr);
    }
    if (sample) cerr << "\n";
    return results;
  }

  vector<unsigned> log_prob_parser_beam(ComputationGraph* hg,
                                        const parser::Sentence& sent,
                                        int beam_size = 1) {
    //vector<unsigned> results;
    // vector<string> stack_content;
    // stack_content.push_back("ROOT_GUARD");

    stack_lstm.disable_dropout();
    if (!NO_BUFFER) term_lstm.disable_dropout();
    if (!NO_HISTORY) action_lstm.disable_dropout();
    const_lstm_fwd.disable_dropout();
    const_lstm_rev.disable_dropout();

    if (!NO_BUFFER) term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    if (!NO_HISTORY) action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    if (!NO_BUFFER) term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    if (!NO_HISTORY) action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    //Expression pbias2 = parameter(*hg, p_pbias2);
    //Expression S2 = parameter(*hg, p_S2);
    //Expression A2 = parameter(*hg, p_A2);

    //Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    //Expression w2l = parameter(*hg, p_w2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    BeamState initial_state;

    if (!NO_HISTORY) {
      action_lstm.add_input(action_start);
      initial_state.action_position = action_lstm.state();
    }

    initial_state.terms.push_back(lookup(*hg, p_w, kSOS));
    if (!NO_BUFFER) {
      term_lstm.add_input(initial_state.terms.back());
      initial_state.term_position = term_lstm.state();
    }

    initial_state.stack.push_back(parameter(*hg, p_stack_guard));

    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(initial_state.stack.back());
    initial_state.stack_position = stack_lstm.state();

    initial_state.is_open_paren.push_back(-1); // corresponds to dummy symbol
    initial_state.action_count = 0; // incremented at each prediction
    initial_state.nt_count = 0; // number of times an NT has been introduced
    initial_state.cons_nt = 0; // number of NTs we've opened with no intervening shifts or reduces
    vector<unsigned> current_valid_actions;
    initial_state.nopen_parens = 0;
    initial_state.prev_a = '0';
    initial_state.termc = 0;
    initial_state.score = 0.0;
    initial_state.action = 0;

    vector<BeamState> completed;

    vector<BeamState> beam;
    beam.push_back(initial_state);

    while(completed.size() < beam_size && !beam.empty()) {
      vector<BeamState> successors;

      // build successors for each item currently in the beam
      while (!beam.empty()) {
        BeamState current = beam.back();
        beam.pop_back();


        // assert (stack.size() == stack_content.size());
        // get list of possible actions for the current parser state
        current_valid_actions.clear();
        for (auto a: possible_actions) {
          if (IsActionForbidden_Generative(adict.Convert(a), current.prev_a, current.terms.size(), current.stack.size(), current.nopen_parens))
            continue;
          current_valid_actions.push_back(a);
        }
        //cerr << "valid actions = " << current_valid_actions.size() << endl;

        //onerep
        Expression stack_summary = stack_lstm.get_h(current.stack_position).back();
        Expression action_summary = (NO_HISTORY) ? Expression() : action_lstm.get_h(current.action_position).back();
        Expression term_summary = (NO_BUFFER) ? Expression() : term_lstm.get_h(current.term_position).back();
        // Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
        Expression p_t;
        if (NO_BUFFER && NO_HISTORY) {
          p_t = affine_transform({pbias, S, stack_summary});
        } else if (NO_BUFFER && !NO_HISTORY) {
          p_t = affine_transform({pbias, S, stack_summary, A, action_summary});
        } else if (!NO_BUFFER && NO_HISTORY) {
          p_t = affine_transform({pbias, S, stack_summary, T, term_summary});
        } else {
          p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
        }
        Expression nlp_t = rectify(p_t);
        Expression r_t = affine_transform({abias, p2a, nlp_t});

        Expression adiste = log_softmax(r_t, current_valid_actions);
        unsigned model_action = 0;
        auto dist = as_vector(hg->incremental_forward());

        assert(!current_valid_actions.empty());
        bool foundValidAction = false;
        for (unsigned i = 0; i < current_valid_actions.size(); i++) {
          unsigned possible_action = current_valid_actions[i];
          double score = dist[possible_action];
          const string &possibleActionString = adict.Convert(possible_action);

          unsigned ssize = current.stack.size();
          //assert(sent.size() + 1 >= termc );
          unsigned bsize = sent.size() + 1 - current.termc; // pretend we have a buffer guard
          bool is_shift = (possibleActionString[0] == 'S' && possibleActionString[1] == 'H');
          bool is_reduce = (possibleActionString[0] == 'R' && possibleActionString[1] == 'E');
          bool is_nt = (possibleActionString[0] == 'N');
          assert(is_shift || is_reduce || is_nt);
          static const unsigned MAX_OPEN_NTS = 100;
          if (is_nt && current.nopen_parens > MAX_OPEN_NTS) continue;
          if (is_nt && current.cons_nt >= MAX_CONS_NT) continue;
          bool skipRest = false;
          if (ssize == 1) {
            if (!is_nt) continue;
            skipRest = true;
          }

          if (!skipRest) {
            if (IMPLICIT_REDUCE_AFTER_SHIFT) {
              // if a SHIFT has an implicit REDUCE, then only shift after an NT:
              if (is_shift && current.prev_a != 'N') continue;
            }

            // be careful with top-level parens- you can only close them if you
            // have fully processed the buffer
            if (current.nopen_parens == 1 && bsize > 1) {
              if (IMPLICIT_REDUCE_AFTER_SHIFT && is_shift) continue;
              if (is_reduce) continue;
            }

            // you can't reduce after an NT action
            if (is_reduce && current.prev_a == 'N') continue;
            if (is_nt && bsize == 1) continue;
            if (is_shift && bsize == 1) continue;
            if (is_reduce && ssize < 3) continue;
          }

          // update only score, log_prob, and action for now
          // other state vars will be updated after pruning beam for efficiency
          BeamState successor = current;
          successor.log_probs.push_back(pick(adiste, possible_action));

          successor.results.push_back(possible_action);
          successor.action = possible_action;

          // update scores and log probs to account for generating the next word in the sent
          if (possibleActionString[0] == 'S' && possibleActionString[1] == 'H') {
            //assert(termc < sent.size());
            if (sent.raw[current.termc] == 0) {
              cerr << "sent.size(): " << sent.size() << endl;
              cerr << "sent.raw[termc] == 0" << endl;
              cerr << "termc: " << current.termc << endl;
              cerr << "sent.raw[termc]: " << termdict.Convert(sent.raw[current.termc]) << endl;
              for (unsigned i = 0; i < sent.raw.size(); i++) {
                cerr << termdict.Convert(sent.raw[i]) << " ";
              }
              cerr << endl;
            }

            Expression word_log_prob = -cfsm->neg_log_softmax(nlp_t, sent.raw[current.termc]);

            if (!IGNORE_WORD_IN_GREEDY) {
              score += as_scalar(word_log_prob.value());
              //score -= as_scalar(cfsm->neg_log_softmax(nlp_t, sent.raw[current.termc]).value());
            }
            successor.log_probs.push_back(word_log_prob);
          }

          successor.score += score;
          ++successor.action_count;

          successors.push_back(successor);

          foundValidAction = true;
        }
        if (!foundValidAction) {
          cerr << "sentence:" << endl;
          for (unsigned i = 0; i < sent.unk.size(); i++) {
            cerr << termdict.Convert(sent.unk[i]) << " ";
          }
          cerr << endl;
          cerr << "termc: " << current.termc << endl;
          cerr << "sent.size(): " << sent.size() << endl;
          cerr << "previous action: " << current.prev_a << endl;
          cerr << "terms.size(): " << current.terms.size() << endl;
          cerr << "stack.size(): " << current.stack.size() << endl;
          cerr << "nopen_parens: " <<  current.nopen_parens << endl;
          cerr << endl;
          cerr << "possible actions:" << endl;
          for (unsigned i = 0; i < current_valid_actions.size(); i++) {
            auto action = current_valid_actions[i];
            const string& actionString=adict.Convert(action);
            cerr << actionString << endl;
          }
          cerr << "actions so far:" << endl;
          for (unsigned i = 0; i < current.results.size(); i++) {
            auto action = current.results[i];
            const string& actionString=adict.Convert(action);
            cerr << actionString << endl;
          }

        }
        assert(foundValidAction);
      }

      // cut down to beam size
      prune(successors, beam_size);

      if (successors.size() == 0) {
        cerr << "warning: successors empty" << endl;
      }

      // update state variables for top K
      // check if any of the successors are complete; add others back to the beam
      for (unsigned i = 0; i < successors.size(); i++) {
        BeamState successor = successors[i];
        unsigned action = successor.action;

        // add current action to action LSTM
        Expression actione = lookup(*hg, p_a, action);
        if (!NO_HISTORY) {
          action_lstm.add_input(successor.action_position, actione);
          successor.action_position = action_lstm.state();
        }

        // do action
        const string& actionString=adict.Convert(action);
        //cerr << "ACT: " << actionString << endl;
        const char ac = actionString[0];
        const char ac2 = actionString[1];
        successor.prev_a = ac;

        if (ac =='S' && ac2=='H') {  // SHIFT
          unsigned wordid = 0;
          assert(successor.termc < sent.size());
          wordid = sent.raw[successor.termc];

          if (wordid == 0) {
            cerr << "wordid == 0" << endl;
            cerr << "termc: " << successor.termc << endl;
            cerr << "sent.raw[termc]: " << termdict.Convert(sent.raw[successor.termc]) << endl;
            for (unsigned i = 0; i < sent.raw.size(); i++) {
              cerr << termdict.Convert(sent.raw[i]) << " ";
            }
            cerr << endl;
          }
          //successor.log_probs.push_back(-cfsm->neg_log_softmax(nlp_t, wordid));
          assert (wordid != 0);
          // stack_content.push_back(termdict.Convert(wordid)); //add the string of the word to the stack
          ++successor.termc;
          Expression word = lookup(*hg, p_w, wordid);
          successor.terms.push_back(word);
          if (!NO_BUFFER) {
            term_lstm.add_input(successor.term_position, word);
            successor.term_position = term_lstm.state();
          }

          successor.stack.push_back(word);
          stack_lstm.add_input(successor.stack_position, word);
          successor.stack_position = stack_lstm.state();

          successor.is_open_paren.push_back(-1);
          successor.cons_nt = 0;

          /*
          if (successor.termc == 1) {
              Expression e_cum_neglogprob = -sum(successor.log_probs);
              double cum_neglogprob = as_scalar(e_cum_neglogprob.value());
              cerr << "pred nlp after " << successor.termc << ": \t" << cum_neglogprob << endl;
          }
          */
        } else if (ac == 'N') { // NT
          ++successor.nopen_parens;
          auto it = action2NTindex.find(action);
          assert(it != action2NTindex.end());
          int nt_index = it->second;
          successor.nt_count++;
          // stack_content.push_back(ntermdict.Convert(nt_index));
          Expression nt_embedding = lookup(*hg, p_nt, nt_index);
          successor.stack.push_back(nt_embedding);
          stack_lstm.add_input(successor.stack_position,nt_embedding);
          successor.stack_position = stack_lstm.state();
          successor.is_open_paren.push_back(nt_index);
          successor.cons_nt += 1;
        } else { // REDUCE
          --successor.nopen_parens;
          assert(successor.stack.size() > 2); // dummy symbol means > 2 (not >= 2)
          // assert(stack_content.size() > 2 && stack.size() == stack_content.size());
          // find what paren we are closing
          int i = successor.is_open_paren.size() - 1; //get the last thing on the stack
          while(successor.is_open_paren[i] < 0) { --i; assert(i >= 0); } //iteratively decide whether or not it's a non-terminal
          Expression nonterminal = lookup(*hg, p_ntup, successor.is_open_paren[i]);
          int nchildren = successor.is_open_paren.size() - i - 1;
          assert(nchildren > 0);
          //cerr << "  number of children to reduce: " << nchildren << endl;
          vector<Expression> children(nchildren);
          const_lstm_fwd.start_new_sequence();
          const_lstm_rev.start_new_sequence();

          // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
          // TO BE COMPOSED INTO A TREE EMBEDDING
          string curr_word;
          //cerr << "--------------------------------" << endl;
          //cerr << "Now printing the children" << endl;
          //cerr << "--------------------------------" << endl;
          for (i = 0; i < nchildren; ++i) {
            // assert (stack_content.size() == stack.size());
            children[i] = successor.stack.back();
            successor.stack.pop_back();
            // stack_lstm.rewind_one_step();
            successor.stack_position = stack_lstm.head_of(successor.stack_position);
            successor.is_open_paren.pop_back();
            // curr_word = stack_content.back();
            //cerr << "At the back of the stack (supposed to be one of the children): " << curr_word << endl;
            // stack_content.pop_back();
          }
          // assert (stack_content.size() == stack.size());
          //cerr << "Doing REDUCE operation" << endl;
          successor.is_open_paren.pop_back(); // nt symbol
          successor.stack.pop_back(); // nonterminal dummy
          // stack_lstm.rewind_one_step(); // nt symbol
          successor.stack_position = stack_lstm.head_of(successor.stack_position); // nt symbol
          // curr_word = stack_content.back();
          //cerr << "--------------------------------" << endl;
          //cerr << "At the back of the stack (supposed to be the non-terminal symbol) : " << curr_word << endl;
          // stack_content.pop_back();
          // assert (stack.size() == stack_content.size());
          //cerr << "Done reducing" << endl;

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
          // stack_content.push_back(curr_word);
          //cerr << curr_word << endl;
          successor.is_open_paren.push_back(-1); // we just closed a paren at this position
          successor.cons_nt = 0;
        } // end REDUCE

        if (successor.stack.size() <= 2 && successor.termc != 0) {
          completed.push_back(successor);
        } else {
          beam.push_back(successor);
        }
      } // end successor iteration
    } // end build completed

    sort(completed.begin(), completed.end(), BeamStateCompare());

    BeamState best = completed[0];

    assert(best.stack.size() == 2); // guard symbol, root
    Expression tot_neglogprob = -sum(best.log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return best.results;
  }

  vector<unsigned> log_prob_parser_beam_within_word(ComputationGraph* hg,
                                        const parser::Sentence& sent,
                                        int beam_size = 1,
                                        int beam_filter_at_word_size = -1) {
    if (beam_filter_at_word_size < 0)
      beam_filter_at_word_size = beam_size;
    //cerr << "beam size: " << beam_size << endl;
    //cerr << "beam filter at word size: " << beam_filter_at_word_size << endl;
    //vector<unsigned> results;
    // vector<string> stack_content;
    // stack_content.push_back("ROOT_GUARD");

    stack_lstm.disable_dropout();
    if (!NO_BUFFER) term_lstm.disable_dropout();
    if (!NO_HISTORY) action_lstm.disable_dropout();
    const_lstm_fwd.disable_dropout();
    const_lstm_rev.disable_dropout();

    if (!NO_BUFFER) term_lstm.new_graph(*hg);
    stack_lstm.new_graph(*hg);
    if (!NO_HISTORY) action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    cfsm->new_graph(*hg);
    if (!NO_BUFFER) term_lstm.start_new_sequence();
    stack_lstm.start_new_sequence();
    if (!NO_HISTORY) action_lstm.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression A = parameter(*hg, p_A);
    Expression T = parameter(*hg, p_T);
    //Expression pbias2 = parameter(*hg, p_pbias2);
    //Expression S2 = parameter(*hg, p_S2);
    //Expression A2 = parameter(*hg, p_A2);

    //Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    //Expression w2l = parameter(*hg, p_w2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);
    Expression cW = parameter(*hg, p_cW);

    BeamState initial_state;

    if (!NO_HISTORY) {
      action_lstm.add_input(action_start);
      initial_state.action_position = action_lstm.state();
    }

    initial_state.terms.push_back(lookup(*hg, p_w, kSOS));
    if (!NO_BUFFER) {
      term_lstm.add_input(initial_state.terms.back());
      initial_state.term_position = term_lstm.state();
    }

    initial_state.stack.push_back(parameter(*hg, p_stack_guard));

    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(initial_state.stack.back());
    initial_state.stack_position = stack_lstm.state();

    initial_state.is_open_paren.push_back(-1); // corresponds to dummy symbol
    initial_state.action_count = 0; // incremented at each prediction
    initial_state.nt_count = 0; // number of times an NT has been introduced
    initial_state.cons_nt = 0; // number of NTs we've opened with no intervening shifts or reduces
    vector<unsigned> current_valid_actions;
    initial_state.nopen_parens = 0;
    initial_state.prev_a = '0';
    initial_state.termc = 0;
    initial_state.score = 0.0;
    initial_state.action = 0;

    vector<BeamState> completed;

    vector<BeamState> beam;
    beam.push_back(initial_state);

    for (unsigned current_termc = 0; current_termc < sent.size(); current_termc++) {
      completed.clear();

      while (completed.size() < beam_size && !beam.empty()) {
        vector<BeamState> successors;

        // build successors for each item currently in the beam
        while (!beam.empty()) {
          BeamState current = beam.back();
          beam.pop_back();


          // assert (stack.size() == stack_content.size());
          // get list of possible actions for the current parser state
          current_valid_actions.clear();
          for (auto a: possible_actions) {
            if (IsActionForbidden_Generative(adict.Convert(a), current.prev_a, current.terms.size(),
                                             current.stack.size(), current.nopen_parens))
              continue;
            current_valid_actions.push_back(a);
          }
          //cerr << "valid actions = " << current_valid_actions.size() << endl;

          //onerep
          Expression stack_summary = stack_lstm.get_h(current.stack_position).back();
          Expression action_summary = (NO_HISTORY) ? Expression() : action_lstm.get_h(current.action_position).back();
          Expression term_summary = (NO_BUFFER) ? Expression() : term_lstm.get_h(current.term_position).back();
          //Expression p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
          Expression p_t;
          if (NO_BUFFER && NO_HISTORY) {
            p_t = affine_transform({pbias, S, stack_summary});
          } else if (NO_BUFFER && !NO_HISTORY) {
            p_t = affine_transform({pbias, S, stack_summary, A, action_summary});
          } else if (!NO_BUFFER && NO_HISTORY) {
            p_t = affine_transform({pbias, S, stack_summary, T, term_summary});
          } else {
            p_t = affine_transform({pbias, S, stack_summary, A, action_summary, T, term_summary});
          }
          Expression nlp_t = rectify(p_t);
          Expression r_t = affine_transform({abias, p2a, nlp_t});

          Expression adiste = log_softmax(r_t, current_valid_actions);
          unsigned model_action = 0;
          auto dist = as_vector(hg->incremental_forward());

          assert(!current_valid_actions.empty());
          bool foundValidAction = false;
          for (unsigned i = 0; i < current_valid_actions.size(); i++) {
            unsigned possible_action = current_valid_actions[i];
            double score = dist[possible_action];
            const string &possibleActionString = adict.Convert(possible_action);

            unsigned ssize = current.stack.size();
            //assert(sent.size() + 1 >= termc );
            unsigned bsize = sent.size() + 1 - current.termc; // pretend we have a buffer guard
            bool is_shift = (possibleActionString[0] == 'S' && possibleActionString[1] == 'H');
            bool is_reduce = (possibleActionString[0] == 'R' && possibleActionString[1] == 'E');
            bool is_nt = (possibleActionString[0] == 'N');
            assert(is_shift || is_reduce || is_nt);
            static const unsigned MAX_OPEN_NTS = 100;
            if (is_nt && current.nopen_parens > MAX_OPEN_NTS) {
                //cerr << "more than max open" << endl;
                continue;
            }
            if (is_nt && current.cons_nt >= MAX_CONS_NT) {
                //cerr << "more than max cons: " << current.cons_nt << " " << MAX_CONS_NT << endl;
                continue;
            }
            bool skipRest = false;
            if (ssize == 1) {
              if (!is_nt) {
                  continue;
              }
              skipRest = true;
            }

            if (!skipRest) {
              if (IMPLICIT_REDUCE_AFTER_SHIFT) {
                // if a SHIFT has an implicit REDUCE, then only shift after an NT:
                if (is_shift && current.prev_a != 'N') continue;
              }

              // be careful with top-level parens- you can only close them if you
              // have fully processed the buffer
              if (current.nopen_parens == 1 && bsize > 1) {
                if (IMPLICIT_REDUCE_AFTER_SHIFT && is_shift) continue;
                if (is_reduce) continue;
              }

              // you can't reduce after an NT action
              if (is_reduce && current.prev_a == 'N') continue;
              if (is_nt && bsize == 1) continue;
              if (is_shift && bsize == 1) continue;
              if (is_reduce && ssize < 3) continue;
            }

            // update only score, log_prob, and action for now
            // other state vars will be updated after pruning beam for efficiency
            BeamState successor = current;
            successor.log_probs.push_back(pick(adiste, possible_action));

            successor.results.push_back(possible_action);
            successor.action = possible_action;

            // update scores and log probs to account for generating the next word in the sent
            if (possibleActionString[0] == 'S' && possibleActionString[1] == 'H') {
              //assert(termc < sent.size());
              if (sent.raw[current.termc] == 0) {
                cerr << "sent.size(): " << sent.size() << endl;
                cerr << "sent.raw[termc] == 0" << endl;
                cerr << "termc: " << current.termc << endl;
                cerr << "sent.raw[termc]: " << termdict.Convert(sent.raw[current.termc]) << endl;
                for (unsigned i = 0; i < sent.raw.size(); i++) {
                  cerr << termdict.Convert(sent.raw[i]) << " ";
                }
                cerr << endl;
              }

              Expression word_log_prob = -cfsm->neg_log_softmax(nlp_t, sent.raw[current.termc]);

              if (!IGNORE_WORD_IN_GREEDY) {
                score += as_scalar(word_log_prob.value());
                //score -= as_scalar(cfsm->neg_log_softmax(nlp_t, sent.raw[current.termc]).value());
              }
              successor.log_probs.push_back(word_log_prob);
            }

            successor.score += score;
            ++successor.action_count;

            successors.push_back(successor);

            foundValidAction = true;
          }
          if (!foundValidAction) {
            cerr << "sentence:" << endl;
            for (unsigned i = 0; i < sent.unk.size(); i++) {
              cerr << termdict.Convert(sent.unk[i]) << " ";
            }
            cerr << endl;
            cerr << "termc: " << current.termc << endl;
            cerr << "sent.size(): " << sent.size() << endl;
            cerr << "previous action: " << current.prev_a << endl;
            cerr << "terms.size(): " << current.terms.size() << endl;
            cerr << "stack.size(): " << current.stack.size() << endl;
            cerr << "nopen_parens: " << current.nopen_parens << endl;
            cerr << endl;
            cerr << "possible actions:" << endl;
            for (unsigned i = 0; i < current_valid_actions.size(); i++) {
              auto action = current_valid_actions[i];
              const string &actionString = adict.Convert(action);
              cerr << actionString << endl;
            }
            cerr << "actions so far:" << endl;
            for (unsigned i = 0; i < current.results.size(); i++) {
              auto action = current.results[i];
              const string &actionString = adict.Convert(action);
              cerr << actionString << endl;
            }

          }
          assert(foundValidAction);
        }

        // cut down to beam size
        prune(successors, beam_size);

        if (successors.size() == 0) {
          cerr << "warning: successors empty" << endl;
        }

        // update state variables for top K
        // check if any of the successors are complete; add others back to the beam
        for (unsigned i = 0; i < successors.size(); i++) {
          BeamState successor = successors[i];
          unsigned action = successor.action;

          // add current action to action LSTM
          Expression actione = lookup(*hg, p_a, action);
          if (!NO_HISTORY) {
            action_lstm.add_input(successor.action_position, actione);
            successor.action_position = action_lstm.state();
          }

          // do action
          const string &actionString = adict.Convert(action);
          //cerr << "ACT: " << actionString << endl;
          const char ac = actionString[0];
          const char ac2 = actionString[1];
          successor.prev_a = ac;

          if (ac == 'S' && ac2 == 'H') {  // SHIFT
            unsigned wordid = 0;
            assert(successor.termc < sent.size());
            wordid = sent.raw[successor.termc];

            if (wordid == 0) {
              cerr << "wordid == 0" << endl;
              cerr << "termc: " << successor.termc << endl;
              cerr << "sent.raw[termc]: " << termdict.Convert(sent.raw[successor.termc]) << endl;
              for (unsigned i = 0; i < sent.raw.size(); i++) {
                cerr << termdict.Convert(sent.raw[i]) << " ";
              }
              cerr << endl;
            }
            //successor.log_probs.push_back(-cfsm->neg_log_softmax(nlp_t, wordid));
            assert (wordid != 0);
            // stack_content.push_back(termdict.Convert(wordid)); //add the string of the word to the stack
            ++successor.termc;
            Expression word = lookup(*hg, p_w, wordid);
            successor.terms.push_back(word);
            if (!NO_BUFFER) {
              term_lstm.add_input(successor.term_position, word);
              successor.term_position = term_lstm.state();
            }

            successor.stack.push_back(word);
            stack_lstm.add_input(successor.stack_position, word);
            successor.stack_position = stack_lstm.state();

            successor.is_open_paren.push_back(-1);

            successor.cons_nt = 0;

            /*
            if (successor.termc == 1) {
              Expression e_cum_neglogprob = -sum(successor.log_probs);
              double cum_neglogprob = as_scalar(e_cum_neglogprob.value());
              cerr << "pred nlp at word 0: \t" << cum_neglogprob << "\t";
            }
            */
          } else if (ac == 'N') { // NT
            ++successor.nopen_parens;
            auto it = action2NTindex.find(action);
            assert(it != action2NTindex.end());
            int nt_index = it->second;
            successor.nt_count++;
            // stack_content.push_back(ntermdict.Convert(nt_index));
            Expression nt_embedding = lookup(*hg, p_nt, nt_index);
            successor.stack.push_back(nt_embedding);
            stack_lstm.add_input(successor.stack_position, nt_embedding);
            successor.stack_position = stack_lstm.state();
            successor.is_open_paren.push_back(nt_index);

            successor.cons_nt += 1;
          } else { // REDUCE
            --successor.nopen_parens;
            assert(successor.stack.size() > 2); // dummy symbol means > 2 (not >= 2)
            // assert(stack_content.size() > 2 && stack.size() == stack_content.size());
            // find what paren we are closing
            int i = successor.is_open_paren.size() - 1; //get the last thing on the stack
            while (successor.is_open_paren[i] < 0) {
              --i;
              assert(i >= 0);
            } //iteratively decide whether or not it's a non-terminal
            Expression nonterminal = lookup(*hg, p_ntup, successor.is_open_paren[i]);
            int nchildren = successor.is_open_paren.size() - i - 1;
            assert(nchildren > 0);
            //cerr << "  number of children to reduce: " << nchildren << endl;
            vector<Expression> children(nchildren);
            const_lstm_fwd.start_new_sequence();
            const_lstm_rev.start_new_sequence();

            // REMOVE EVERYTHING FROM THE STACK THAT IS GOING
            // TO BE COMPOSED INTO A TREE EMBEDDING
            string curr_word;
            //cerr << "--------------------------------" << endl;
            //cerr << "Now printing the children" << endl;
            //cerr << "--------------------------------" << endl;
            for (i = 0; i < nchildren; ++i) {
              // assert (stack_content.size() == stack.size());
              children[i] = successor.stack.back();
              successor.stack.pop_back();
              // stack_lstm.rewind_one_step();
              successor.stack_position = stack_lstm.head_of(successor.stack_position);
              successor.is_open_paren.pop_back();
              // curr_word = stack_content.back();
              //cerr << "At the back of the stack (supposed to be one of the children): " << curr_word << endl;
              // stack_content.pop_back();
            }
            // assert (stack_content.size() == stack.size());
            //cerr << "Doing REDUCE operation" << endl;
            successor.is_open_paren.pop_back(); // nt symbol
            successor.stack.pop_back(); // nonterminal dummy
            // stack_lstm.rewind_one_step(); // nt symbol
            successor.stack_position = stack_lstm.head_of(successor.stack_position); // nt symbol
            // curr_word = stack_content.back();
            //cerr << "--------------------------------" << endl;
            //cerr << "At the back of the stack (supposed to be the non-terminal symbol) : " << curr_word << endl;
            // stack_content.pop_back();
            // assert (stack.size() == stack_content.size());
            //cerr << "Done reducing" << endl;

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
            // stack_content.push_back(curr_word);
            //cerr << curr_word << endl;
            successor.is_open_paren.push_back(-1); // we just closed a paren at this position

            successor.cons_nt = 0;
          } // end REDUCE

          bool termc_completed = false;

          if (WORD_COMPLETION_IS_SHIFT) {
            // handle special case, we're only totally done with the last word if we close all parens
            if (current_termc == sent.size() - 1)
              // TODO: possibly simpler if we check # of open parens?
              termc_completed = successor.stack.size() <= 2 && successor.termc > current_termc;
            else
              termc_completed = (ac == 'S');
          } else {
            // word is completed when we've decided to finish reducing, either by shifting next word or opening NT (or closing all open parens if we're at the end)
            if (current_termc == sent.size() - 1) { // at the end
              termc_completed = (successor.stack.size() <= 2) && (successor.termc > current_termc); // guard symbol, root
              if (termc_completed) {
                if (successor.termc != sent.size()) {
                  cerr << "current_termc: " << current_termc << endl;
                  cerr << "termc: " << successor.termc << endl;
                  cerr << "stack.size: " << successor.stack.size() << endl;
                }
                assert(successor.termc == sent.size());
              }
            } else { // not at the end, check for shift or NT
              if (ac == 'S' && ac2 == 'H') {  // SHIFT
                // this termc is completed if the current action shifted the next word
                termc_completed = (successor.termc > current_termc + 1);
                if (termc_completed) assert(successor.termc == current_termc + 2);
              } else if (ac == 'N') {
                // this termc is completed if the current action
                termc_completed = (successor.termc > current_termc);
                if (termc_completed) assert(successor.termc == current_termc + 1);
              }
              // reduce never completes the current word
            }
          }

          if (termc_completed) {
            completed.push_back(successor);
          } else {
            beam.push_back(successor);
          }
        } // end successor iteration
      } // end build completed for current_termc

      prune(completed, beam_filter_at_word_size);

      BeamState best_completed = completed[0];
      Expression e_cum_neglogprob = -sum(best_completed.log_probs);
      double cum_neglogprob = as_scalar(e_cum_neglogprob.value());
      unsigned last_termc;
      if (!WORD_COMPLETION_IS_SHIFT && best_completed.prev_a == 'S')
          last_termc = best_completed.termc - 1;
      else 
          last_termc = best_completed.termc;
      /*
      cerr << "best nlp after " << last_termc << "[" << best_completed.log_probs.size() << "]: \t" << cum_neglogprob << endl;
      for (unsigned i = 0; i < best_completed.log_probs.size(); i++) {
        cerr << as_scalar(best_completed.log_probs[i].value()) << " ";
      }
      cerr << endl;
      print_parse(best_completed.results, sent, false, cerr);
       */

      // replace beam with completed, for next termc
      beam.clear();
      for (unsigned i = 0; i < completed.size(); i++) {
        beam.push_back(completed[i]);
      }
    } // end current_termc increment

    sort(completed.begin(), completed.end(), BeamStateCompare());

    BeamState best = completed[0];

    assert(best.stack.size() == 2); // guard symbol, root
    Expression tot_neglogprob = -sum(best.log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return best.results;
  }

};

void signal_callback_handler(int signum) {
  if (signum == SIGINT) {
    if (requested_stop) {
        cerr << "\nReceived SIGINT again, quitting.\n";
        _exit(1);
    }
    cerr << "\nReceived SIGINT terminating optimization early...\n";
    requested_stop = true;
  } else if (signum == SIGSEGV) {
    fprintf(stderr, "Error: signal %d:\n", signum);
    void *array[255];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, sizeof(array) / sizeof(void *));

    // print out all the frames to stderr
    fprintf(stderr, "stacktrace:\n", signum);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);

  }
}

int main(int argc, char** argv) {
  signal(SIGSEGV, signal_callback_handler);
  unsigned random_seed = cnn::Initialize(argc, argv);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  if (conf.count("clusters") == 0) {
    cerr << "Please specify vocabulary clustering with --clusters FILE when training generative model\n";
    return 1;
  }
  if(conf.count("beam_within_word") && !conf.count("greedy_decode_dev")) {
    cerr << "Must specify greedy_decode_dev when passing beam_within_word" << endl;
    return 1;
  }
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  MAX_CONS_NT = conf["max_cons_nt"].as<unsigned>();

  SILVER_BLOCKS_PER_GOLD = conf["silver_blocks_per_gold"].as<unsigned>();

  cerr << "max cons nt: " << MAX_CONS_NT << endl;

  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }

  IGNORE_WORD_IN_GREEDY = conf.count("ignore_word_in_greedy");

  WORD_COMPLETION_IS_SHIFT = conf.count("word_completion_is_shift");

  NO_HISTORY = conf.count("no_history");
  NO_BUFFER = conf.count("no_buffer");

  ostringstream os;
  os << "ntparse_gen"
     << "_D" << DROPOUT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << (NO_HISTORY ? "_no-history" : "")
     << (NO_BUFFER ? "_no-buffer" : "")
     << "-seed" << random_seed
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  kSOS = termdict.Convert("<s>");
  Model model;
  cfsm = new ClassFactoredSoftmaxBuilder(HIDDEN_DIM, conf["clusters"].as<string>(), &termdict, &model);

  parser::TopDownOracleGen corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracleGen dev_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracleGen2 test_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  parser::TopDownOracleGen gold_corpus(&termdict, &adict, &posdict, &non_unked_termdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>());
  if (conf.count("bracketing_dev_data")) {
    corpus.load_bdata(conf["bracketing_dev_data"].as<string>());
  }

  bool has_gold_training_data = false;

  if (conf.count("gold_training_data")) {
    gold_corpus.load_oracle(conf["gold_training_data"].as<string>());
    if (conf.count("bracketing_dev_data")) {
      gold_corpus.load_bdata(conf["bracketing_dev_data"].as<string>());
    }
    has_gold_training_data = true;
  }

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze();

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>());
  }

  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>());
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
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  ParserBuilder parser(&model, pretrained);



  SimpleSGDTrainer sgd(&model);
  cerr << "using sgd for training" << endl;

  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    assert(in);
    boost::archive::binary_iarchive ia(in);
    ia >> model >> sgd;
    // TODO: figure out deserialization ordering
    // ia >> termdict >> adict >> ntermdict >> posdict;
  }

  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    //AdamTrainer sgd(&model);
    //sgd.eta = 0.01;
    //sgd.eta0 = 0.01;
    //MomentumSGDTrainer sgd(&model);
    sgd.eta_decay = 0.08;
    //sgd.eta_decay = 0.05;

    unsigned tot_seen = 0;
    int iter = -1;

    double best_dev_llh = 9e99;
    double bestf1=0.0;

    int logc = 0;

    auto train_block = [&](const parser::TopDownOracleGen& corpus, vector<unsigned>::iterator indices_begin, vector<unsigned>::iterator indices_end, int epoch_size) {
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

          ++logc;
          if (logc > 50) {
            // generate random sample
            ComputationGraph cg;
            double x;
            // sample tree and sentence
            parser.log_prob_parser(&cg, parser::Sentence(), vector<int>(),&x,true);
          }
          if (logc % 100 == 0) { // report on dev set
            unsigned dev_size = dev_corpus.size();
            double llh = 0;
            double trs = 0;
            double right = 0;
            double dwords = 0;
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
              double lp = 0;
              trs += actions.size();
            }
            auto t_end = chrono::high_resolution_clock::now();
            double err = (trs - right) / trs;
            //parser::EvalBResults res = parser::Evaluate("foo", pfx);
            cerr << "  **dev (iter=" << iter << " epoch=" << (static_cast<double>(tot_seen) / epoch_size) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
            if (llh < best_dev_llh) {
              cerr << "  new best...writing model to " << fname << ".bin ...\n";
              best_dev_llh = llh;
              ofstream out(fname + ".bin");
              boost::archive::binary_oarchive oa(out);
              oa << model << sgd;
              oa << termdict << adict << ntermdict << posdict;
              // oa << model;
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
      parser::TopDownOracleGen* main_corpus = &corpus;

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
  if (conf.count("greedy_decode_dev")) { // do test evaluation
    unsigned start_index = 0;
    unsigned stop_index = dev_corpus.size();
    unsigned block_count = conf["block_count"].as<unsigned>();
    unsigned block_num = conf["block_num"].as<unsigned>();

    if (block_count > 0) {
      assert(block_num < block_count);
      unsigned q = dev_corpus.size() / block_count;
      unsigned r = dev_corpus.size() % block_count;
      start_index = q * block_num + min(block_num, r);
      stop_index = q * (block_num + 1) + min(block_num + 1, r);
    }

    cerr << "decoding sentences " << start_index << " .. " << (stop_index - 1) << endl;


    ostringstream os;
    if (conf.count("dev_output_file")) {
      os << conf["dev_output_file"].as<string>();
    } else {
      os << "/tmp/parser_dev_eval." << getpid() << ".txt";
    }
    if (block_count > 0) {
      os << "_block-" << block_num;
    }
    const string pfx = os.str();
    cerr << "writing to " << pfx << endl;
    ofstream out(pfx.c_str());
    auto t_start = chrono::high_resolution_clock::now();
    for (unsigned sii = start_index; sii < stop_index; ++sii) {
      auto t_sentence_start =  chrono::high_resolution_clock::now();
      const auto &sentence = dev_corpus.sents[sii];
      const vector<int> &actions = dev_corpus.actions[sii];
      cerr << endl;
      cerr << endl << "sentence: " << sii << endl;
      cerr << "gold:\t";
      print_parse(vector<unsigned>(actions.begin(), actions.end()), sentence, true, cerr);
      {
        ComputationGraph hg;
        parser.log_prob_parser(&hg, sentence, actions, nullptr, true);
        double nlp = as_scalar(hg.incremental_forward());
        cerr << "gold score:\t" << -nlp << endl;
      }
      vector<unsigned> pred;
      double pred_nlp;
      {
        ComputationGraph hg;
        // greedy predict
        if (conf.count("beam_within_word"))
          pred = parser.log_prob_parser_beam_within_word(&hg,
                                                         sentence,
                                                         conf["decode_beam_size"].as<unsigned>(),
                                                         conf["decode_beam_filter_at_word_size"].as<int>());
        else
          pred = parser.log_prob_parser_beam(&hg, sentence, conf["decode_beam_size"].as<unsigned>());

        pred_nlp = as_scalar(hg.incremental_forward());
      }
      vector<int> pred_int = vector<int>(pred.begin(), pred.end());
      cerr << "pred:\t";
      print_parse(pred, sentence, true, cerr);
      cerr << "pred score:\t" << -pred_nlp << endl;
      {
        // rescore, to check for errors in beam search scoring
        ComputationGraph hg;
        // get log likelihood of gold
        parser.log_prob_parser(&hg, sentence, pred_int, nullptr, true);
        double pred_rescore = -as_scalar(hg.incremental_forward());
        cerr << "pred rescore:\t" << pred_rescore << endl;
      }
      cerr << "match?:\t" << (pred_int == actions ? "True" : "False");

      // print decode to file
      print_parse(pred, sentence, true, out);
      double lp = 0;
    }
    auto t_end = chrono::high_resolution_clock::now();
    out.close();
    cerr << "Test output in " << pfx << endl;
    //parser::EvalBResults res = parser::Evaluate("foo", pfx);
    std::string evaluable_fname = pfx + "_evaluable.txt";
    std::string evalbout_fname = pfx + "_evalbout.txt";
    std::string command="python remove_dev_unk.py "+ corpus.devdata +" "+pfx+" > " + evaluable_fname;
    const char *cmd = command.c_str();
    system(cmd);

    std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" " + evaluable_fname + " > " + evalbout_fname;
    const char *cmd2 = command2.c_str();

    system(cmd2);

    std::ifstream evalfile(evalbout_fname);
    std::string lineS;
    std::string brackstr = "Bracketing FMeasure";
    double newfmeasure = 0.0;
    std::string strfmeasure = "";
    bool found = 0;
    while (getline(evalfile, lineS) && !newfmeasure) {
      if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
        //std::cout<<lineS<<"\n";
        strfmeasure = lineS.substr(lineS.size() - 5, lineS.size());
        std::string::size_type sz;
        newfmeasure = std::stod(strfmeasure, &sz);
        //std::cout<<strfmeasure<<"\n";
      }
    }

    cerr << "F1score: " << newfmeasure << "\n";

  }
  if (test_corpus.size() > 0) {
    // if rescoring, we may have many repeats, cache them
    unordered_map<vector<int>, unordered_map<vector<int>, double, boost::hash<vector<int>>>, boost::hash<vector<int>>> s2a2p;
    unsigned test_size = test_corpus.size();
    double llh = 0;
    double right = 0;
    double dwords = 0;
    for (unsigned sii = 0; sii < test_size; ++sii) {
      const auto &sentence = test_corpus.sents[sii];
      const vector<int> &actions = test_corpus.actions[sii];
      dwords += sentence.size();
      double &lp = s2a2p[sentence.raw][actions];
      if (!lp) {
        ComputationGraph hg;
        parser.log_prob_parser(&hg, sentence, actions, &right, true);
        lp = as_scalar(hg.incremental_forward());
      }
      cout << sentence.size() << '\t' << lp << endl;
      llh += lp;
    }
    cerr << "test     total -llh=" << llh << endl;
    cerr << "test ppl (per word)=" << exp(llh / dwords) << endl;
  }
}
