//
// Created by dfried on 12/29/17.
//
#include "nt-parser/tree.h"
#include "nt-parser/oracle.h"
#include "nt-parser/stack.h"

#ifndef CNN_NT_PARSER_H
#define CNN_NT_PARSER_H

struct AbstractParserState {
  virtual bool is_finished() const = 0;
  virtual vector<unsigned> get_valid_actions() const = 0;
  virtual Expression get_action_log_probs(const vector<unsigned>& valid_actions) const = 0;
  virtual std::shared_ptr<AbstractParserState> perform_action(unsigned action) const = 0;
  virtual void finish_sentence() const = 0;
  virtual bool word_completed() const = 0;

  virtual bool action_is_valid(unsigned action) const = 0;
  virtual Stack<Bracket> get_completed_brackets() const = 0;
  virtual Stack<OpenBracket> get_open_brackets() const = 0;
  virtual unsigned get_words_shifted() const = 0;
};

struct SymbolicParserState;
struct ParserState;
struct EnsembledParserState;

struct AbstractParser;
struct ParserBuilder;
struct EnsembledParser;

vector<Bracket> complete_actions_to_brackets(const parser::Sentence& sentence, const vector<int>& actions);

struct DynamicOracle {
  enum class ExplorationType { none, greedy, sample };

  DynamicOracle(const vector<Bracket>& gold_brackets):
          gold_brackets(gold_brackets) {}

  DynamicOracle(const parser::Sentence& sentence, const vector<int>& actions):
          DynamicOracle(complete_actions_to_brackets(sentence, actions)) {}

  const vector<Bracket> gold_brackets;

  unsigned oracle_action(const AbstractParserState& parser_state);
};

#endif //CNN_NT_PARSER_H
