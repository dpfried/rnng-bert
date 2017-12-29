//
// Created by dfried on 11/24/17.
// mostly a port of https://github.com/jhcross/span-parser/blob/master/src/phrase_tree.py
//

#include <assert.h>
#include <stdio.h>
#include <string>
#include <set>
#include <map>
#include "nt-parser/eval.h"

#ifndef CNN_TREE_H
#define CNN_TREE_H

using namespace std;

typedef map<tuple<string, unsigned, unsigned>, unsigned> BracketCounts;

const set<string> PUNCTUATION = {",", ".", ":", "``", "''", "PU"};

struct Bracket {
  Bracket(int nt_index, unsigned start, unsigned end):
          nt_index(nt_index),
          start(start),
          end(end) {}

  const int nt_index;
  const unsigned start;
  const unsigned end;
};

struct OpenBracket {

  OpenBracket(int nt_index, unsigned start):
          nt_index(nt_index),
          start(start) {}

  Bracket close(unsigned end) const {
      return Bracket(nt_index, start, end);
  }

  const int nt_index;
  const unsigned start;
};

class Tree {
public:

  Tree(const string& symbol, const vector<Tree>& children, shared_ptr<vector<string>> sentence, int leaf_index):
          symbol(symbol), children(children), sentence(sentence), leaf_index(leaf_index) {}

  unsigned left_span() {
      if (_left_span >= 0) return (unsigned) _left_span;

      if (leaf_index >= 0) {
          _left_span = leaf_index;
      } else {
          _left_span = children.front().left_span();
      }
      return (unsigned) _left_span;
  }

  unsigned right_span() {
      if (_right_span >= 0) return (unsigned) _right_span;

      if (leaf_index >= 0) {
          _right_span = leaf_index;
      } else {
          _right_span = children.back().right_span();
      }
      return (unsigned) _right_span;
  }

  BracketCounts brackets(bool advp_prt=true) {
      BracketCounts bracket_counts;
      update_bracket_counts(bracket_counts, advp_prt);
      return bracket_counts;
  }

  void update_bracket_counts(BracketCounts& bracket_counts, bool advp_prt=true) {
      if (leaf_index >= 0) return;

      string nonterm = symbol;
      if (advp_prt and nonterm == "(PRT")
          nonterm = "(ADVP";

      unsigned left = left_span();
      unsigned right = right_span();

      while(left < right_span() && PUNCTUATION.find((*sentence)[left]) != PUNCTUATION.end()) {
          left++;
      }
      while(right > left_span() && PUNCTUATION.find((*sentence)[right]) != PUNCTUATION.end()) {
          right--;
      }

      if (left <= right && nonterm != "(TOP") {
          auto key = tuple<string, unsigned, unsigned>(nonterm, left, right);
          bracket_counts[key]++;
      }
      for (auto child: children) {
          child.update_bracket_counts(bracket_counts, advp_prt);
      }
  }

  MatchCounts compare(Tree& gold, bool advp_prt=true, bool verbose=false) {
      BracketCounts predicted_brackets = brackets(advp_prt);
      BracketCounts gold_brackets = gold.brackets(advp_prt);

      if (verbose) {
          cout << "predicted" << endl;
          for (auto &pair: predicted_brackets) {
              cout << get<0>(pair.first) << "-" << get<1>(pair.first) << "-" << get<2>(pair.first) << " " << pair.second
                   << "\t";
          }
          cout << endl;
          cout << "gold" << endl;
          for (auto &pair: gold_brackets) {
              cout << get<0>(pair.first) << "-" << get<1>(pair.first) << "-" << get<2>(pair.first) << " " << pair.second
                   << "\t";
          }
          cout << endl;
      }

      MatchCounts match_counts;
      for (const auto& pair: gold_brackets) {
          match_counts.gold += pair.second;
          if (predicted_brackets.count(pair.first) > 0)
              match_counts.correct += min(predicted_brackets[pair.first], pair.second);
      }

      for (const auto& pair: predicted_brackets) {
          match_counts.predicted += pair.second;
      }

      return match_counts;
  }

  /*
  void set_sentence(vector<string>& sent) {
      sentence = sent;
      for (auto& child: children) {
          child.set_sentence(sent);
      }
  }
  */

private:
  string symbol;
  vector<Tree> children;
  shared_ptr<vector<string>> sentence;
  int leaf_index = -1;
  int _left_span = -1;
  int _right_span = -1;
};

tuple<Tree, unsigned, unsigned> parse_linearized_helper(const vector<string>& linearized_tree_tokens, shared_ptr<vector<string>> sentence, unsigned start_pos, unsigned leaf_index) {
    unsigned pos = start_pos;
    assert(pos < linearized_tree_tokens.size());
    string symbol = linearized_tree_tokens[pos];
    assert(symbol[0] == '(');

    vector<Tree> children;

    while(true) {
        pos += 1;
        assert(pos < linearized_tree_tokens.size());
        string next_symbol = linearized_tree_tokens[pos];
        if (next_symbol[0] == '(') {
            auto parsed = parse_linearized_helper(linearized_tree_tokens, sentence, pos, leaf_index);
            Tree subtree = get<0>(parsed);
            children.push_back(subtree);
            pos = get<1>(parsed);
            leaf_index = get<2>(parsed);
        } else if (next_symbol[0] == ')') {
            break;
        } else {
            vector<string> child_leaves;
            sentence->push_back(next_symbol);
            children.push_back(Tree(next_symbol, vector<Tree>(), sentence, leaf_index));
            leaf_index++;
        }
    }
    return tuple<Tree, unsigned, unsigned> (Tree(symbol, children, sentence, -1), pos, leaf_index);
};

Tree parse_linearized(const vector<string>& linearized_tree_tokens) {
    shared_ptr<vector<string>> sentence = make_shared<vector<string>>();
    return get<0>(parse_linearized_helper(linearized_tree_tokens, sentence, 0, 0));
}

#endif //CNN_TREE_H
