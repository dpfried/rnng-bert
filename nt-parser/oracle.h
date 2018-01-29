#ifndef PARSER_ORACLE_H_
#define PARSER_ORACLE_H_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

namespace cnn { class Dict; }

namespace parser {

// a sentence can be viewed in 4 different ways:
//   raw tokens, UNKed, lowercased, and POS tags
struct Sentence {
  bool SizesMatch() const { return raw.size() == unk.size() && raw.size() == lc.size() && raw.size() == pos.size(); }
  size_t size() const { return raw.size(); }
  std::vector<int> raw, unk, lc, pos, non_unked_raw;
};

// base class for transition based parse oracles
struct Oracle {
  virtual ~Oracle();
  Oracle(cnn::Dict* dict, cnn::Dict* adict, cnn::Dict* pdict, cnn::Dict* non_unked_dict) : d(dict), ad(adict), pd(pdict), nud(non_unked_dict), sents() {}
  unsigned size() const { return sents.size(); }
  cnn::Dict* d;  // dictionary of terminal symbols
  cnn::Dict* ad; // dictionary of action types
  cnn::Dict* pd; // dictionary of POS tags (preterminal symbols)
  cnn::Dict* nud; // dictionary of non-unked terminal symbols
  std::string devdata;
  std::vector<Sentence> sents;
  std::vector<std::vector<int>> actions;
  std::unordered_map<unsigned, unsigned> raw_term_counts;
 protected:
  static void ReadSentenceView(const std::string& line, cnn::Dict* dict, std::vector<int>* sent);
};

// oracle that predicts nonterminal symbols with a NT(X) action
// the action NT(X) effectively introduces an "(X" on the stack
// # (S (NP ...
// raw tokens
// tokens with OOVs replaced
class TopDownOracle : public Oracle {
 public:
  TopDownOracle(cnn::Dict* termdict, cnn::Dict* adict, cnn::Dict* pdict, cnn::Dict* non_unked_dict, cnn::Dict* nontermdict) :
      Oracle(termdict, adict, pdict, non_unked_dict), nd(nontermdict) {}
  // if is_training is true, then both the "raw" tokens and the mapped tokens
  // will be read, and both will be available. if false, then only the mapped
  // tokens will be available
  void load_bdata(const std::string& file);
  void load_oracle(const std::string& file, bool is_training, bool discard_sentences, bool in_order);
  cnn::Dict* nd; // dictionary of nonterminal types
};

// oracle that predicts nonterminal symbols with a NT(X) action
// the action NT(X) effectively introduces an "(X" on the stack
// # (S (NP ...
// raw tokens
// tokens with OOVs replaced
class TopDownOracleGen : public Oracle {
 public:
  TopDownOracleGen(cnn::Dict* termdict, cnn::Dict* adict, cnn::Dict* pdict, cnn::Dict* non_unked_dict, cnn::Dict* nontermdict) :
      Oracle(termdict, adict, pdict, non_unked_dict), nd(nontermdict) {}
  void load_bdata(const std::string& file);
  void load_oracle(const std::string& file, bool discard_sentences);
  cnn::Dict* nd; // dictionary of nonterminal types
};

class TopDownOracleGen2 : public Oracle {
 public:
  TopDownOracleGen2(cnn::Dict* termdict, cnn::Dict* adict, cnn::Dict* pdict, cnn::Dict* non_unked_dict, cnn::Dict* nontermdict) :
      Oracle(termdict, adict, pdict, non_unked_dict), nd(nontermdict) {}
  void load_oracle(const std::string& file);
  cnn::Dict* nd; // dictionary of nonterminal types
};

} // namespace parser

#endif
