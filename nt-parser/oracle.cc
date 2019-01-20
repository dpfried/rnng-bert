#include "nt-parser/oracle.h"

#include <cassert>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cnn/dict.h"
#include "nt-parser/compressed-fstream.h"

#include <boost/algorithm/string.hpp>

using namespace std;

namespace parser {


Oracle::~Oracle() {}

inline bool is_ws(char x) { //check whether the character is a space or tab delimiter
  return (x == ' ' || x == '\t');
}

inline bool is_not_ws(char x) {
  return (x != ' ' && x != '\t');
}

void Oracle::ReadSentenceView(const std::string& line, cnn::Dict* dict, vector<int>* sent) {
  unsigned cur = 0;
  while(cur < line.size()) {
    while(cur < line.size() && is_ws(line[cur])) { ++cur; }
    unsigned start = cur;
    while(cur < line.size() && is_not_ws(line[cur])) { ++cur; }
    unsigned end = cur;
    if (end > start) {
      unsigned x = dict->Convert(line.substr(start, end - start));
      sent->push_back(x);
    }
  }
  assert(sent->size() > 0); // empty sentences not allowed
}

void TopDownOracle::ReadMorphologyFeatures(const std::string& line, std::vector<std::unordered_map<unsigned, unsigned>>* morphology_feats) {
  std::vector<std::string> features_by_word;
  std::string trimmed_line = boost::trim_copy(line);
  boost::split(features_by_word, trimmed_line, boost::is_space(), boost::token_compress_on);
  for (const string& word_features: features_by_word) {
    std::vector<std::string> features_by_class;
    boost::split(features_by_class, word_features, boost::is_any_of("|"));

    std::unordered_map<unsigned, unsigned> word_feature_map;

    for (std::string& class_feature: features_by_class) {
      //std::vector<std::string> class_and_feature;
      //boost::split(class_and_feature, class_feature, boost::is_any_of("="));
      /*
      if (class_and_feature.size() != 2) {
        if (class_feature != "_") {
          cerr << "line: " << line << endl;
          cerr << "class_feature: " << class_feature << endl;
        }
        assert(class_feature == "_");
        continue;
      }
      std::string _class = class_and_feature.at(0);
      std::string feature = class_and_feature.at(1);
      */
      auto pos = class_feature.find('=');
      if (pos == std::string::npos) {
        continue;
      }
      std::string _class = class_feature.substr(0, pos);
      std::string feature = class_feature.substr(pos+1);

      if (!morphology_dicts->count(_class)) {
        (*morphology_dicts)[_class] = cnn::Dict();
      }
      if (!morphology_singletons->count(_class)) {
        (*morphology_singletons)[_class] = std::vector<bool>();
      }
      int class_index = morphology_classes->Convert(_class);
      auto& dict = (*morphology_dicts)[_class];
      int feature_index = dict.Convert(feature);
      auto& singletons = (*morphology_singletons)[_class];
      if (feature_index >= singletons.size()) {
        singletons.push_back(true);
        assert(singletons.size() == dict.size());
        assert(singletons.size() == feature_index + 1);
      } else {
        singletons[feature_index] = false;
      }
      assert(class_index >= 0);
      assert(feature_index >= 0);
      assert(!word_feature_map.count((unsigned) class_index));
      word_feature_map[(unsigned) class_index] = (unsigned) feature_index;
    }

    morphology_feats->push_back(word_feature_map);
  }
  assert(morphology_feats->size() > 0); // empty sentences not allowed
}


  void Oracle::ReadWordEndMask(const std::string& line, vector<unsigned>& lengths, vector<int32_t>& word_end_mask) {
    unsigned current_length = 0;
    unsigned cur = 0;
    while(cur < line.size()) {
      while(cur < line.size() && is_ws(line[cur])) { ++cur; }
      unsigned start = cur;
      while(cur < line.size() && is_not_ws(line[cur])) { ++cur; }
      unsigned end = cur;
      if (end > start) {
        // TODO(dfried): double-check that this is cross-system compliant
        int32_t mask = std::stoi(line.substr(start, end - start));
        assert(mask == 0 || mask == 1);
        word_end_mask.push_back((unsigned) mask);
        current_length += 1;
        if (mask == 1) {
          lengths.push_back(current_length);
          current_length = 0;
        }
      }
    }
    assert(current_length == 0);
  }

  void Oracle::ReadWordPieceIds(const std::string& line, const std::vector<unsigned>& lengths, std::vector<std::vector<int32_t>>& word_piece_ids, std::vector<int32_t>& word_piece_ids_flat) {
    unsigned cur = 0;
    unsigned current_word_index = 0;
    unsigned current_word_pieces_read = 0;

    while(cur < line.size()) {
      while(cur < line.size() && is_ws(line[cur])) { ++cur; }
      unsigned start = cur;
      while(cur < line.size() && is_not_ws(line[cur])) { ++cur; }
      unsigned end = cur;

      if (end > start) {
        assert(current_word_index < lengths.size());
        if (current_word_pieces_read >= lengths[current_word_index]) {
          current_word_index += 1;
          current_word_pieces_read = 0;
        }
        if (current_word_pieces_read == 0) {
          word_piece_ids.emplace_back();
        }
        // TODO(dfried): double-check that this is cross-system compliant
        int32_t word_piece_id = std::stoi(line.substr(start, end - start));
        word_piece_ids.back().push_back(word_piece_id);
        word_piece_ids_flat.push_back(word_piece_id);
        current_word_pieces_read++;
      }
    }
    assert(lengths.size() == word_piece_ids.size());
    for (unsigned i = 0; i < lengths.size(); i++) {
      assert(word_piece_ids[i].size() == lengths[i]);
    }
  }

void TopDownOracle::load_bdata(const string& file) {
   devdata=file;
}

void TopDownOracle::load_oracle(const string& file, bool is_training, bool discard_sentences, bool in_order, bool read_morphology_features) {
  cerr << "Loading top-down oracle from " << file << " [" << (is_training ? "training" : "non-training") << "] ...\n";
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const string kREDUCE = "REDUCE";
  const string kSHIFT = "SHIFT";
  const string kTERM = "TERM";
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  // only add to the dictionary if we need it
  const int kTERM_INT = in_order ? ad->Convert(kTERM) : std::numeric_limits<int>::max();

  int lc = 0;
  string line;
  vector<int> cur_acts;
  int sent_count = 0;
  Sentence blank_sent;
  while(getline(in, line)) {
    ++lc;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    // modification to deal with the sentence # 200 million ... in the PTB training set
    // https://github.com/danifg/Dynamic-InOrderParser/blob/343f000cdb0d5f7fa09b77aa7e476487d1568ff6/impl/oracle.cc#L60
    if (line.size() == 0 || (line[0] == '!' && line[3] == '(')) continue;
    sent_count++;
    if (sent_count % 1000 == 0) {
      cerr << "\rsent " << sent_count;
    }
    if (discard_sentences)
      blank_sent = Sentence();
    else
      sents.resize(sents.size() + 1);
    auto& cur_sent = discard_sentences ? blank_sent : sents.back();
    if (is_training) {  // at training time, we load both "UNKified" versions of the data, and raw versions
      ReadSentenceView(line, pd, &cur_sent.pos);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.raw);
      ReadSentenceView(line, nud, &cur_sent.non_unked_raw);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.lc);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.unk);

    } else { // at test time, we ignore the raw strings and just use the "UNKified" versions
      ReadSentenceView(line, pd, &cur_sent.pos);
      getline(in, line);
      ReadSentenceView(line, nud, &cur_sent.non_unked_raw);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.lc);
      getline(in, line);
      ReadSentenceView(line, d, &cur_sent.unk);
      cur_sent.raw = cur_sent.unk;
    }
    // read morph features?
    getline(in, line);
    if (read_morphology_features) {
      ReadMorphologyFeatures(line, &cur_sent.morphology_features);
      assert(cur_sent.morphology_features.size() == cur_sent.raw.size());
    }

    // read BERT word masks and ids
    getline(in, line);
    vector<unsigned> word_lengths_in_pieces;
    ReadWordEndMask(line, word_lengths_in_pieces, cur_sent.word_end_mask);
    getline(in, line);
    ReadWordPieceIds(line, word_lengths_in_pieces, cur_sent.word_piece_ids, cur_sent.word_piece_ids_flat);
    assert(cur_sent.word_end_mask.size() == cur_sent.word_piece_ids_flat.size());

    for (auto word : cur_sent.raw) raw_term_counts[word]++;
    // re: commit changing line below from 3->6, it should have been 4 previously (but it's ok b/c it's only used for error reporting)
    lc += 6;
    if (!cur_sent.SizesMatch()) {
      cerr << "Mismatched lengths of input strings in oracle before line " << lc << endl;
      abort();
    }
    int termc = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      assert(line.find(' ') == string::npos);
      if (line == kREDUCE) {
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        // Convert NT
        nd->Convert(line.substr(3, line.size() - 4));
        // NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(ad->Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
      } else if (line == kTERM) {
        assert(in_order);
        cur_acts.push_back(kTERM_INT);
      } else {
        cerr << "Malformed input in line " << lc << endl;
        abort();
      }
    }
    if (!discard_sentences)
      actions.push_back(cur_acts);
    if (termc != cur_sent.size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      cerr << "num tokens: " << cur_sent.size() << endl;
      cerr << "num shifts: " << termc << endl;
      abort();
    }
  }
  cerr << endl;
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}

void TopDownOracleGen::load_bdata(const string& file) {
  devdata=file;
}

void TopDownOracleGen::load_oracle(const string& file, bool discard_sentences) {
  cerr << "Loading top-down generative oracle from " << file << endl;
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const string kREDUCE = "REDUCE";
  const string kSHIFT = "SHIFT";
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  int lc = 0;
  string line;
  vector<int> cur_acts;
  Sentence blank_sent;
  while(getline(in, line)) {
    ++lc;
    //cerr << "line number = " << lc << endl;
    cur_acts.clear();
    if (line.size() == 0 || line[0] == '#') continue;
    if (discard_sentences)
      blank_sent = Sentence();
    else
      sents.resize(sents.size() + 1);
    auto& cur_sent = discard_sentences ? blank_sent : sents.back();
    ReadSentenceView(line, nud, &cur_sent.non_unked_raw);
    getline(in, line);
    ReadSentenceView(line, d, &cur_sent.raw);
    //getline(in, line);
    //ReadSentenceView(line, d, &cur_sent.unk);
    cur_sent.pos = cur_sent.unk = cur_sent.lc = cur_sent.raw;
    lc += 1;
    if (!cur_sent.SizesMatch()) {
      cerr << "Mismatched lengths of input strings in oracle before line " << lc << endl;
      cerr << "raw: " << cur_sent.raw.size() << endl;
      cerr << "unk: " << cur_sent.unk.size() << endl;
      abort();
    }
    int termc = 0;
    while(getline(in, line)) {
      ++lc;
      //cerr << "line number = " << lc << endl;
      if (line.size() == 0) break;
      assert(line.find(' ') == string::npos);
      if (line == kREDUCE) {
        cur_acts.push_back(kREDUCE_INT);
      } else if (line.find("NT(") == 0) {
        // Convert NT
        nd->Convert(line.substr(3, line.size() - 4));
        // NT(X) is put into the actions list as NT(X)
        cur_acts.push_back(ad->Convert(line));
      } else if (line == kSHIFT) {
        cur_acts.push_back(kSHIFT_INT);
        termc++;
      } else {
        cerr << "Malformed input in line " << lc << endl;
        abort();
      }
    }
    if (!discard_sentences)
      actions.push_back(cur_acts);
    if (termc != cur_sent.size()) {
      cerr << "Mismatched number of tokens and SHIFTs in oracle before line " << lc << endl;
      cerr << "num tokens: " << cur_sent.size() << endl;
      cerr << "num shifts: " << termc << endl;
      abort();
    }
  }
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}

void TopDownOracleGen2::load_oracle(const string& file) {
  cerr << "Loading top-down generative oracle from " << file << endl;
  cnn::compressed_ifstream in(file.c_str());
  assert(in);
  const int kREDUCE_INT = ad->Convert("REDUCE");
  const int kSHIFT_INT = ad->Convert("SHIFT");
  int lc = 0;
  string line;
  vector<int> cur_acts;
  //cerr << "Checkpoint 1" << endl;
  while(getline(in, line)) {
    cur_acts.clear();
    sents.push_back(Sentence());
    auto& raw = sents.back().raw;
    ++lc;
    unsigned len = line.size();
    unsigned i = 0;
    while(i < len) {
      while(i < len && line[i] == ' ') { i++; }
      if (i == len) break;
      if (line[i] == '(') { // NT
        unsigned start = i + 1;
        unsigned end = start;
        while(end < len && line[end] != ' ') { end++; }
        assert(end > start);
        //cerr << "Checkpoint 1.5" << endl;
        int ntidx = nd->Convert(line.substr(start, end - start));
        //cerr << "Checkpoint 2" << endl;
        string act = "NT(" + nd->Convert(ntidx) + ')';
        //cerr << "Checkpoint 3" << endl;
        cur_acts.push_back(ad->Convert(act));
        //cerr << "Checkpoint 4" << endl;
        i = end;
        if (i >= len || line[i] == ')') { cerr << "Malformed input: " << line << endl; abort(); }
        continue;
      } else if (line[i] == ')') {
        unsigned start = i;
        unsigned end = i + 1;
        while(end < len && line[end] == ')') { end++; }
        for (unsigned j = start; j < end; ++j)
          cur_acts.push_back(kREDUCE_INT);
        i = end;
        continue;
      }
      // terminal symbol ...
      unsigned start = i;
      unsigned end = start + 1;
      while(end < len && line[end] != ' ' && line[end] != ')') { end++; }
      if (end >= len) { cerr << "Malformed input: " << line << endl; abort(); }
      int term = d->Convert(line.substr(start, end - start));
      raw.push_back(term);
      cur_acts.push_back(kSHIFT_INT);
      i = end;
    }
    sents.back().pos = sents.back().lc = sents.back().unk = sents.back().raw;
    actions.push_back(cur_acts);
  }
  //cerr << "Checkpoint 5" << endl;
  cerr << "Loaded " << sents.size() << " sentences\n";
  cerr << "    cumulative      action vocab size: " << ad->size() << endl;
  cerr << "    cumulative    terminal vocab size: " << d->size() << endl;
  cerr << "    cumulative nonterminal vocab size: " << nd->size() << endl;
  cerr << "    cumulative         pos vocab size: " << pd->size() << endl;
}

} // namespace parser
