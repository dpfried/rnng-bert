import sys, os
from bert_path import BERT_CODE_PATH
# bert version used: https://github.com/google-research/bert/tree/f39e881b169b9d53bea03d2d341b31707a6c052b
# BERT_CODE_PATH should be the parent folder of the repo, so "import bert" works

sys.path.append(BERT_CODE_PATH)

import bert
import bert.tokenization
import numpy as np

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    }


class Tokenizer(object):
    def __init__(self, bert_model_dir, do_lower_case=None):
        self.model_dir = bert_model_dir
        if do_lower_case is None:
            self.do_lower_case = ("uncased" in bert_model_dir)
        else:
            self.do_lower_case = do_lower_case
        self.tokenizer = bert.tokenization.FullTokenizer(
            os.path.join(bert_model_dir, "vocab.txt"),
            do_lower_case=self.do_lower_case
        )


    def tokenize(self, sentence, return_filtered_sentence=False):
        tokens = []
        word_end_mask = []

        tokens.append("[CLS]")
        word_end_mask.append(1)

        cleaned_words = []
        for word in sentence:
            word = BERT_TOKEN_MAPPING.get(word, word)
            word = word.replace('\\/', '/').replace('\\*', '*')
            word = word.replace('-LSB-', '[').replace('-RSB-', ']').replace('-LRB-', '(').replace('-RRB-', ')') # added for Genia
            if word == "n't" and cleaned_words:
                cleaned_words[-1] = cleaned_words[-1] + "n"
                word = "'t"
            cleaned_words.append(word)

        filtered_sentence = []
        assert len(cleaned_words) == len(sentence)
        for word, original_word in zip(cleaned_words, sentence):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                filtered_sentence.append(original_word)
            for _ in range(len(word_tokens)):
                word_end_mask.append(0)
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        tokens.append("[SEP]")
        word_end_mask.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if return_filtered_sentence:
            return np.array(input_ids), np.array(word_end_mask), filtered_sentence
        else:
            return np.array(input_ids), np.array(word_end_mask)


if __name__ == "__main__":
    # bert_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uncased_L-12_H-768_A-12")
    bert_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bert_models", "uncased_L-12_H-768_A-12")
    tokenizer = Tokenizer(bert_model_dir)
    # sentences = [
    #     ["This", "is", "a", "test", "antidisestablishmentarianism", "."],
    #     "Don ' t say things like that . Please ? ".split(),
    #     "Don't say things like that. Please? ".split(),
    #     "Channels were typically priced between $ 4 and $ 7 , making bundled packages the better deal for all but the most frugal subscribers .".split(),
    #     "Channels were typically priced between $4 and $7, making bundled packages the better deal for all but the most frugal subscribers.".split(),
    #     "But you can't dismiss Mr. Stolzman's music".split(),
    # ]
    # for sentence in sentences:
    #     input_ids, word_end_mask = tokenizer.tokenize(sentence)
    #     print(sentence)
    #     print(input_ids.tolist())
    #     print(word_end_mask.tolist())
    #     with open(os.path.join(bert_model_dir, "vocab.txt")) as f:
    #         piece_by_id = dict(enumerate(line.strip() for line in f))

    #     toks = []
    #     for input_id, word_end_mask in zip(input_ids.tolist(), word_end_mask.tolist()):
    #         toks.append(piece_by_id[input_id])
    #         if word_end_mask:
    #             toks.append(' ')
    #     print(''.join(toks))
    #     print()
    sentences = ["But you can't dismiss Mr. Stolzman's music, which he purchased for $4,500."]
    import nltk
    for sentence in sentences:
        punkt = nltk.tokenize.word_tokenize(sentence)
        input_ids, word_end_mask = tokenizer.tokenize(punkt)
        print(' '.join(punkt))
        tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids)
        print(' '.join(tokens))
        print(' '.join(tokenizer.tokenizer.tokenize(sentence)))
