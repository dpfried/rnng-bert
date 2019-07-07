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


    def tokenize(self, sentence):
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

        for word in cleaned_words:
            word_tokens = self.tokenizer.tokenize(word)
            for _ in range(len(word_tokens)):
                word_end_mask.append(0)
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        tokens.append("[SEP]")
        word_end_mask.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return np.array(input_ids), np.array(word_end_mask)


if __name__ == "__main__":
    bert_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uncased_L-12_H-768_A-12")
    tokenizer = Tokenizer(bert_model_dir)
    input_ids, word_end_mask = tokenizer.tokenize(["This", "is", "a", "test", "antidisestablishmentarianism", "."])
    print(input_ids.tolist())
    print(word_end_mask.tolist())
