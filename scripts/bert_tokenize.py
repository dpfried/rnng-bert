BERT_CODE_PATH = "~/dev/bert"
BERT_MODEL_DIR = "uncased_L-12_H-768_A-12"
BERT_DO_LOWER_CASE = ("uncased" in BERT_MODEL_DIR)
# bert version used: https://github.com/google-research/bert/tree/f39e881b169b9d53bea03d2d341b31707a6c052b
# BERT_CODE_PATH should be the parent folder of the repo, so "import bert" works

import sys, os
sys.path.append(os.path.expanduser(BERT_CODE_PATH))

import bert
import bert.tokenization
import numpy as np

#%%

tokenizer = bert.tokenization.FullTokenizer(
    os.path.join(BERT_MODEL_DIR, "vocab.txt"),
    do_lower_case=BERT_DO_LOWER_CASE
    )

#%%

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

def tokenize(sentence):
    tokens = []
    word_end_mask = []

    tokens.append("[CLS]")
    word_end_mask.append(1)

    cleaned_words = []
    for word in sentence:
        word = BERT_TOKEN_MAPPING.get(word, word).replace('\\/', '/').replace('\\*', '*')
        if word == "n't" and cleaned_words:
            cleaned_words[-1] = cleaned_words[-1] + "n"
            word = "'t"
        cleaned_words.append(word)

    for word in cleaned_words:
        word_tokens = tokenizer.tokenize(word)
        for _ in range(len(word_tokens)):
            word_end_mask.append(0)
        word_end_mask[-1] = 1
        tokens.extend(word_tokens)
    tokens.append("[SEP]")
    word_end_mask.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return np.array(input_ids), np.array(word_end_mask)

# %%

input_ids, word_end_mask = tokenize(["This", "is", "a", "test", "."])
print(input_ids.tolist())
print(word_end_mask.tolist())
