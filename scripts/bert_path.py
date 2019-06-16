import os

# this should be the directory containing the bert submodule so that "import bert" works, i.e. the root rnng-bert directory
BERT_CODE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
