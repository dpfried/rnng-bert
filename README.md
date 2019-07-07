# Shift-Reduce Constituency Parsing with Contextual Representations

This repository contains an implementation of the [Recurrent Neural Network Grammars (Dyer et al. 2016)](https://arxiv.org/abs/1602.07776) and [In-Order (Liu and Zhang 2017)](https://aclweb.org/anthology/Q17-1029) constituency parsers, both integrated with a [BERT (Devlin et al. 2019)](https://aclweb.org/anthology/papers/N/N19/N19-1423/) sentence encoder. 

Our best current models for the In-Order system with BERT obtain 96.0 F1 on the English PTB test set and 92.0 F1 on the Chinese Treebank v5.1 test set. More results (including out-of-domain transfer) are described in [Cross-Domain Generalization of Neural Constituency Parsers (Fried*, Kitaev*, and Klein, 2019)](TODO).

Modifications to the RNNG and In-Order parsers implemented here include:

- BERT integration for the discriminative models
- Beam search decoding for the discriminative and generative models
- Minimum-risk training using policy gradient for the discriminative models 
- Dynamic oracle training for the RNNG discriminative model

This repo contains a compilation of code from many people and multiple research projects; please see [Credits](#credits) and [Citations](#citations) below for details.

Note: for most practical parsing purposes, we'd recommend using the [BERT-equipped Chart parser](https://github.com/nikitakit/self-attentive-parser) of [Kitaev, Cao, and Klein, 2019](https://arxiv.org/abs/1812.11760), which is easier to setup, faster, has a smaller model size, and achieves performance nearly as strong as this parser.

## Contents
1. [Available Models](#available-models)
2. [Prerequisites](#prerequisites)
3. [Build Instructions](#build-instructions)
4. [Usage](#usage)
5. [Training](#training)
6. [Citations](#citations)
7. [Credits](#credits)

## Available Models

| Model | Language | Info |
| ----- | -------- | ---- |
| [english](https://berkeleynlp.s3.amazonaws.com/inorder-bert-models/english.tgz)| English | 95.65 F1 / 57.28 EM on the PTB test set (with beam size 10). 1.2GB. This is the model that is the best-scoring on the development set out of the five runs of In-Order+BERT English models described in our [ACL 2019 paper](TODO).|
| [english-wwm](https://berkeleynlp.s3.amazonaws.com/inorder-bert-models/english-wwm.tgz) | English | 95.99 F1 / 57.99 EM on the PTB test set (with beam size 10). 1.2GB. This model is identical to `english` above, but uses a BERT model pre-trained with [whole-word masking](https://github.com/google-research/bert/commit/0fce551b55caabcfba52c61e18f34b541aef186a).
| [chinese](https://berkeleynlp.s3.amazonaws.com/inorder-bert-models/chinese.tgz) | Chinese | 91.96 F1 / 44.54 EM on the CTB v5.1 test set (with beam size 10). 370MB. This is the model that is best-scoring on the development set out of the five runs of In-Order+BERT Chinese models.

## Prerequisites
 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries, version 1.58.0 or later (earlier versions may work for training parsers, but may be unable to load our serialized models)
 * [Eigen](http://eigen.tuxfamily.org) (latest development release)
 * [CMake](http://www.cmake.org/)
 * Python 3
 * [TensorFlow for Python](https://www.tensorflow.org/install). We've tested against version 1.12.0, but newer versions will likely work as well.
 * [TensorFlow C API](https://www.tensorflow.org/install/lang_c). We've tested against version 1.12.0, but newer versions may work as well. You can download the precompiled C API for common architectures at [https://www.tensorflow.org/install/lang_c](https://www.tensorflow.org/install/lang_c). To obtain version 1.12.0 that we used, change the version string in the download URLs on that page to 1.12.0 (e.g. [https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz) for Linux with GPU support). Download this and extract the .tar.gz file, e.g. to `$HOME/lib/libtensorflow-gpu-linux-x86_64-1.12.0`. This directory should now contain a `lib` and `include` directory.

Optional:
* [MKL](https://software.intel.com/en-us/mkl) allows faster processing for the non-BERT CPU operations

We use a submodule for the BERT code. To get this when cloning our repository:
```
git clone --recursive https://github.com/dpfried/rnng-bert.git
```

If you didn't clone with `--recursive`, you'll need to manually get the `bert` submodule. Run the following inside the top-level `rnng-bert` directory:
```
git submodule update --init --recursive
```

## Build Instructions

Assuming the latest development version of Eigen is stored at: `/opt/tools/eigen-dev`, and you've extracted or built the TensorFlow C files (see prerequisites above) at `$HOME/lib/libtensorflow-gpu-linux-x86_64-1.12.0`: 

```
mkdir build
cd build
cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev -DTENSORFLOW_ROOT=$HOME/lib/libtensorflow-gpu-linux-x86_64-1.12.0 -DCMAKE_BUILD_TYPE=Release ..
make -j2
```

If your BOOST installation is in a non-standard location, also specify -DBOOST_ROOT=/path/to/boost

Optional: to compile with MKL, assuming MKL is stored at `/opt/intel/mkl`, instead run:

```
mkdir build
cd build
cmake -DEIGEN3_INCLUDE_DIR=/opt/tools/eigen-dev -DTENSORFLOW_ROOT=$HOME/lib/libtensorflow-gpu-linux-x86_64-1.12.0 -DMKL=TRUE -DMKL_ROOT=/opt/intel/mkl -DCMAKE_BUILD_TYPE=Release ..
make -j2
```

Optional: If training the parser, you'll also need the evalb executable. Build it by running `make` inside the EVALB directory.

## Usage

First, download and extract one of the [models](#available-models). For the rest of this section, we'll assume that you've downloaded and extracted `english-wwm` into the `bert_models` folder.

### Parsing Raw Text
Input should be a file with one sentence per line, consisting of space-separated tokens. For best performance, you should use tokenization in the style of the Penn Treebank.

For English, you can tokenize sentences using a tokenizer such as [nltk.word_tokenize](https://www.nltk.org/api/nltk.tokenize.html). Here is an example tokenized sentence (taken from the Penn Treebank):

`No , it was n't Black Monday .`

(note that "wasn't" is split into "was" and "n't").

For Chinese, use a tokenizer such as [jieba](https://github.com/fxsjy/jieba) or [unofficial tokenizers for SpaCy](https://github.com/howl-anderson/Chinese_models_for_SpaCy). Here is an example tokenized sentence (from the Penn Chinese Treebank and using its tokenization; automatic tokenizers may return different tokeizations):

`“ 中 美 合作 高 科技 项目 签字 仪式 ” 今天 在 上海 举行 。`

Once the token input file is constructed, run `python3 scripts/bert_parse.py $model_dir $token_file --beam_size 10` to parse the token file and print parse trees to standard out. For example,

`python3 scripts/bert_parse.py bert_models/english-wwm tokens.txt --beam_size 10`

Note: These parsers are not currently designed to predict part-of-speech (POS) tags, and will output trees that use XX for all POS tags.

### Comparing Against a Treebank

Given a treebank file in `$treebank_file` with one tree per line (for example, as produced by [our PTB data generation](https://github.com/nikitakit/parser-data-gen) code), you can parse the tokens in these sentences and compute parse evaluation scores using the following:

```
python3 scripts/dump_tokens.py $treebank_file > treebank.tokens
python3 scripts/bert_parse.py bert_models/english-wwm treebank.tokens --beam_size 10 > treebank.parsed
python3 scripts/retag.py $treebank_file treebank.parsed > treebank.parsed.retagged
EVALB/evalb -p EVALB/COLLINS_ch.prm $treebank_file treebank.parsed.retagged
```
(`COLLINS_ch.prm` is a parameter file that can be used to evaluate on either the English or Chinese Penn Treebanks; it is modified from COLLINS.prm to drop the PU punctuation tag which is found in the CTB corpora.)

## Training

Instructions should (hopefully) be coming soon. Please contact `dfried AT cs DOT berkeley DOT edu` if you'd like help training the models that use BERT in the meantime. The oracle generation scripts we used are in `corpora/*/build_corpus.sh` and training scripts are in `train_*.sh`, but there are currently some missing dependencies and hard-coded paths. 

# Citations

This repo contains code from a number of papers.

For the RNNG or In-Order models, please cite the original papers:

```
@inproceedings{dyer-rnng:16,
  author = {Chris Dyer and Adhiguna Kuncoro and Miguel Ballesteros and Noah A. Smith},
  title = {Recurrent Neural Network Grammars},
  booktitle = {Proc. of NAACL},
  year = {2016},
} 

@article{TACL1199,
  author = {Liu, Jiangming and Zhang, Yue },
  title = {In-Order Transition-based Constituent Parsing},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {5},
  year = {2017},
  issn = {2307-387X},
  pages = {413--424}          
}
```

For beam search in the generative model:

```
@InProceedings{Stern-Fried-Klein:2017:GenerativeParserInference,
  title     = {Effective Inference for Generative Neural Parsing},
  author    = {Mitchell Stern and Daniel Fried and Dan Klein},
  booktitle = {Proceedings of EMNLP},
  month     = {September},
  year      = {2017},
}
```

For policy gradient or dynamic oracle training:

```
@InProceedings{Fried-Klein:2018:PolicyGradientParsing,
  title     = {Policy Gradient as a Proxy for Dynamic Oracles in Constituency Parsing},
  author    = {Daniel Fried and Dan Klein},
  booktitle = {Proceedings of ACL},
  month     = {July},
  year      = {2018},
}
```

For the BERT integration:

```
@inproceedings{devlin-etal-2019-bert,
  title = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author = {Devlin, Jacob  and
    Chang, Ming-Wei  and
    Lee, Kenton  and
    Toutanova, Kristina},
  booktitle = {Proceedings of NAACL},
  month = {June},
  year = {2019},
}

@InProceedings{Fried-Kitaev-Klein:2019:ParserGeneralization,
  title     = {Cross-Domain Generalization of Neural Constituency Parsers},
  author    = {Daniel Fried, Nikita Kitaev, and Dan Klein},
  booktitle = {Proceedings of ACL},
  month     = {July},
  year      = {2019},
}
```

## Credits

The code in this repo (and parts of this readme) is derived from the [RNNG parser](https://github.com/clab/rnng) by Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah Smith, incorporating the [In-Order transition system](https://github.com/LeonCrashCode/InOrderParser) of Jiangming Liu and Yue Zhang. Additional modifications (beam search, abstraction of the parser state and ensembling, BERT integration, the RNNG dynamic oracle, and min-risk policy gradient training) were made by Daniel Fried, Mitchell Stern, and Nikita Kitaev.
