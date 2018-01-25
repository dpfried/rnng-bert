import subprocess
import os
import os.path
import itertools
import math
import bllipparser

tagger_jar = "/home/dfried/lib/stanford-postagger-full-2017-06-09/stanford-postagger.jar"

def replace_tags(tree_string, replacement_tag_seq):
    tree = bllipparser.Tree("(S1 " + tree_string + ")")
    position = 0
    for node in tree.all_subtrees():
        if node.is_preterminal():
            node.label = replacement_tag_seq[position]
            position += 1
    assert position == len(replacement_tag_seq)
    tree_str = str(tree)
    assert tree_str.startswith("(S1 ") and tree_str.endswith(")")
    return tree_str[4:-1]

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]

def train_tagger(props_file, training_file, output_model_file, training_log_output=None):
    command = [
        "java",
        "-cp", tagger_jar,
        "-Xmx1G",
        "edu.stanford.nlp.tagger.maxent.MaxentTagger",
        "-props", props_file,
        "-model", output_model_file,
        "-trainFile", "format=TREES,%s" % training_file
    ]
    print(' '.join(command))
    if training_log_output:
        with open(training_log_output, 'w') as f:
            subprocess.call(command, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.call(command)

def run_tagger(test_file, model_file, output_file, stderr_file=None):
    command = [
        "java",
        "-cp", tagger_jar,
        "-Xmx200M",
        "edu.stanford.nlp.tagger.maxent.MaxentTagger",
        "-model", model_file,
        "-textFile", "format=TREES,%s" % test_file,
        "-outputFile", output_file,
        "-outputFormat", "tsv",
    ]
    print(' '.join(command))
    if stderr_file is not None:
        with open(stderr_file, 'w') as ferr:
            subprocess.call(command, stderr=ferr)
    else:
        subprocess.call(command)

def read_tagged(tagged_file):
    tagged_sents = []
    words = []
    tags = []
    with open(tagged_file) as f:
        for line in f:
            if not line.strip():
                assert words and tags
                assert len(words) == len(tags)
                tagged_sents.append(list(zip(words, tags)))
                words = []
                tags = []
                continue
            w, t = line.strip().split('\t')
            words.append(w)
            tags.append(t)
    if words or tags:
        assert len(words) == len(tags)
        tagged_sents.append(list(zip(words, tags)))
    return tagged_sents

def read(fname, strip=True):
    with open(fname) as f:
        return list([line.rstrip() if strip else line 
                     for line in f])

def tree_string_to_tagged(tree_string):
    tags, toks, lc = get_tags_tokens_lowercase(tree_string)
    assert len(tags) == len(toks) == len(lc)
    return list(zip(toks, tags))

def accuracy(predicted_tagged, gold_tagged):
    assert len(predicted_tagged) == len(gold_tagged)
    count = 0
    correct = 0
    for pred, gold in zip(predicted_tagged, gold_tagged):
        pred_w, pred_t = zip(*pred)
        gold_w, gold_t = zip(*gold)
        assert pred_w == gold_w
        for p, g in zip(pred_t, gold_t):
            count += 1
            if p == g:
                correct += 1
    return correct, count

def run_partition(props_file, train_file, test_files, model_file, train_log_file, predicted_tags_files, predicted_tree_files, tag_log_files=None, names=None, train_models=False):
    if train_models:
        with open(train_file, 'w') as f:
            for line in itertools.chain(*train_splits):
                f.write(line)
        with open(test_file, 'w') as f:
            for line in test_split:
                f.write(line)
        train_tagger(props_file, train_file, model_file, train_log_file)

    assert len(test_files) == len(predicted_tags_files) == len(predicted_tree_files)
    if names is None:
        names = [None for _ in test_files]
    if tag_log_files is None:
        tag_log_files = [None for _ in test_files]
    for name, test_file, predicted_tree_file, predicted_tags_file, tag_log_file in zip(names, test_files, predicted_tree_files, predicted_tags_files, tag_log_files):
        run_tagger(test_file, model_file, predicted_tags_file, stderr_file=tag_log_file)

        all_predicted_tagged = read_tagged(predicted_tags_file)
        all_gold_trees = read(test_file)
        all_gold_tagged = [
            tree_string_to_tagged(tree) for tree in all_gold_trees
        ]

        correct, total = accuracy(all_predicted_tagged, all_gold_tagged)
        if name:
            print("accuracy for %s:\t %d / %d (%0.2f%%)" % (name, correct, total, 100.0 * correct / total))

        pred_trees = []
        assert len(all_gold_trees) == len(all_gold_tagged) == len(all_predicted_tagged)
        for gold_tree, gold_tagged, pred_tagged in zip(all_gold_trees, all_gold_tagged, all_predicted_tagged):
            pred_words, pred_tags = zip(*pred_tagged)
            gold_words, gold_tags = zip(*gold_tagged)
            assert pred_words == gold_words
            pred_tree = replace_tags(gold_tree, pred_tags)
            pred_trees.append(pred_tree)

        with open(predicted_tree_file, 'w') as f:
            for tree in pred_trees:
                f.write(tree + "\n")

def jackknife(props_file, train_gold_fname, output_train_pred_fname, num_splits=10, train_models=False):
    with open(train_gold_fname) as f:
        gold_trees = list(f)

    split_size = int(math.ceil(float(len(gold_trees)) / num_splits))
    splits = [gold_trees[i:i + split_size] for i in range(0, len(gold_trees), split_size)]
    assert len(splits) == num_splits

    all_pred_trees = []

    for split_index in range(num_splits):
        split_dir = "split_%d" % split_index
        try:
            os.mkdir(split_dir)
        except:
            pass
        train_splits = [sp for i, sp in enumerate(splits)
                        if i != split_index]
        test_splits = [sp for i, sp in enumerate(splits)
                        if i == split_index]
        assert len(test_splits) == 1
        test_split = test_splits[0]

        train_file = os.path.join(split_dir, "train_%d.gold.stripped" % split_index)
        test_file = os.path.join(split_dir, "test_%d.gold.stripped" % split_index)
        model_file = os.path.join(split_dir, "model_%d.bin" % split_index)
        train_log_file = os.path.join(split_dir, "train_%d.log" % split_index)
        tag_log_file = os.path.join(split_dir, "tag_%d.log" % split_index)
        predicted_tags_file = os.path.join(split_dir, "test_%d.pred.tags" % split_index)
        predicted_tree_file = os.path.join(split_dir, "test_%d.pred.stripped" % split_index)

        run_partition(props_file, train_file, [test_file], model_file, train_log_file, [predicted_tags_file], [predicted_tree_file], [tag_log_file], [split_dir], train_models=train_models)

        pred_trees = read(predicted_tree_file)
        all_pred_trees.extend(pred_trees)

    assert len(all_pred_trees) == len(gold_trees)
    with open(output_train_pred_fname, 'w') as f:
        for pred_tree in all_pred_trees:
            pred_tree = pred_tree.rstrip()
            f.write(pred_tree + "\n")

if __name__ == "__main__":
    num_splits = 10
    train_models=False

    train_gold_file = '../train.gold.stripped'
    dev_gold_file = '../dev.gold.stripped'
    test_gold_file = '../test.gold.stripped'

    train_pred_file = '../train.pred.stripped'
    dev_pred_file = '../dev.pred.stripped'
    test_pred_file = '../test.pred.stripped'

    props_file="train-chinese-nodistsim.tagger.props"

    jackknife(props_file, train_gold_file, train_pred_file, num_splits)
    run_partition(props_file, 
                  train_gold_file,
                  [dev_gold_file, test_gold_file],
                  'model_full.bin',
                  'train_log_full.txt',
                  ['dev.pred.tags', 'test.pred.tags'],
                  [dev_pred_file, test_pred_file],
                  ['tag_dev.log', 'tag_test.log'],
                  ['dev', 'test'],
                  train_models=train_models)
