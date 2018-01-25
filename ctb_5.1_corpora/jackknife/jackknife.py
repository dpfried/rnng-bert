import subprocess
import os
import os.path
import itertools
import math

tagger_jar = "/home/dfried/lib/stanford-postagger-full-2017-06-09/stanford-postagger.jar"

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

def train_tagger(training_file, output_model_file, training_log_output=None, props_file="train-chinese-nodistsim.tagger.props"):
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

def parse_tagged(tagged_file):
    tagged_sents = []
    words = []
    tags = []
    with open(tagged_file) as f:
        for i, line in enumerate(f):
            if not line.strip():
                assert words and tags
                assert len(words) == len(tags)
                tagged_sents.append(zip(words, tags))
                words = []
                tags = []
                continue
            w, t = line.strip().split('\t')
            words.append(w)
            tags.append(t)
    if words or tags:
        assert len(words) == len(tags)
        tagged_sents.append(zip(words, tags))
    return tagged_sents

def parse_trees(tree_file):
    tagged_sents = []
    with open(tree_file) as f:
        for line in f:
            tags, toks, lc = get_tags_tokens_lowercase(line)
            assert len(tags) == len(toks) == len(lc)
            tagged_sents.append(zip(toks, tags))
    return tagged_sents

def accuracy(pred_tag_file, gold_tree_file):
    predicted_tagged = parse_tagged(pred_tag_file)
    gold_tagged = parse_trees(gold_tree_file)
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

def jackknife(train_corpus, num_splits=10):
    with open(train_corpus) as f:
        lines = list(f)

    for split_index in range(num_splits):
        split_dir = "split_%d" % split_index
        os.mkdir(split_dir)
        split_size = int(math.ceil(float(len(lines)) / num_splits))
        splits = [lines[i:i + split_size] for i in range(0, len(lines), split_size)]
        assert len(splits) == num_splits
        train_splits = [sp for i, sp in enumerate(splits)
                        if i != split_index]
        test_splits = [sp for i, sp in enumerate(splits)
                        if i == split_index]
        assert len(test_splits) == 1
        test_split = test_splits[0]

        train_file = os.path.join(split_dir, "train_%d.gold.stripped" % split_index)
        test_file = os.path.join(split_dir, "test_%d.gold.stripped" % split_index)
        model_file = os.path.join(split_dir, "model_%d.bin" % split_index)
        train_log_file = os.path.join(split_dir, "train_log_%d.txt" % split_index)
        with open(train_file, 'w') as f:
            for line in itertools.chain(*train_splits):
                f.write(line)
        with open(test_file, 'w') as f:
            for line in test_split:
                f.write(line)
        train_tagger(train_file, model_file, train_log_file)


if __name__ == "__main__":
    jackknife('../train.gold.stripped', 10)
