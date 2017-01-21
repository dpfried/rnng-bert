import subprocess
import sys

# Paths relative to root of repository
vocab_path = "corpora/silver_train.dictionary"
oracle_path = "corpora/silver_train.oracle"
sentences_path = "silver.unk.sentences"

print "Counting number of words in silver vocabulary file..."
num_words = 0
with open(vocab_path) as infile:
    for line in infile:
        num_words += 1
num_clusters = int(num_words ** 0.5 + 1)
print "Found {:,} words, using ceil(sqrt({:,})) = {:,} clusters...".format(num_words, num_words, num_clusters)

print "Extracting sentences from silver oracle file..."
with open(oracle_path) as infile, open(sentences_path, "w") as outfile:
    line_index = 0
    for line in infile:
        if not line.strip():
            line_index = 0
        else:
            if line_index == 4:
                outfile.write(line)
            line_index += 1
sys.stdout.write("\n")

command = ["./brown-cluster/wcluster", "--text", sentences_path, "--c", str(num_clusters), "--threads", "30"]
print "Running {}...".format(" ".join(command))
subprocess.call(command)
