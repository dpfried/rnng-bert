import subprocess

from collections import namedtuple

ParseMetrics = namedtuple("ParseMetrics", ["recall", "precision", "f1", "complete_match"])

def eval_b(ref_file, out_file):
    command = ["EVALB/evalb", "-p", "EVALB/COLLINS.prm", ref_file, out_file]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    recall = None
    precision = None
    f1 = None
    complete_match = None
    for line in proc.stdout:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if line.startswith('Bracketing Recall'):
            assert(recall is None)
            recall = float(line.strip().split('=')[1].strip())
        if line.startswith('Bracketing Precision'):
            assert(precision is None)
            precision = float(line.strip().split('=')[1].strip())
        if line.startswith('Bracketing FMeasure'):
            assert(f1 is None)
            f1 = float(line.strip().split('=')[1].strip())
        if line.startswith('Complete match'):
            assert(complete_match is None)
            complete_match = float(line.strip().split('=')[1].strip())
            # return here before reading <= 40
            return ParseMetrics(recall, precision, f1, complete_match)
