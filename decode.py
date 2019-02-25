import argparse
import glob
import os
import re
import subprocess
import sys

CORPORA = {
    "ptb": ["dev", "test"],
    "brown": ["cf", "cg", "ck", "cl", "cm", "cn", "cp", "cr"],
    "genia": ["dev", "test"], 
    "ewt": ["answers.dev", "answers.test",
            "email.dev", "email.test",
            "newsgroup.dev", "newsgroup.test",
            "reviews.dev", "reviews.test",
            "weblog.dev", "weblog.test"], 
    "ctb_5.1": ["dev", "test"], 
    "ctb_9.0": ["broadcast_conversations", "broadcast_news", "discussion_forums", "newswire", "weblogs"],
}

LEX_REP = {
    # bert, pos, emb
    (True, False, False): "BERT_bs=32_lr=2e-5_adam_patience=2",
    (False, False, False): "no-emb_no-pos_input-dim=INPUTDIM_bs=32",
    (False, True, False): "no-emb_input-dim=INPUTDIM_bs=32",
    (False, True, True): "emb-nofilt_bs=32",
}

ITERATION_AND_F1 = re.compile(r"best-epoch.*_it-(\d+)-f1-([\d\.]+)_")

def get_iteration_and_f1(model_name):
    match = ITERATION_AND_F1.search(model_name)
    return (int(match.group(1)), float(match.group(2)))

def is_chinese(args):
    return args.corpus.startswith("ctb")

def model_root(args):
    if is_chinese(args):
        return "models_ctb"
    else:
        return "models"

def model_base_name(args):
    lex_rep = LEX_REP[(args.bert, args.pos, args.emb)]
    lex_rep = lex_rep.replace("INPUTDIM", "112" if is_chinese(args) else "132")
    lex_rep = lex_rep.replace("BERT", "bert" if is_chinese(args) else "bert_large")
    return "{}_{}_seed={}".format(
        "inorder" if args.inorder else "topdown",
        lex_rep,
        args.seed
    )

def experiment_name(args):
    base = [args.corpus, args.subcorpus]
    if args.inorder:
        base.append("inorder")
    else:
        base.append("topdown")
    if args.bert:
        base.append("bert")
    else:
        if args.emb:
            base.append("emb")
        if args.pos:
            base.append("pos")
    base.append("seed={}".format(args.seed))
    base.append("beam={}".format(args.beam_size))
    return "-".join(base)

def get_best_model(args):
    glob_path = os.path.join(model_root(args), "{}_best-epoch*_model".format(model_base_name(args)))
    models = glob.glob(glob_path)
    assert models, "no models found for glob {}".format(glob_path)
    return max(models, key=get_iteration_and_f1)

def get_run_string(args):

    assert args.corpus in CORPORA 
    assert args.subcorpus in CORPORA[args.corpus]

    model_dir = get_best_model(args)
    beam_size = args.beam_size
    chinese = is_chinese(args)

    corp_trans_syst = "in_order" if args.inorder else "top_down"


    if chinese:
        train_oracle = "corpora/ctb_5.1/{}/train.{}.oracle".format(
            corp_trans_syst, "gold" if args.bert else "pred"
        )
    else:
        train_oracle = "corpora/english/{}/train.oracle".format(
            corp_trans_syst
        )

    pred_oracle = "corpora/{}/{}/{}.pred.oracle".format(
        args.corpus, corp_trans_syst, args.subcorpus
    )

    bracketing_test_data = "corpora/{}/{}.gold.stripped".format(args.corpus, args.subcorpus)
    eval_files_prefix = "decodes/{}".format(experiment_name(args))

    base = [
        "build/nt-parser/nt-parser",
        "--cnn-seed 1",
        "--cnn-mem 1000,1000,500",
        "--model_dir {}".format(model_dir),
        "-T {}".format(train_oracle),
        "-p {}".format(pred_oracle),
        "--bracketing_test_data {}".format(bracketing_test_data),
        "--lstm_input_dim 128",
        "--hidden_dim 128",
        "--beam_size {}".format(beam_size),
        "--batch_size 8",
        "--eval_files_prefix {}".format(eval_files_prefix),
    ]

    if chinese:
        # which one of these is applied depends on the transition scheme
        base.append("--max_unary 5 --max_cons_nt 15")

    if args.inorder:
        base.append("--inorder")
    if args.bert:
        base.append("--bert --bert_large")
        if chinese:
            base.append("--bert_graph_path bert_models/chinese_L-12_H-768_A-12_graph.pb")
    else:
        if args.pos:
            base.append("-P")
        if args.emb:
            if chinese:
                base.append("--pretrained_dim 80 -w embeddings/zzgiga.sskip.80.vectors")
            else:
                base.append("--pretrained_dim 100 -w embeddings/sskip.100.vectors")

    return ' '.join(base)

def run(args):
    command = get_run_string(args)
    proc = subprocess.Popen(command, stderr=subprocess.PIPE, shell=True)
    for line in proc.stderr:
        dec = line.decode("utf-8")
        if dec.startswith("recall=") or dec.startswith("WARNING"):
            print(dec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", choices=CORPORA.keys(), required=True)
    parser.add_argument("--subcorpus", required=True)

    parser.add_argument("--beam_size", type=int, default=10)

    parser.add_argument("--bert", action='store_true')
    parser.add_argument("--inorder", action='store_true')
    parser.add_argument("--pos", action='store_true')
    parser.add_argument("--emb", action='store_true')

    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--print_only", action='store_true')
    parser.add_argument("--print_best_model", action='store_true')

    args = parser.parse_args()

    if args.print_best_model:
        print(get_best_model(args))
        sys.exit()


    print(get_run_string(args))
    if not args.print_only:
        run(args)
