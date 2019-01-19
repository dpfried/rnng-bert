import subprocess
import re
import glob
import os.path
from collections import defaultdict

embeddings_by_language = {
    'english': (100, "embeddings/sskip.100.filtered.vectors"),
    'french': (100, "embeddings/french.wiki.sskip.100.filtered.vectors"),
    'chinese': (80, "embeddings/zzgiga.sskip.80.filtered_ctb_5.1.vectors"),
}

corpora_by_language = {
    'english': ("corpora/train", "corpora/dev", "corpora/test"),
    'french': ("french_corpora/train", "french_corpora/dev", "french_corpora/test"),
    'chinese': ("ctb_5.1_corpora/train.pred", "ctb_5.1_corpora/dev.pred", "ctb_5.1_corpora/test.pred"),
}

inorder_corpora_by_language = {
    'english': ("corpora/train_inorder", "corpora/dev_inorder", "corpora/test_inorder"),
    'french': ("french_corpora/train_inorder", "french_corpora/dev_inorder", "french_corpora/test_inorder"),
    'chinese': ("ctb_5.1_corpora/train_inorder.pred", "ctb_5.1_corpora/dev_inorder.pred", "ctb_5.1_corpora/test_inorder.pred"),
}

def run_decode(model, dim, language, corpus, beam_size, inorder):
    lookup = inorder_corpora_by_language if inorder else corpora_by_language
    train, dev, test = lookup[language]
    if corpus == "dev":
        eval_corpus = dev
    else:
        assert corpus == "test"
        eval_corpus = test
    latest_model_in_epoch = re.sub(r"(best-epoch-\d+_it-).*\.in", r"\1*.in", model)
    pretrained_dim, pretrained_file = embeddings_by_language[language]
    command = "source activate.sh; export MKL_NUM_THREADS=4; build/nt-parser/nt-parser \
--cnn-mem 2000,0,500 \
--model {} \
-x \
-T {}.oracle \
-p {}.oracle \
-C {}.stripped \
-P \
--pretrained_dim {} \
--w {} \
--lstm_input_dim {} \
--hidden_dim {} \
--beam_size {} \
-D 0.2".format(
                latest_model_in_epoch, train, eval_corpus, eval_corpus, pretrained_dim, pretrained_file, dim, dim, beam_size
            )
    if inorder:
        command += " --inorder"
    if language == "french":
        command += " --use_morph_features --spmrl"

    print(command)

    proc = subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    for line in stderr.split('\n'):
        if line.startswith("recall="):
            return line
    else:
        return stderr

def models_by_epoch(prefix):
    by_epoch = {}
    for fname in glob.glob(prefix + "*_best-epoch*.bin"):
        match = re.match(r".*best-epoch-([\d\.]+)", fname)
        if not match:
            raise ValueError(fname)
        epoch = int(match.group(1))
        by_epoch[epoch] = fname
    if not by_epoch:
        print("no models found for prefix " + prefix)
    return by_epoch

if __name__ == "__main__":
    import argparse
    from time import gmtime, strftime
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["english", "french", "chinese"], default="english")
    parser.add_argument("--inorder", action='store_true')
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--beam_sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--yoked", action='store_true')
    parser.add_argument("--comparison", action='store_true')
    args = parser.parse_args()
    print(vars(args))

    if args.inorder:
        root_dir = 'in_order'
    else:
        root_dir = '.'

    if args.language == 'english':
        root_dir = os.path.join(root_dir, '.')
    elif args.language == 'french':
        root_dir = os.path.join(root_dir, 'expts_french_morph-embeddings')
    elif args.language == 'chinese':
        root_dir = os.path.join(root_dir, 'expts_ctb_5.1_pred')
    else:
        raise ValueError(args.language)

    models_to_run = defaultdict(list)

    def fastest_below(model_dict, epoch):
        epoch = max(ep for ep in model_dict if ep <= epoch)
        return model_dict[epoch]

    seed = 1
    if args.language == 'english' and not args.inorder and not args.comparison:
        if args.dim == 128:
            seed = 8
        elif args.dim == 256:
            seed = 2
        else:
            raise ValueError("bad dim " + str(args.dim))

    static_models = models_by_epoch(os.path.join(root_dir, "expts_jan-18", "*_%s_lstm_input_dim=%s" % (seed, args.dim)))
    assert static_models

    static_fastest_epoch = max(static_models.keys())
    models_to_run[static_models[static_fastest_epoch]].append("static-fastest")

    reinforce_models = {}
    reinforce_fastest = None
    reinforce_fastest_epoch = 0
    reinforce_slowest = None
    reinforce_slowest_epoch = 1e6
    # TODO: added args.comparison check below becuse of english seed mismatch, take this out
    for candidates in [2, 5, 10] if (args.dim == 128 and not args.inorder and args.comparison) else [10]:
        models = models_by_epoch(
            os.path.join(root_dir, "sequence_level", "%s_method=reinforce_candidates=%s_opt=sgd_include-gold_dim=%s" % (seed, candidates, args.dim))
        )
        assert models
        reinforce_models[candidates] = models

        epoch = max(models.keys())
        if epoch >= reinforce_fastest_epoch:
            reinforce_fastest = candidates
            reinforce_fastest_epoch = epoch
        if epoch <= reinforce_slowest_epoch:
            reinforce_slowest = candidates
            reinforce_slowest_epoch = epoch

    models_to_run[reinforce_models[reinforce_fastest][reinforce_fastest_epoch]].append("reinforce-fastest")

    yoked_epoch = min(static_fastest_epoch, reinforce_fastest_epoch)

    if args.comparison:
        for cands, models in reinforce_models.items():
            models_to_run[fastest_below(models, reinforce_slowest_epoch)].append("reinforce-comparison")

    if args.dim == 128 and not args.inorder:
        dynamic_models = {}
        dynamic_fastest = None
        dynamic_fastest_epoch = 0
        dynamic_slowest = None
        dynamic_slowest_epoch = 1e6
        for candidates in [2, 5, 10]:
            models = models_by_epoch(
                os.path.join(root_dir, "dynamic_oracle", "1_method=sample_dpe=1.0_candidates=%s_include-gold" % candidates)
            )
            assert models
            dynamic_models[candidates] = models

            epoch = max(models.keys())
            if epoch >= dynamic_fastest_epoch:
                dynamic_fastest = candidates
                dynamic_fastest_epoch = epoch
            if epoch <= dynamic_slowest_epoch:
                dynamic_slowest = candidates
                dynamic_slowest_epoch = epoch
        models_to_run[dynamic_models[dynamic_fastest][dynamic_fastest_epoch]].append("dynamic-fastest")
        yoked_epoch = min(yoked_epoch, dynamic_fastest_epoch)
        if args.yoked:
            models_to_run[fastest_below(dynamic_models[dynamic_fastest], yoked_epoch)].append("dynamic-yoked")

        if args.comparison:
            for cands, models in dynamic_models.items():
                models_to_run[fastest_below(models, dynamic_slowest_epoch)].append("dynamic-comparison")

    if args.yoked:
        models_to_run[fastest_below(static_models, yoked_epoch)].append("static-yoked")
        models_to_run[fastest_below(reinforce_models[reinforce_fastest], yoked_epoch)].append("reinforce-yoked")

    for beam_size in args.beam_sizes:
        for corpus in ['dev', 'test']:
            print("*** %s beam_size=%s ***" % (corpus, beam_size))
            for model, tags in sorted(models_to_run.items()):
                print strftime("%Y-%m-%d %H:%M:%S", gmtime())
                print model, ' '.join(tags)
                output = run_decode(model, args.dim, args.language, corpus, beam_size, args.inorder)
                print output
                print
