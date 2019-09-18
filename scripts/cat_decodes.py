import glob
import fileinput
import sys
import bert_tokenize

from get_oracle import get_tags_tokens_lowercase_morphfeats, _clean_text, PAREN_NORM, norm_parens

class TokenizingBuffer(object):
    def __init__(self, fname):
        self.f = open(fname, 'r')
        self.open = True
        self.advance()

    def poll(self):
        return self._next
    
    def advance(self):
        print('advancing buffer')
        assert self.open
        try:
            self._next = next(self.f).split()
        except StopIteration:
            self._next = None


    def close(self):
        self.f.close()
        self.open = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenized_sentsplit_file")
    parser.add_argument("tagged_decode_basename")
    parser.add_argument("output_file")
    parser.add_argument("--max_block_num", type=int)
    parser.add_argument("--multi_sentence_symbol", default='MULTI')
    parser.add_argument("--bert_model_dir", required=True)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--long_sentences_file", required=True)
    parser.add_argument("--use_blocks", action='store_true')

    args = parser.parse_args()

    if args.use_blocks:
        gname = args.tagged_decode_basename + "-block-*"
        files = glob.glob(gname)
        assert files, "no files found for glob path {}".format(gname)

        files_and_blocks = [(f, int(f.split('-block-')[-1]))
                            for f in files]
        files_and_blocks = sorted(files_and_blocks, key=lambda p: p[1])

        if args.max_block_num is not None:
            files_and_blocks = [(f, b) for (f, b) in files_and_blocks
                                if b <= args.max_block_num]
        print("decoding blocks: {}".format([b for f, b in files_and_blocks]))
        assert len(files_and_blocks) == files_and_blocks[-1][1] + 1, "must have contiguous blocks, {}".format(map(lambda p: p[1], files_and_blocks))
        files = [f for f, b in files_and_blocks]
        decode_iter = fileinput.input(files=files)
    else:
        decode_iter = fileinput.input(files=[args.tagged_decode_basename])

    bert_tokenizer = bert_tokenize.Tokenizer(args.bert_model_dir)

    if args.resume:
        # read the file and rewrite it,
        with open(args.output_file, 'r') as f:
            resume_line_iter = iter(list(f))
    else:
        resume_line_iter = None

    long_sentences_buffer = TokenizingBuffer(args.long_sentences_file)

    with open(args.tokenized_sentsplit_file) as f_toks, open(args.output_file, 'a' if args.resume else 'w') as f_out:
        it = enumerate(f_toks)
        try:
            import tqdm
            it = tqdm.tqdm(it, ncols=80)
        except:
            pass
        for tok_line_num, tok_line in it:
            skip_line = False
            if resume_line_iter is not None:
                try:
                    # if there's already a line in f_out for this line in f_toks, continue
                    resume_line = next(resume_line_iter)
                    skip_line = True
                except:
                    pass

            # preserve blank lines, which will be missing a decode (since we had the oracle skip them)
            tok_line = _clean_text(tok_line).strip()
            segments = [l.strip() for l in tok_line.split('|||')]
            this_decodes = []

            this_line_tokens = []
            is_long = False

            #for segment in segments:
            #    if segment:
            #        try:
            #            decode_line = next(decode_iter)
            #        except StopIteration as e:
            #            print("ran out of lines in block files; exiting")
            #            sys.exit(1)
            #        this_decodes.append(decode_line.rstrip())
            #if skip_line:
            #    continue

            #if len(segments) == 1 and segments[0]:
            #    tokens = tok_line.split()
            #    #tokens = [PAREN_NORM.get(token, token) for token in tokens]
            #    tokens = [norm_parens(token, lc=False) for token in tokens]
            #    _, _, filtered_tokens = bert_tokenizer.tokenize(tokens, return_filtered_sentence=True)
            #    decode_tokens = get_tags_tokens_lowercase_morphfeats(this_decodes[0], try_to_fix_parens=True)[1]
            #    assert filtered_tokens == decode_tokens, "line {}: {} != {}".format(tok_line_num, filtered_tokens, decode_tokens)

            for segment in segments:
                if not segment:
                    continue

                long_tokens = long_sentences_buffer.poll()

                tokens = segment.split()
                this_line_tokens.extend(tokens)

                if tokens == long_tokens:
                    is_long = True
                    continue

                try:
                    decode_line = next(decode_iter)
                except StopIteration as e:
                    print("ran out of lines in block files; exiting")
                    sys.exit(1)
                this_decodes.append(decode_line.rstrip())

                tokens = [norm_parens(token, lc=False) for token in tokens]
                _, _, filtered_tokens = bert_tokenizer.tokenize(tokens, return_filtered_sentence=True)
                decode_tokens = get_tags_tokens_lowercase_morphfeats(this_decodes[-1], try_to_fix_parens=True)[1]
                assert filtered_tokens == decode_tokens, "line {}: {} != {}".format(tok_line_num, filtered_tokens, decode_tokens)

            if skip_line:
                continue

            if is_long:
                f_out.write("*UNPARSABLE*: {}".format(tok_line))
            elif not this_decodes:
                pass
            elif len(this_decodes) == 1:
                f_out.write(this_decodes[0])
            else:
                f_out.write("({} {})".format(
                    args.multi_sentence_symbol,
                    ' '.join(this_decodes)))
            f_out.write('\n')

    long_sentences_buffer.close()
