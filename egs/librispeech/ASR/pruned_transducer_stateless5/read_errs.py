"""
Read errs- files created after decoding datasets
"""

import re

def read_utterances(errors_file):
    inside = 0
    with open(errors_file) as f:
        for line in f:
            line = line.strip()
            if line == 'PER-UTT DETAILS: corr or (ref->hyp)':
                inside = 1
            elif inside and not line:
                break
            elif inside:
                utt_id, text = line.split(':', maxsplit=1)
                yield utt_id, text.strip()


EPS = '*'

def both(toks):
    for tok in toks:
        if tok == '(':
            yield from ref(toks)
        else:
            yield tok, tok

def ref(toks):
    for tok in toks:
        if tok == '->':
            yield from hyp(toks)
            break
        elif tok == '*': continue
        else:
            yield tok, EPS

def hyp(toks):
    for tok in toks:
        if tok == ')': break
        elif tok == '*': continue
        else:
            yield EPS, tok

def tokenize(s, _toks=re.compile(r'(->|\(|\)|\*|\s)')):
    return [t for t in _toks.split(s) if t.strip()]

def read_alignment(s):
    ali = list(both(iter(tokenize(s))))
    sub, cor, ins, del_ = 0, 0, 0, 0
    hypothesis = []
    for ref, hyp in ali:
        if ref == hyp:
            cor += 1
            hypothesis.append(hyp)
        elif ref == '*':
            ins += 1
            hypothesis.append(hyp)
        elif hyp == '*':
            del_ += 1
        else:
            sub += 1
            hypothesis.append(hyp)
    return dict(err=ins+del_+sub, tot=(ins+del_+cor), hyp=' '.join(hypothesis))

if __name__ == '__main__':
    read_alignment("бугай (сотих->кубометрів в) два")