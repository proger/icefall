import k2
from kaldilm import arpa2fst
from icefall.shared.make_kn_lm import NgramCounts
from typing import List, Optional
import tempfile
from pathlib import Path


def make_G(corpus: List[str], words_txt: Optional[Path], order=2):
    ngram_counts = NgramCounts(order)
    for line in corpus:
        ngram_counts.add_raw_counts_from_line(line)
    ngram_counts.cal_discounting_constants()
    ngram_counts.cal_f()
    ngram_counts.cal_bow()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as fp:
        ngram_counts.print_as_arpa(fp)
        fp.close()

        read_symbol_table = str(words_txt) if words_txt else ''
        fst = arpa2fst(fp.name,
                       disambig_symbol='#0',
                       max_order=order,
                       read_symbol_table=read_symbol_table)

    G = k2.Fsa.from_openfst(fst, acceptor=False)
    return G


if __name__ == '__main__':
    lang_dir = Path('uk/data/lang_bpe_250')

    print(make_G(["привіт котику"], read_symbol_table=lang_dir / "words.txt"))