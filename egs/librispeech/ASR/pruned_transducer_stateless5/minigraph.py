import k2
import logging


def compile_LG(L_disambig, G, first_token_disambig_id, first_word_disambig_id) -> k2.Fsa:
    L = L_disambig
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG, k2.DeterminizeWeightPushingType.kLogWeightPushing)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    LG.labels[LG.labels >= first_token_disambig_id] = 0
    # See https://github.com/k2-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    return LG


if __name__ == '__main__':
  from pathlib import Path
  import sentencepiece as spm

  from minig import make_G
  from minilexicon_bpe import make_L_disambig

  lang_dir = Path('uk/data/lang_bpe_250')
  sp = spm.SentencePieceProcessor()
  sp.load(str(lang_dir / "bpe.model"))

  words_txt = lang_dir / "words.txt"
  word_sym_table = k2.SymbolTable.from_file(words_txt)

  tokens_txt = lang_dir / "tokens.txt"
  token_sym_table = k2.SymbolTable.from_file(tokens_txt)

  L_disambig = make_L_disambig(sp, word_sym_table)
  G = make_G(["привіт котику"], words_txt)

  first_token_disambig_id = token_sym_table["#0"]
  first_word_disambig_id = word_sym_table["#0"]

  print(compile_LG(L_disambig, G, first_token_disambig_id, first_word_disambig_id))