import string
from collections import defaultdict
from utils import get_word_tag
from utils import assign_unk

def build_vocab(corpus_path):
    with open(corpus_path, 'r') as f:
        lines = f.readlines()

    tokens = [line.split('\t')[0] for line in lines]
    freqs = defaultdict(int)
    for tok in tokens:
        freqs[tok] += 1

    vocab = [k for k, v in freqs.items() if (k != '\n')]
    unk_toks = ["--unk--", "--unk_adj--", "--unk_adv--", "--unk_digit--", "--unk_noun--", "--unk_punct--", "--unk_upper--", "--unk_verb--"]
    vocab.extend(unk_toks)
    vocab.append("--n--")
    vocab.append(" ")
    vocab = sorted(set(vocab))
    return vocab
