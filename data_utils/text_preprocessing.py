import re
import numpy as np

def simple_tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    return tokens


def build_vocabulary(texts, min_freq=1):
    freq = {}
    for t in texts:
        tokens = t if isinstance(t, list) else simple_tokenize(t)
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab


def text_to_sequence(text, vocab, max_len=20):
    if isinstance(text, str):
        tokens = simple_tokenize(text)
    else:
        tokens = text

    seq = []
    for token in tokens:
        if token in vocab:
            seq.append(vocab[token])
        else:
            seq.append(vocab["<UNK>"])
    seq = seq[:max_len]
    while len(seq) < max_len:
        seq.append(vocab["<PAD>"])

    return np.array(seq, dtype=np.int32)


def batch_texts_to_sequences(texts, vocab, max_len=20):
    sequences = []
    for txt in texts:
        seq = text_to_sequence(txt, vocab, max_len)
        sequences.append(seq)
    return np.array(sequences, dtype=np.int32)
