import re
import collections
import tensorflow as tf
import numpy as np


def count_corpus(tokens):
    # flat tokens arrays.
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return [re.sub(pattern='[^A-Za-z]+', repl=' ', string=line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def load_corpus_time_machine(max_tokens=-1):
    lines = read_file('txt/time_machine.txt')
    print(f'# 文本总行数: {len(lines)}\n')

    tokens = tokenize(lines, token='char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    # tuple-2 tokens
    # bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    # vocab = Vocab(bigram_tokens)
    # print(vocab.token_freqs[:10])

    return corpus, vocab


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = self.seq_data_iter_random
        else:
            self.data_iter_fn = self.seq_data_iter_sequential
        self.corpus, selfcab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn()

    def seq_data_iter_random(self):
        self.corpus = self.corpus[np.random.randint(0, self.num_steps - 1):]
        num_subseqs = (len(self.corpus) - 1) // self.num_steps
        initial_indices = list(range(0, num_subseqs * self.num_steps, self.num_steps))
        np.random.shuffle(initial_indices)

        def get_subseq(pos):
            return self.corpus[pos: pos + self.num_steps]

        num_batches = num_subseqs // self.batch_size
        for i in range(0, self.batch_size * num_batches, self.batch_size):
            initial_indices_per_batch = initial_indices[i: i + self.batch_size]
            x = [get_subseq(j) for j in initial_indices_per_batch]
            y = [get_subseq(j + 1) for j in initial_indices_per_batch]
            yield tf.constant(x), tf.constant(y)

    def seq_data_iter_sequential(self):
        offset = np.random.randint(0, self.num_steps)
        num_tokens = ((len(self.corpus) - offset - 1) // self.batch_size) * self.batch_size
        Xs = tf.constant(self.corpus[offset: offset + num_tokens])
        Ys = tf.constant(self.corpus[offset + 1: offset + 1 + num_tokens])
        Xs = tf.reshape(Xs, (self.batch_size, -1))
        Ys = tf.reshape(Ys, (self.batch_size, -1))
        num_batches = Xs.shape[1] // self.num_steps
        for i in range(0, num_batches * self.num_steps, self.num_steps):
            X = Xs[:, i: i + self.num_steps]
            Y = Ys[:, i: i + self.num_steps]
            yield X, Y


def main():
    data_iter = SeqDataLoader(batch_size=2, num_steps=5, use_random_iter=False, max_tokens=1000)
    print(f'vocab size {len(data_iter.vocab)}, corpus size {len(data_iter.corpus)}')
    #for x, y in data_iter:
    #    print(f'x: {x}, y: {y}')



if __name__ == "__main__":
    main()
