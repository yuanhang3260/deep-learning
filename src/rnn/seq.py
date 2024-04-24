import matplotlib.pyplot as plt
import tensorflow as tf

import rnn.text
import datasets as ds

lang_file = {
    'chz': 'txt/chz_eng.txt',
    'fra': 'txt/fra_eng_min.txt'
}

lang = 'fra'

def read_file(filename, num_examples=1):
    with open(filename, 'r') as f:
        lines = f.readlines()
    #return lines[::(len(lines) // num_examples)]
    return lines[:num_examples]


def tokenize_nmt(lines):
    source, target = [], []
    for line in lines:
        line = line.strip()
        parts = line.split('\t')
        if len(parts) >= 2:
            source.append(parts[0])
            target.append(parts[1])

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    def process_seq(text):
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text)]
        return ''.join(out)

    for i in range(len(source)):
        source[i] = process_seq(source[i]).split(' ')
        if lang == 'chz':
            target[i] = list(process_seq(target[i]))  # split chinese characters
        else:
            target[i] = process_seq(target[i]).split(' ')

    return source, target


def plot_token_hist(x_label, y_label_list, y_data_list):
    fig, ax = plt.subplots()

    ax.set_xlabel(x_label)
    ax.hist(y_data_list, label=y_label_list, bins=10)

    ax.legend()

    plt.grid()
    plt.show()


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab['<eos>']] for line in lines]
    array = tf.constant([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_len = tf.reduce_sum(tf.cast(array != vocab['<pad>'], tf.int32), axis=1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    source, target = tokenize_nmt(read_file(lang_file[lang], num_examples))
    # plot_token_hist(x_label='seq tokens num', y_label_list=['source', 'target'],
    #                y_data_list=[[len(seq) for seq in source], [len(seq) for seq in target]])

    src_vocab = rnn.text.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = rnn.text.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = ds.load_dataset(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


def main():
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8, num_examples=100)
    for x, x_valid_len, y, y_valid_len in train_iter:
        print('x:', tf.cast(x, tf.int32))
        print('x的有效长度:', x_valid_len)
        print('y:', tf.cast(y, tf.int32))
        print('y的有效长度:', y_valid_len)
        break


if __name__ == "__main__":
    main()
