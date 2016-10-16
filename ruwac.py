#!/usr/bin/env python
import argparse
import re
import lzma


tags_re = re.compile(r'<[^<]+?>')
word_re = re.compile(r'(\w[\w\-]*\w|\w)', re.U)


def sentences_iter(filename, only_words=False):
    with smart_open(filename, 'rb') as f:
        sentence = []
        for line in f:
            try:
                word, tag, _ = line.decode('utf-8').split('\t', 2)
            except ValueError:
                continue
            word = word.strip().lower()
            word = tags_re.sub('', word)
            if word and (not only_words or word_re.match(word)):
                sentence.append(word)
            if tag == 'SENT':
                if sentence:
                    yield sentence
                sentence = []


def smart_open(filename, mode):
    if filename.endswith('.xz'):
        inp = lzma.open(filename, mode)
    else:
        inp = open(filename, mode)
    return inp


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('filename')
    arg('out')
    arg('--only-words')
    args = parser.parse_args()

    with open(args.out, 'wt') as f:
        for sent in sentences_iter(args.filename, only_words=args.only_words):
            f.write(' '.join(sent))
            f.write('\n')


if __name__ == '__main__':
    main()
