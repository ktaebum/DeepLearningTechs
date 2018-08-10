"""
File: nlp.py
Author: Taebum Kim
Email: phya.ktaebum@gmail.com
Github: https://github.com/ktaebum
Description:
    NLP Module
"""
import os
import re
import json

import nltk
from collections import Counter


class Vocabulary:
    """
    Simple vocabulary for nlp
    """
    SOS = '<SOS>'  # start of sentence
    EOS = '<EOS>'  # end of sentence
    UNK = '<UNK>'  # unknown word
    PAD = '<PAD>'  # for padding

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

        self.n_words = 0  # number of words

        self(Vocabulary.SOS)
        self(Vocabulary.EOS)
        self(Vocabulary.UNK)
        self(Vocabulary.PAD)

    def __getitem__(self, word_or_idx):
        """
        if in dictionary, return idx
        else return UNK's idx
        """

        if isinstance(word_or_idx, str):
            return self.word2idx.get(word_or_idx,
                                     self.word2idx[Vocabulary.UNK])
        else:
            return self.idx2word.get(word_or_idx, Vocabulary.UNK)

    def __len__(self):
        return self.n_words

    def __call__(self, word):
        """
        Add new word to dictionary

        return index id
        """
        if word not in self.word2idx:
            # new word
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

        return self.word2idx[word]


def sentence2words(sentence, normalize=True):
    """
    Convert sentence to word list
    """
    sentence = sentence.lower().strip()

    if normalize:
        sentence = re.sub(r'([.!?])', ' \1', sentence)
        sentence = re.sub(r'[^a-zA-Z.!?]+', r' ', sentence)

    words = sentence.split()

    return words


def build_coco_vocabulary(filename=None, threshold=4):
    """
    Vocabulary builder for coco caption

    From caption train2014 json
    """
    vocab = Vocabulary()
    if filename is None:
        filename = 'captions_train2014.json'

    annotation_file = os.path.join('./data/annotations', filename)

    imgs = json.load(open(annotation_file))
    imgs = imgs['annotations']

    counter = Counter()
    for img in imgs:
        words = nltk.tokenize.word_tokenize(img['caption'].lower())
        counter.update(words)

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    for word in words:
        vocab(word)

    return vocab
