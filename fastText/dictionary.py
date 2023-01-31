import re
from string import punctuation
import numpy as np

EOS = "</s>"
BOW = "<"
EOW = ">"

digits = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
          6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

MAX_VOCAB_SIZE = 30000000
MAX_N = 3
MIN_N = 1
BUCKET = 2000000
wordNgrams = 2
MIN_COUNT = 1


class Entry:
    def __init__(self, word):
        self.word = word
        self.count = 1
        self.sub_words = []


class Dictionary:
    def __init__(self):
        self.words_ = []
        self.word2int = np.zeros(MAX_VOCAB_SIZE, dtype=int) - 1
        self.size_ = 0
        self.n_tokens = 0
        self.n_word = 0

    def pushHash(self, hashes, id_):
        hashes.append(int(self.n_word + id_))

    @staticmethod
    def readWord(line):
        line = line.lower()
        line = re.sub(r"[^a-z0-9 ]", "", line)
        for d in digits.items():
            line = re.sub(r"{}".format(d[0]), f'{d[1]} ', line)

        return line

    def __find(self, w, h):
        word2size = len(self.word2int)
        id_ = h % word2size
        while (self.word2int[id_] != -1) and (self.words_[int(self.word2int[id_])].word != w):
            id_ = (id_ + 1) % word2size
        return id_

    def hash(self, w):
        FNV_prime = 16777619
        offset_basis = 2166136261
        h = offset_basis
        for char in w:
            h = np.uint32(h ^ np.int8(ord(char)))
            h = np.uint32(h * FNV_prime)
        return np.uint32(h)

    def find(self, w):
        return self.__find(w, self.hash(w))

    def getId(self, w, h):
        id_ = self.__find(w, h)
        return self.word2int[id_]

    def threshold(self, t):
        self.words_ = [word for word in self.words_ if word.count >= t]
        self.word2int = np.zeros(MAX_VOCAB_SIZE, dtype=int) - 1
        self.size_ = 0
        for word in self.words_:
            h = self.find(w=word.word)
            self.word2int[h] = self.size_
            self.size_ += 1
            self.n_word += 1

    def add(self, w):
        h = self.find(w)
        self.n_tokens += 1
        if self.word2int[h] == -1:
            entry = Entry(word=w)
            self.words_.append(entry)
            self.word2int[h] = self.size_
            self.size_ += 1
        else:
            self.words_[int(self.word2int[h])].count += 1

    def computeSubWords(self, word, ngrams):
        for i in range(len(word)):
            ngram = ''
            j = i
            n = 1
            while j < len(word) and n <= MAX_N:
                ngram += word[j]
                j += 1
                if (n >= MIN_N) and not ((n == 1) and (i == 0 or j == len(word))):
                    h = self.hash(ngram) % BUCKET
                    self.pushHash(ngrams, h)
                n += 1

    def addWordNgrams(self, line, hashes, n):
        for i in range(len(hashes)):
            h = hashes[i]
            j = i + 1
            while j < len(hashes) and j < i + n:
                h = np.uint64(h * 116049371 + hashes[j])
                self.pushHash(line, np.uint64(h % BUCKET))
                j += 1

    def getSubWords(self, wid):
        return self.words_[wid].sub_words

    def addSubWords(self, line, token, wid):
        if wid < 0:
            self.computeSubWords(BOW + token + EOW, line)
        else:
            if MAX_N <= 0:
                line.append(wid)
            else:
                ngrams = self.getSubWords(wid)
                line.extend(ngrams)

    def getLine(self, line):
        words = []
        words_hashes = []
        for token in line:
            h = self.find(token)
            wid = self.getId(token, h)
            if wid < 0:
                continue
            self.addSubWords(words, token, wid)
            words_hashes.append(h)
        self.addWordNgrams(words, words_hashes, wordNgrams)
        return words

    def initNgrams(self):
        for i in range(self.size_):
            word = BOW + self.words_[i].word + EOW
            self.words_[i].sub_words = []
            self.words_[i].sub_words.append(i)

            self.computeSubWords(word, self.words_[i].sub_words)

    def readFromFile(self, lines):
        min_t = 1
        for line in lines:
            for word in line.split():
                self.add(word)
                if self.size_ > 0.75 * MAX_VOCAB_SIZE:
                    min_t += 1
                    self.threshold(t=min_t)

        self.threshold(MIN_COUNT)
        self.initNgrams()


if __name__ == '__main__':
    from utils.text_processing import preprocessing_text
    import pandas as pd

    df = pd.DataFrame(data={
        'labels': [0, 0],
        # 'text': ['cat']
        'text': ['the man went out for a walk 12', 'the children sat around the fire 03']
    })
    dictionary = Dictionary()
    texts = df['text']

    print()
    tmp = ['the man went out for a walk', 'the children sat around the fire']

    dictionary = Dictionary()
    dictionary.readFromFile(tmp)

    a = dictionary.getLine('the man went out for a walk'.split())

    print()
