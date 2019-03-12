import os
import torch
import numpy as np
import pickle as pkl

class Dictionary(object):
    """Build word2idx and idx2word from Corpus(train/val/test)"""
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        """Create/Update word2idx and idx2word"""
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """Corpus Tokenizer"""
    def __init__(self, path):
        itos = pkl.load(open(os.path.join(path, 'itos.pkl'), 'rb'))
        self.dictionary = {itos[i]:i for i in range(len(itos))}
        
        train = np.load(os.path.join(path, 'trn_ids.npy'))
        val = np.load(os.path.join(path, 'val_ids.npy'))
        test = np.load(os.path.join(path, 'val_ids.npy'))

        self.train = self.get_tensor(train)
        self.valid = self.get_tensor(val)
        self.test = self.get_tensor(test)

    def get_tensor(self, data):
        
        tokens=[]
        for article in data:
            for id in article:
                tokens.append(id)
        
        return torch.LongTensor(tokens)
        

