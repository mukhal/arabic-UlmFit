from nltk.tokenize import wordpunct_tokenize
import argparse
import pandas as pd
import pickle as pkl
import numpy as np


from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='train', help='location of the data corpus')
parser.add_argument('--dictionary', type=str, default='../wiki/ar/tmp/itos.pkl')

if __name__ == '__main__':
    args = parser.parse_args()
    itos = pkl.load(open(args.dictionary, 'rb'))

    print(len(itos))
    word_dictionary = defaultdict(lambda : 0)

    for index, word in enumerate(itos):
        word_dictionary[word] = index

    data_file = args.data
    df = pd.read_csv(data_file+'.csv')

    all_ids = []
    labels= []

    for sent, label in zip(df['text'].astype(str), df['label'].astype(str)):
        tokens = wordpunct_tokenize(sent)
        ids = [word_dictionary['xbos']] +  [word_dictionary[t] for t in tokens]
        all_ids.append(ids)
        labels.append(1 if label=='DIAL' else 0)


    zero_count=0
    total_len = 0

    for sent in all_ids:
        unique, counts = np.unique(sent, return_counts=True)
        d = dict(zip(unique, counts))
        zero_count += d[0] if 0 in d else 0
        total_len+= len(sent)

    print("unk ratio = ", zero_count * 1.0 / total_len)


        
    m = np.array(all_ids)
    labels = np.array(labels)
    assert len(m) == len(labels)
    print(labels)
    np.save(args.data +'_ids.npy', m)
    np.save(args.data +'_lbls.npy', labels)
