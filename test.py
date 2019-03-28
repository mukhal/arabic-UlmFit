import numpy as np
import pickle as pkl
from pyarabic.araby import  normalize_hamza, normalize_ligature


trn_ids = np.load('data/wiki/ar/tmp/val_ids.npy')
it = pkl.load(open('data/wiki/ar/tmp/itos.pkl', 'rb'))

t= 'وإن لم يريد أن يفعل بي الفصحى'

print(normalize_ligature(t))

