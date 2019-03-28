import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim 

import data
import model
import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
import torch.nn.init as weight_init

torch.backends.cudnn.enabled = False

with open('./trained_models/LM_GRU_finetuned.pt', 'rb') as f:
        original_model = torch.load(f)

original_model.cuda()
N_CLASSES = 2

criterion = nn.CrossEntropyLoss()
criterion.cuda()

lr = 0.004

class FinetunedModel(nn.Module):

    def __init__(self, fc_size=50, freeze=True):
        super(FinetunedModel, self).__init__()
        self.original_model = original_model
        self.fc= nn.Linear(200, fc_size)
        self.decoder = nn.Linear(400, N_CLASSES)

        self.drop = nn.Dropout(0.6)


        if freeze:
            for p in self.original_model.parameters():
                p.requires_grad=False
        

    def forward(self, input, hidden):
        emb = self.drop(self.original_model.encoder(input))
        output, hidden = self.original_model.rnn(emb, hidden)
        output = self.drop(self.original_model.drop(output))

        max_pooled, _ = torch.max(output, dim=0)
        mean_pooled = torch.mean(output, dim=0)
        last_step = output[-1,:,:]

        concat_pooling = torch.cat([mean_pooled, max_pooled], dim=-1) # S x B x 3H

        #decoded = torch.tanh(self.fc(concat_pooling))
        decoded=concat_pooling
        #decoded = self.drop(torch.relu(self.fc(decoded)))
        decoded= self.decoder(decoded)
        decoded = decoded.view(output.size(1), N_CLASSES) # B x N_CLASSES
 

        return decoded, hidden


model =FinetunedModel()
model.cuda()



def get_discriminative_lr(parameter_name, current_lr, n_layers):

    return current_lr
    discriminative_lrs = []
    discriminative_lrs.append(lr) # decoder lr

    for k in range(n_layers): ## Discriminative learning rates
        discriminative_lrs.insert(0, lr / 2.6**k) # LSTM Lr

    discriminative_lrs.insert(0, discriminative_lrs[0] / 2.6)# Encoder Lr

    if 'encoder' in parameter_name:
        return discriminative_lrs[0]
    elif 'decoder' in parameter_name:
        return discriminative_lrs[-1]
    else:
        for layer in range(len(discriminative_lrs) - 2):
            if 'l%d'%(layer) in parameter_name:
                return discriminative_lrs[layer+1]
    

def get_batches(seq, lbls, batch_size=32, max_seq_length=30):

    assert len(seq) == len(lbls)

    seq = pad_sequences(seq, maxlen=max_seq_length)

    for i in range(0, len(seq), batch_size):
        yield seq[i:i+batch_size], lbls[i:i+batch_size]


def train():
    model.train()

    start_time = time.time()

    iter = 0
    total_loss=0
    acc = 0

    optimizer = optim.Adam(model.parameters(), lr = lr)


    for x, y in get_batches(train_seq, train_labels, batch_size=args.batch_size):
        iter+=1

        hidden = model.original_model.init_hidden(len(x))
        model.zero_grad()

        x= torch.LongTensor(x).cuda().transpose(0, 1)
        y= torch.LongTensor(y).cuda()

        output, hidden = model(x, hidden)

        loss = criterion(output.view(-1, N_CLASSES), y)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        '''
        for name, p in model.named_parameters():
            if 'original_model.decoder' not in name and p.requires_grad:
                p.data.add_(-get_discriminative_lr(name, lr, 2), p.grad.data) # p.data = p.data += -lr * p.grad.data
        
        '''
        optimizer.step()
        
        total_loss += loss.item()

        _, predictions = torch.max(output, dim=-1)
        predictions = predictions.cpu().numpy()
        acc += accuracy_score( y.cpu().numpy(), predictions) 

        if iter % 200 == 0 and iter > 0:
            cur_loss = total_loss / 200
            acc = acc / 200
            elapsed = time.time() - start_time
            print('| epoch {:3d} |loss {:5.4f}| accuracy {:5.3f}'.format(
                epoch, cur_loss, acc))
            total_loss = 0
            acc=0
            start_time = time.time()


def evaluate():
    model.eval()

    iter = 0
    total_loss=0
    acc = 0

    all_predictions=[]
    for x, y in get_batches(val_seq, val_labels, batch_size=args.batch_size):
        iter+=1

        hidden = model.original_model.init_hidden(len(x))

        x= torch.LongTensor(x).cuda().transpose(0, 1)
        y= torch.LongTensor(y).cuda()

        output, hidden = model(x, hidden)

        loss = criterion(output.view(-1, N_CLASSES), y)
        total_loss += loss.item()

        _, predictions = torch.max(output, dim=-1)
        predictions = predictions.cpu().numpy()
        all_predictions.extend(predictions)

    
    cur_loss = total_loss / iter
    acc = accuracy_score(val_labels, all_predictions)
    print('Validation loss {:5.4f}| accuracy {:5.4f}'.format(
        cur_loss, acc))
    
    return cur_loss, acc

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../data/dialect-identification')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save', type=str, default='./trained_models/dial_clf.pt')

    #parser.add_argument('--lr', type=float, default=0.001)


    args = parser.parse_args()                                                                                 
    
    train_seq, train_labels = np.load(os.path.join(args.data, 'trn_ids.npy')), np.load(os.path.join(args.data, 'trn_lbls.npy'))
    val_seq, val_labels = np.load(os.path.join(args.data, 'val_ids.npy')), np.load(os.path.join(args.data, 'val_lbls.npy'))

    train_seq, train_labels = unison_shuffled_copies(train_seq, train_labels)
    print(len(train_seq))

    best_acc = 0
    for epoch in range(args.epochs):

        if (epoch >=3):
            for p in model.original_model.parameters():
                p.requires_grad=True

        train()
        print('-' * 89)
        ls, acc = evaluate()
        print('-' * 89)

        if acc > best_acc:
            best_acc=acc
            print ("saving model")
            with open(args.save, 'wb+') as f:
                torch.save(model, f)


            
        