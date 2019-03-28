import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

# Add ckp
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='/input', # /input
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='',
                    help='model checkpoint to use')
parser.add_argument('--pretrained', type=str, default='',
                    help='pre-trained model to finetune')
parser.add_argument('--finetune', action='store_true', default=False, 
                    help='finetune existing model')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--bidirectional', action='store_true', default=False)
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./trained_models/lm_model.pt', # /output
                    help='path to save the final model')

parser.add_argument('--slanted_lr', action='store_true', default=False, help='Use Slanted Learning Rates')
parser.add_argument('--cut_frac', type=float, default=0.1, help='Slanet LR cut fraction')
parser.add_argument('--ratio', type=float, default=32.0, help='Slanted Learning Rates Ratio')
parser.add_argument('--lr_max', type=float,default=0.01, help='slanted learning rate max value')
 


args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

def get_discriminative_lr(parameter_name, current_lr, n_layers=args.nlayers):

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
    

def scheduled_slanted_lr(iter, T, cut_frac, ratio, lr_max):
    cut = int(cut_frac * T)
    if iter < cut:
        p= iter / cut
    else:
        p = 1 - (iter-cut) / (cut*(1/cut_frac - 1))
    
    return lr_max * (1+p *(ratio - 1)) / ratio

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #print("batchify result size = " , data.size())
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size)
print(train_data.size())
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)


T = (train_data.size(0) // args.bptt) * args.epochs
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, args.bidirectional)

# Load checkpoint
if args.checkpoint != '':
    if args.cuda:
        model = torch.load(args.checkpoint)
    else:
        # Load GPU model on CPU
        model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

if args.finetune:
    assert args.pretrained, "you must specify a pre-trained model"
    
    with open(args.pretrained, 'rb') as f:
        model = torch.load(f)
    print("loaded pre-trained model...")

if args.cuda:
    model.cuda()
else:
    model.cpu()
print (model)

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        if isinstance(h, tuple) or isinstance(h, list):
            return tuple(repackage_hidden(v) for v in h)
        else:
            return h.detach()


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).item()
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset 
        # (previous batches).
        
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data) # p.data = p.data += -lr * p.grad.data

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def finetune(cur_iter, cur_learning_rate):
 # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    

    print("cur learning rate = ", cur_learning_rate)
    print("iteration" , cur_iter)
       

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

        data, targets = get_batch(train_data, i)
       # print(data.size())
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset 
        # (previous batches).
        
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.add_(-get_discriminative_lr(name, cur_learning_rate), p.grad.data) # p.data = p.data += -lr * p.grad.data
        
        total_loss += loss.item()

        
        cur_learning_rate = scheduled_slanted_lr(cur_iter, T, args.cut_frac, args.ratio, args.lr)
        cur_iter+=1
        #print(batch)
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:04.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, cur_learning_rate,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return cur_iter, cur_learning_rate


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.

if __name__ =='__main__':

    try:
        it=0
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()

            if args.finetune: # finetuning
                print(f"finetuning {args.pretrained}")    
                it, lr = finetune(it, lr)   
            
            else : 
                train()
      
            val_loss = evaluate(val_data)

            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb+') as f:
                    torch.save(model, f)
            
                best_val_loss = val_loss
            
            elif not args.finetune:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
                
            
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
           
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
