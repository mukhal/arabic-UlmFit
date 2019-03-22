import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model


with open('./trained_models/lm_model.pt', 'rb') as f:
        model = torch.load(f)


CLF_LAYER = nn.Linear(300, 2)

def clf_forward(input, hidden):
    
    emb = model.drop(model.encoder(input))
    output, hidden = model.rnn(emb, hidden)
    output = model.drop(output)
    decoded = CLF_LAYER(output.view(output.size(0)*output.size(1), output.size(2)))

    return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        