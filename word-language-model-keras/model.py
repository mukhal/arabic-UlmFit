import numpy as np
import pandas as pd

from keras.layers import merge, TimeDistributed
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional,GRU, GlobalMaxPool1D, Dropout, Conv1D, MaxPooling1D, Flatten,Convolution1D, Reshape, TimeDistributed
from keras.regularizers import L1L2
from keras.initializers import RandomNormal
from keras.layers.merge import Concatenate
from keras.models import model_from_json
import os 

from keras.preprocessing.sequence import pad_sequences

DATA_DIR = '../data/dialect-identification/'
class General():
    def __init__(self,):
       self.model=None
       # Training parameters
       self.batch_size = None
       self.num_epochs = None
       # Prepossessing parameters
       self.sequence_length = None
       self.vocab_size = None  ## changed to fit data size
       self.LoadedModel=None
       self.Model=None
       self.ExternalEmbeddingModel = None
       self.EmbeddingType=None

    def set_etxrernal_embedding(self,ModelFile,ModelType):
        self.ExternalEmbeddingModel=ModelFile
        self.EmbeddingType=ModelType

    def set_training_paramters(self,batch_size,num_epochs):
        self.batch_size=batch_size
        self.num_epochs=num_epochs

    def set_processing_parameters(self,sequence_length,vocab_size):
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size

    def train_model(self,Model,X_train,Y_train,X_valid,Y_valid):
        Model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=self.num_epochs, batch_size=self.batch_size)

    def Evaluate_model(self,Model,X_test,Y_test):
        score=Model.evaluate(X_test,Y_test,verbose=0)
        return score

    def save_model(self,ModelFileName,Model):
        print("Saving model in directory:")
        JsonModel = Model.to_json()
        with open('models/' + ModelFileName + ".json", "w") as json_file:
            json_file.write(JsonModel)
        Model.save_weights('models/' + ModelFileName + ".h")
        print('model saved in directory')

    def Load_model(self,ModelFileName):
        print("Loading Model from directory!")

        JsonFile = open(ModelFileName+".json",'r')

        # Load Json file
        LoadedModel = model_from_json(JsonFile)

        # Load weights
        LoadedModel.load_weights(ModelFileName+".h5")
        LoadedModel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return LoadedModel


class BasicBiGRUs(General): ## inherits General
    '''
    BiLSTM Our implementation
    '''
    def __init__(self,BiGRU_rand=True,STATIC=False,ExternalEmbeddingModel=None,EmbeddingType=None,n_symbols=None,wordmap=None):
        self.embedding_dim = 300
        self.hidden_dims = 100
        self.dropout_prob = (0.5, 0.8)
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'
        self.l1_reg = 0
        self.l2_reg = 3  ##according to kim14
        self.std = 0.05  ## standard deviation
        # Training Parameters
        self.set_training_paramters(batch_size=64, num_epochs=10)
        self.set_processing_parameters(sequence_length=30, vocab_size=50002)  ## changed to fit short text
        # Defining Model Layers        if clstm_rand:
        ##Embedding Layer Randomly initialized
        if BiGRU_rand:
            ##Embedding Layer Randomly initialized
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size)
            Classes = ['DIAL','MSA']
            n_classes = len(Classes)

        else:
            ## Use pretrained model
            # n_symbols, wordmap = dh.get_word_map_num_symbols()
            self.set_etxrernal_embedding(ExternalEmbeddingModel, ModelType=EmbeddingType)
            if self.EmbeddingType == "skipgram" or self.EmbeddingType == "CBOW":
                vecDic = dh.GetVecDicFromGensim(self.ExternalEmbeddingModel)
            elif self.EmbeddingType == "fastText":
                vecDic = dh.load_fasttext(self.ExternalEmbeddingModel)
            Classes = dh.read_labels()
            n_classes = len(Classes)
            ## Define Embedding Layer
            embedding_weights = dh.GetEmbeddingWeights(embedding_dim=300, n_symbols=n_symbols, wordmap=wordmap,
                                                       vecDic=vecDic)
            embedding_layer = Embedding(output_dim=self.embedding_dim, input_dim=n_symbols, trainable=STATIC)
            embedding_layer.build((None,))  # if you don't do this, the next step won't work
            embedding_layer.set_weights([embedding_weights])

        Sequence_in = Input(shape=(self.sequence_length,), dtype='int32')
        embedding_seq = embedding_layer(Sequence_in)
        x = Dropout(self.dropout_prob[0])(embedding_seq)
        x = Bidirectional(GRU(self.hidden_dims, kernel_initializer=RandomNormal(stddev=self.std),
                 kernel_regularizer=L1L2(l1=self.l1_reg, l2=self.l2_reg), return_sequences=True))(x)
        
        preds = Dense(n_classes, activation='softmax')

        
        lm_output = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)
        ## return graph model
        model = Model(Sequence_in, lm_output)
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])


        self.model = model

### configs
bptt=35
batch_size=64
seq_len = 30


def get_batch(source, i, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target


def get_x_y(data):
    X = []
    Y = []
    for i in range(0, len(data), bptt):
        x, y = get_batch(data, i)
        X.append(x)
        Y.append(y)

    
    X = pad_sequences(X, maxlen=seq_len)
    Y = pad_sequences(Y, maxlen=seq_len)

    print(X.shape)
    return X, Y   



def get_flattened(data):

    f=[]
    for l in data:
        for id in l:
            f.append(id)
    return f

if __name__=='__main__':

    model = BasicBiGRUs().model

    train = np.load(os.path.join(DATA_DIR, 'trn_ids.npy'))
    val = np.load(os.path.join(DATA_DIR, 'val_ids.npy'))
    test = np.load(os.path.join(DATA_DIR, 'val_ids.npy'))

    train = get_flattened(train)
    val = get_flattened(val)
    test= get_flattened(test)


    x_train, y_train =get_x_y(train)
    x_val, y_val = get_x_y(val)


    model.fit(x_train, y_train, batch_size=32)
    

