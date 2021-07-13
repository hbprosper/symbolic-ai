import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from IPython.display import display

# symbolic symbols
from sympy import Symbol, exp, \
    cos, sin, tan, \
    cosh, sinh, tanh, ln
x = Symbol('x')

class Seq2SeqDataPreparer:
    '''
    This class maps the source (i.e., input) and target (i.e, output) 
    sequences of characters into sequences of indices. The source data 
    are split into x_train, x_valid, and x_test sets and similarly for 
    the target data.
    
    Create a data preparer using
    
    dd = Seq2SeqDataPreparer(X, Y, fractions)
    
    where the shape of dd.x_* and dd.y_* is 
       
       (max_seq_len, batch_size)
       
    (* = train, valid, test)
    
    and where,
      size:         number of instances in data set
      max_seq_len:  max sequence length (# characters)
      fractions:    a 2-tuple containing the three-way split of data.
                    e.g.: (5/6, 5.5/6) means split the data as follows
                    (50000, 5000, 5000)
    Note: max_seq_len in general differ for source and target.
    '''
    def __init__(self, X, Y=None, fractions=[5/6,5.5/6]):
        
        # get maximum sequence length for input expressions
        self.x_max_seq_len =  max([len(z) for z in X])
        
        # code data
        N = int(len(X)*fractions[0])
        M = int(len(X)*fractions[1])
        
        # create token to index map from training data
        t = self.token_tofrom_index(X[:N])
        self.x_token2index, self.x_index2token = t
        
        
        self.x_train = self.code_data(X[:N], 
                                      self.x_token2index,
                                      self.x_max_seq_len)
        
        self.x_valid = self.code_data(X[N:M], 
                                      self.x_token2index,
                                      self.x_max_seq_len)
        
        self.x_test  = self.code_data(X[M:], 
                                      self.x_token2index,
                                      self.x_max_seq_len)
        
        if not None:
            self.y_max_seq_len =  max([len(z) for z in Y])
        
            # create token to index map from training data
            t = self.token_tofrom_index(Y[:N])
            self.y_token2index,self.y_index2token = t
            
            self.y_train = self.code_data(Y[:N], 
                                          self.y_token2index, 
                                          self.y_max_seq_len)
        
            self.y_valid = self.code_data(Y[N:M], 
                                          self.y_token2index, 
                                          self.y_max_seq_len)

            self.y_test  = self.code_data(Y[M:], 
                                          self.y_token2index, 
                                          self.y_max_seq_len)
        
    def __del__(self):
        pass
    
    def __len__(self):
        # shape (max_seq_len, size)
        n  = 0
        n += len(self.x_train[1])
        n += len(self.x_valid[1])
        n += len(self.x_test[1])
        return n
    
    def __str__(self):
        s  = ''
        s += 'number of seq-pairs (train): %8d\n'%len(self.x_train[1])
        s += 'number of seq-pairs (valid): %8d\n'%len(self.x_valid[1])
        s += 'number of seq-pairs (test):  %8d\n'%len(self.x_test[1])
        s += '\n'
        s += 'number of source tokens:     %8d\n' % \
        len(self.x_token2index)
        s += 'max source sequence length:  %8d\n' % \
        self.x_max_seq_len
        
        try:
            s += '\n'
            s += 'number of target tokens:     %8d\n' % \
            len(self.y_token2index)
            s += 'max target sequence length:  %8d' % \
            self.y_max_seq_len
        except:
            pass
        return s
         
    def num_tokens(self, which='source'):
        if which[0] in ['s', 'i']:
            return len(self.x_token2index)
        else:
            return len(self.y_token2index)
    
    def max_seq_len(self, which='source'):
        if which[0] in ['s', 'i']:
            return self.x_max_seq_len
        else:
            return self.y_max_seq_len
        
    def decode(self, indices):
        # map list of indices to a list of tokens
        return [self.y_index2token[i] for i in indices]

    def token_tofrom_index(self, expressions):
        chars = set()
        chars.add(' ')  # for padding
        chars.add('?')  # for unknown characters
        for expression in expressions:
            for char in expression:
                chars.add(char)
        chars = sorted(list(chars))
        
        char2index = dict([(char, i) for i, char in enumerate(chars)])
        index2char = dict([(1, char) for i, char in enumerate(chars)])
        return (char2index, index2char)
        
    def code_data(self, data, token2index, maxseqlen):
        
        # shape of data: (max_seq_len, size)
        
        cdata   = np.zeros((maxseqlen, len(data)), dtype='long')
        space   = token2index[' ']
        unknown = token2index['?']
        for i, expression in enumerate(data):
            for t, char in enumerate(expression):
                try:
                    cdata[t, i] = token2index[char]
                except:
                    cdata[t, i] = unknown
        
            # pad with spaces
            cdata[t + 1:, i] = space
        return cdata
    
    
# Dataset class to return source and target "sentences"
class Seq2SeqDataset(Dataset):
    '''
    dataset = Seq2SeqDataset(X, Y)
    
    shape of data: (max_seq_len, size)
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
 
    def __len__(self):
        return len(self.X[1])
  
    def __getitem__(self, index):
        # shape of output data: (max_seq_len)
        return self.X[:,index], self.Y[:,index]
    
# See tips on how to increase PyTorch performance:
# https://towardsdatascience.com/
# 7-tips-for-squeezing-maximum-performance-from-pytorch-ca4a40951259

class Seq2SeqDataLoader:
    '''
    dataloader = Seq2seqDataLoader(X, Y, batch_size=128, shuffle=True)
    
    '''
    def __init__(self, X, Y, 
                 batch_size=128, 
                 shuffle=False):
        self.dataset    = Seq2SeqDataset(X, Y)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.dataloader = DataLoader(self.dataset, 
                                     batch_size=batch_size, 
                                     shuffle=shuffle,
                                     pin_memory=True)
        self.iter = iter(self.dataloader)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            # If GPU is being used, even though the memory is pinned,
            # we may still have to transfer these to the GPU explicitly
            X, Y  = self.iter.next()
            # need shape: (max_seq_len, batch_size)
            return X.transpose(0,1), Y.transpose(0,1)
        except:
            raise StopIteration
            
    def reset(self):
        self.iter = iter(self.dataloader)
        
# Delimit each sequence in filtered sequences
# The start of sequence (SOS) and end of sequence (EOS) 
# tokens are "\t" and "\n", respectively.

def loadData(inpfile):
    # format of data:
    # input expression<tab>target expression<newline>
    data = [a.split('\t') for a in open(inpfile).readlines()]
    
    X, Y = [], []
    for i, (x, y) in enumerate(data):
        X.append('\t%s\n' % x)
        # get rid of spaces in target sequence
        y = ''.join(y.split())
        Y.append('\t%s\n' % y)
        if i % 2000 == 0:
            print(i)
            # pretty print expressions
            pprint(X[-1])
            pprint(Y[-1])
            print()
    return (X, Y)

def pprint(expr):
    display(eval(expr))
