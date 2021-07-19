import numpy as np
import torch
from IPython.display import display

# symbolic symbols
from sympy import Symbol, exp, \
    cos, sin, tan, \
    cosh, sinh, tanh, ln, log, E
x = Symbol('x')

class Seq2SeqDataPreparer:
    '''
    This class maps the source (i.e., input) and target (i.e, output) 
    sequences of characters into sequences of indices. The source data 
    are split into x_train, x_valid, and x_test sets and similarly for 
    the target data.
    
    Create a data preparer using
    
    dd = Seq2SeqDataPreparer(X, Y, fractions)
    
    where,

      fractions:    a 2-tuple containing the three-way split of data.
                    e.g.: (50/60, 55/60) means split the data as follows
                    (50000, 5000, 5000)
    '''
    def __init__(self, X, Y,
                 fractions=[50/60, 55/60]): 
        
        self.fractions = fractions
        
        # Get maximum sequence length for input expressions
        self.x_max_seq_len =  max([len(z) for z in X])
        
        # Get maximum sequence length for target expressions
        self.y_max_seq_len =  max([len(z) for z in Y])
        
        # get length of splits into train, valid, test
        N = int(len(X)*fractions[0])
        M = int(len(X)*fractions[1])
        
        # Create token to index map for source sequences
        t = self.token_tofrom_index(X[:N])
        self.x_token2index, self.x_index2token = t
        
        # Create token to index map for target sequences
        t = self.token_tofrom_index(Y[:N])
        self.y_token2index,self.y_index2token = t
        
        # Structure data into a list of blocks, where each block
        # comprises a tuple (x_data, y_data) whose elements have
        #   x_data.shape: (x_seq_len, batch_size)
        #   y_data.shape: (y_seq_len, batch_size)
        #
        # The sequence and batch sizes can vary from block to block.
        
        self.train_data, self.n_train = self.code_data(X[:N], Y[:N])         
        self.valid_data, self.n_valid = self.code_data(X[N:M],Y[N:M])
        self.test_data,  self.n_test  = self.code_data(X[M:], Y[M:])

    def __del__(self):
        pass
    
    def __len__(self):
        n  = 0
        n += self.n_train
        n += self.n_valid
        n += self.n_test
        return n
    
    def __str__(self):
        s  = ''
        s += 'number of seq-pairs (train): %8d\n' % self.n_train
        s += 'number of seq-pairs (valid): %8d\n' % self.n_valid
        s += 'number of seq-pairs (test):  %8d\n' % self.n_test
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
        return ''.join([self.y_index2token[i] for i in indices])

    def token_tofrom_index(self, expressions):
        chars = set()
        chars.add(' ')  # for padding
        chars.add('?')  # for unknown characters
        for expression in expressions:
            for char in expression:
                chars.add(char)
        chars = sorted(list(chars))
        
        char2index = dict([(char, i) for i, char in enumerate(chars)])
        index2char = dict([(i, char) for i, char in enumerate(chars)])
        return (char2index, index2char)
       
    def get_block_indices(self, X, Y):
        # X, and Y are just arrays of strings.
        #
        # 1. Following Michael Andrews' suggestion double sort 
        #    expressions, first with targets then sources. But, also
        #    note the ordinal values "i" of the expressions in X, Y.
        sizes = [(len(a), len(b), i) 
                 for i, (a, b) in enumerate(zip(Y, X))]
        sizes.sort()
  
        # 2. Find ordinal values (indices) of all expression pairs 
        #    for which the sources are the same length and the
        #    targets are the same length. In general, the sources and
        #    targets differ in length.
     
        block_indices = []
        n, m, i  = sizes[0] # n, m, i = len(target), len(source), index
        previous = (n, m)
        indices  = [i] # cache index of first expression
        
        for n, m, i in sizes[1:]: # skip first expression
            
            size = (n, m)
            
            if size == previous:
                indices.append(i) # cache index of expression
            else:
                # found a new boundary, so save previous 
                # set of indices...
                block_indices.append(indices)
                
                # ...and start a new list of indices
                indices = [i]

            previous = size
            
        # cache expression indices of last block
        block_indices.append(indices)
        
        return block_indices
    
    
    def make_block(self, expressions, indices, token2index, unknown):
        
        # batch size of current block
        batch_size = len(indices)
        
        # By construction, all expressions of a block have 
        # the same length, so can use the length of first expression
        seq_len = len(expressions[indices[0]])
        
        # Create an empty block of correct shape and size
        data    = np.zeros((seq_len, batch_size), dtype='long')
        #print('seq_len, batch_size: (%d, %d)' % (seq_len, batch_size))
        
        # loop over expressions for current block
        # m: ordinal value of expression in current block
        # k: ordinal value of expression in original list of expressions
        # n: ordinal value of character in a given expression
        
        for m, k in enumerate(indices):
            
            expr = expressions[k]
            
            #print('%5d expr[%d] | %s |' % (m, k, expr[1:-1]))
            
            # copy coded characters to 2D arrays
        
            for n, char in enumerate(expr):
                #print('\t\t(n, m): (%d, %d)' % (n, m))
                try:
                    data[n, m] = token2index[char]
                except:
                    data[n, m] = unknown
                    
        return data
    
    def code_data(self, X, Y):
        # Implement Arvind's idea
        
        # X, Y consist of delimited strings: 
        #   \tab<characters\newline
        
        # loop over sequence pairs and convert them to sequences
        # of integers using the two token2index maps
      
        x_space   = self.x_token2index[' ']
        x_unknown = self.x_token2index['?']
        
        y_space   = self.y_token2index[' ']
        y_unknown = self.y_token2index['?']
 
        # 1. Get blocks containing sequences of the same length.
        
        block_indices = self.get_block_indices(X, Y)
        
        # 2. Loop over the indices associated with each block of coded
        #    sequences. The indices are the ordinal values of the
        #    sequence pairs X and Y.
        
        blocks = []
        n_data = 0
       
        for indices in block_indices:

            x_data = self.make_block(X, indices, 
                                     self.x_token2index, x_unknown)
 
            y_data = self.make_block(Y, indices, 
                                     self.y_token2index, y_unknown)

            blocks.append((x_data, y_data))
            n = len(indices)
            n_data += n
        
        assert n_data == len(X)
        
        return blocks, n_data
    
class Seq2SeqDataLoader:
    '''
    dataloader = Seq2seqDataLoader(dataset, device, sample=True)    
    '''
    def __init__(self, dataset, device, sample=True):
        self.dataset = dataset
        self.device  = device
        self.sample  = sample  
        self.init()

    def __iter__(self):
        return self
    
    def __next__(self):
        
        # increment iteration counter
        self.count += 1
        
        if self.count <= self.max_count:
            
            # 1. randomly pick a block or return blocks in order.
            if self.sample:
                k = np.random.randint(len(self.dataset))
            else:
                k = self.count-1 # must subtract one!
            
            # 2. create tensors directly on the device of interest
            X = torch.tensor(self.dataset[k][0], 
                             device=self.device)
            
            Y = torch.tensor(self.dataset[k][1], 
                             device=self.device)
        
            # shape of X and Y: (seq_len, batch_size)
            return X, Y
        else:
            self.count = 0
            raise StopIteration
            
    def init(self, max_count=0, sample=True):
        n_data = len(self.dataset)
        self.max_count = n_data if max_count < 1 else min(max_count, 
                                                          n_data)
        self.sample= sample
        self.count = 0
        
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
        
    print('Example source:')
    print(X[-1])
    pprint(X[-1])
    print('Example target:')
    print(Y[-1])
    pprint(Y[-1])

    return (X, Y)

def pprint(expr):
    display(eval(expr))
