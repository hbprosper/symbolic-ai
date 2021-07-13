# symbolic-ai
A sequence to sequence (seq2seq) model to map mathematical expressions to their Taylor series expansion. For July 2021 Carnegie Mellon University hackathon in AI for physics conference: https://events.mcs.cmu.edu/qtc2021/

## Introduction
The jupyter notebooks in this package depend on the following Python packages:

| __modules__   | __description__     |
| :---          | :---        |
| re            | regular expressions |
| sympy         | an excellent symbolic algebra module |
| numpy         | array manipulation and numerical analysis      |
| random        | random numbers |
| torch         | PyTorch machine learning toolkit |

This package can be installed as follows:
```
git clone https://github.com/hbprosper/symbolic-ai
```

## Notebooks

| __notebook__ | __description__ |
| :---         | :--- |
| seq2seq_data_generation | generate pairs of mathematical expressions |
| seq2seq_data_prep | data preparation  |
| seq2seq_train | train! |

## Quick start
The file __data/seq2seq_data.txt__ contains sequence pairs, one per line, with format

```python
<symbolic-expression><\tab><symbolic-Taylor-series-expression>
```

which were created using __seq2seq_data_generation.ipynb__. For example, the first line in that file is
```python
sinh(-2*x)      -2*x - 4*x**3/3
```  
Since the data are already available, there is no need to call this notebook. The notebook __seq2seq_data_prep.ipynb__ applies some filtering to __data/seq2seq_data.txt__ and creates the filtered text files __data/seq2seq_data_10000.txt__ and __data/seq2seq_data_60000.txt__ in which all spaces in the expressions have been removed. The first file contains 10,000 sequence pairs and the second contains 60,000 sequence pairs.

### Training
The symbolic translation model can be trained using the jupyter notebook __seq2seq_train.ipynb__, which should be run on a system with GPU support (e.g., Google Colaboratory). The notebook defines a sequence to sequence (seq2seq) model comprising a sequence encoder followed by a sequence decoder, each built using two or more layers of Long Short Term Memories (LSTM). An LSTM returns __output, (hidden, cell)__, where, for a given sequence, output is a tensor of floats equal in size to the number of unique tokens from which the sequences are formed plus 2. The extra length of 2 is for a 2 extra tokens, one for padding (a space) and another for an unknown character (a question mark). The objects __hidden__ and __cell__ are the so-called hidden and cell states, respectively, which provide encodings of the input sequence. In spite of its suggestive name an LSTM is just another very clever non-linear function that was developed by conceptualizing a device containing various filtering elements.

The __seq2seq_train.ipynb__ notebook does the following:

  1. Read a filtered text file and delimit each sequence of characters with a tab and a newline character.
  2. Build a character (i.e., token) to integer map for the input sequences and another for the target (that is, output) sequences.
  3. Use the maps to convert each sequence to an array of integers, where each integer corresponds to a unique character, and pad the sequences with spaces so that the sequences are of the same length. Do this separately for input and target sequences. (Padding is needed to simplify the use of *batches* during training.)
  4. Create an Encoder, which performs the following tasks:
     1. Map each integer encoding of a token to a dense vector representation using the PyTorch __Embedding__ class.
     2. Call a stack of LSTMs keeping only the hidden and cell states.

__Encoder__
  .1 Map 

```python
from google.colab import drive 
drive.mount('/content/gdrive')
import sys
sys.path.append('/content/gdrive/My Drive/AI')
```
