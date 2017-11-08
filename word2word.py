import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import io
import torch
from torchvision import models, transforms, datasets
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as t
from torch.nn import Parameter

import os
import os.path
import pickle

class TranscriptionsReader(data.Dataset):


    def __init__(self, file_path):


        self.word2id = {}
        self.id2word = {}

        self.char2id = {}
        self.id2char = {}

        self.characters = []
        self.targets = []

        self.padding = self.getWordId('PADDING_TOKEN')
        self.char_padding = self.getCharId('PADDING_TOKEN')
#        self.unknownToken = self.getWordId('UNKNOWN_TOKEN')

        with open(file_path) as text_file:
            for line in text_file:
                word=line.strip()
                wordId = self.getWordId(word)
                charlist = []
                for char in word:
                    charID = self.getCharId(char)
                    charlist.append(charID)
                self.characters.append(charlist)
                self.targets.append(wordId)


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: ([word_ids], [target]) where target is the list of scores of each of the                     personality traits.
        """

        target = self.targets[index]
        charlist = self.characters[index]

        return(charlist, target)

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does              not exist and create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # Get the id if the word already exist
        wordId = self.word2id.get(word, -1)

        # If not, we create a new entry
        if wordId == -1:
            if create:
                wordId = len(self.word2id)
                self.word2id[word] = wordId
                self.id2word[wordId] = word

            else:
                wordId = self.unknownToken

        return wordId


    def getCharId(self, char, create=True):
        """Get the id of the char (and add it to the dictionary if not existing). If the char does              not exist and create is set to False, the function will return the unknownToken value
        Args:
            char (str): char to add
            create (Bool): if True and the char does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only chars with more than one occurrence ?

        char = char.lower()  # Ignore case

        # Get the id if the char already exist
        charId = self.char2id.get(char, -1)

        # If not, we create a new entry
        if charId == -1:
            if create:
                charId = len(self.char2id)
                self.char2id[char] = charId
                self.id2char[charId] = char

            else:
                charId = self.unknownToken

        return charId

def my_test_collate(batch):

    _prueba_batch = batch
    max_length=18
    nwords=18

    data_tensor = torch.LongTensor(
        len(_prueba_batch),
        max_length
        ).zero_()
    data_tensor.size()

    for n in range(len(_prueba_batch)):
        nwords2 = len(_prueba_batch[n][0])
        for word_idx in range(nwords2):
            data_tensor[n][word_idx+round((nwords-nwords2)/2)] = _prueba_batch[n][0][word_idx]

    target = torch.LongTensor( len(_prueba_batch)).zero_()
    for n in range(len(_prueba_batch)):
        target[n] = _prueba_batch[n][1]

    #print(_prueba_batch)
    #print(data_tensor)
    #print(target)

    return((data_tensor, target))

class TDNN(nn.Module):
    def __init__(self, kernels, input_nchar, bias=False):
        """
        :param kernels: array of pairs (width, out_dim)
        :param input_nchar: number of chars of every input word
        :param bias: whether to use bias when convolution is performed
        """
        super(TDNN, self).__init__()

        self.input_nchar = input_nchar

        self.kernels = kernels

        #self.kernels = (t.Tensor(out_dim, 20, kW).normal_(0, 0.05) for kW, out_dim in kernels)

        self.use_bias = bias

        if self.use_bias:
            self.biases = nn.ParameterList([Parameter(t.Tensor(out_dim).normal_(0, 0.05))
                                           for _, out_dim in kernels])

        self.convolutions = [None]*len(kernels)
        i = 0
        new_embedded = 0
        for kW, out_dim in kernels:

            self.convolutions[i] = nn.Sequential(
                nn.Conv1d(self.input_nchar,out_dim, kW, bias = False),
                nn.ReLU()
            )
            i+=1
            new_embedded += out_dim

        self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(new_embedded,1000, bias = False),
                nn.ReLU(),
                nn.Linear(1000, 10001, bias = False),
        )

    def forward(self, x):

        results = [self.convolutions[i](x) for i in range(0,len(kernels))]
        a1 = [result.max(2)[0].squeeze(0) for result in results]
        a = t.cat(a1, 1)
        return F.log_softmax(self.classifier(a))

def train(epoch, embedding):
        model.train()
        for batch_idx, data in enumerate(train_loader):
                input = Variable(torch.LongTensor(data[0]))
                target = Variable(torch.LongTensor(data[1]))
                load = embedding(input)
                optimizer.zero_grad()
                output = model(load)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), loss.data[0]))

        output2 = output.max(1)[1]
        print(output2)
        print(data[1])


train_loader = torch.utils.data.DataLoader(
    TranscriptionsReader('/mnt/3T-NAS/b.apd/notebook/words_meneame10k2.txt'),
    batch_size=20, shuffle=True, num_workers=1, collate_fn=my_test_collate)

kernels = [(1, 4), (2, 4), (3, 4), (4, 5), (5, 5), (6, 5), (7, 5)]
model = TDNN(kernels, 18)
embedding = nn.Embedding(54, 10)

for i in range(1,1000):
    print(i)
    optimizer = optim.SGD(model.parameters(),lr = 0.3/pow(2,round(i/20)))
    train(i, embedding)


