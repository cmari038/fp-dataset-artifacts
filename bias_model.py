import random
import string
from collections import Counter

import fasttext
import fasttext.util
import nltk
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer)

from data import getFeatures, prependCorrectLabel
from helpers import prepare_dataset_nli

#fasttext.util.download_model('en')
#nltk.download('punkt_tab')

class Hypo(nn.Module):
    def __init__(self):
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels, features):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids, labels)
        return elektra.logits
    
class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4,128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,3)
    
    def forward(self, input):
        print(input)
        # return self.linear(torch.tensor(input[0], device=self.linear.weight.device))
        return self.linear2(self.relu(self.linear1(input)))  
    
class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.eval_model = False
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
        self.biasModel = train_bias()
        for parameter in self.biasModel.parameters():
          parameter.requires_grad = False
        #self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels, features):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids, labels)
        logits = elektra.logits
        if self.eval_model == False:
            biased_logits = self.biasModel(features)
            #biased_logits = self.biasModel(input_ids)
            output = (self.log_softmax(logits) + self.log_softmax(biased_logits))
        else:
            output = logits
        return {'logits': output, "loss": self.loss_fcn(output, labels)}
    
def train_bias():
    #tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)
    #model = Hypo()
    model = BiasModel()
    model.zero_grad()
    model.train()
    snli = load_dataset("snli")
    #anli = load_dataset("facebook/anli")
    dataset = snli['train'].select(range(16000))
    #dataset = anli['train_r1'].select(range(16000))
    dataset = dataset.map(getFeatures)
    #dataset = dataset.map(prependLabel)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(5):
        #model.zero_grad()
        dataset = dataset.shuffle(seed=i)
        start = 0
        for batch in range(128, len(dataset), 128):
            if batch == 0:
                continue
            set = []
            labels = []
            #print(batch)
            for i in range(start, batch):
                set.append(dataset[i]['features'])
                #set.append(prepare_dataset_nli({'premise': dataset[i]['premise'], 'hypothesis':dataset[i]['hypothesis'] }, tokenizer, 128))
                labels.append(dataset[i]['label'])
                #print(set)
                #print(labels)
            start = batch
            output = model.forward(torch.tensor(set))
            #output = model.predict(synthetic(set["hypothesis"], set["label"]))
            loss = loss_fcn(output, torch.tensor(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    train_bias()