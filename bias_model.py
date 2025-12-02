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

from data import adversarial, getFeatures, prependCorrectLabel
from helpers import prepare_dataset_nli

#fasttext.util.download_model('en')
#nltk.download('punkt_tab')

class Hypo(nn.Module):
    def __init__(self): # accuracy for hypothesis only nli: 0.6158854365348816
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids)
        output = elektra.logits
        return {'logits': output, "loss": self.loss_fcn(output, labels)}
        #return elektra.logits
    
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
    def __init__(self, model=None):
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.eval_model = False
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
        if model != None:
            self.biasModel = model
        else:
            self.biasModel = train_bias()
        for parameter in self.biasModel.parameters():
          parameter.requires_grad = False
        #self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids)
        logits = elektra.logits
        if self.eval_model == False:
            #biased_logits = self.biasModel(features)
            biased_logits = self.biasModel(input_ids, attention_mask, token_type_ids, labels)
            biased_logits = biased_logits['logits']
            output = biased_logits + logits
            #output = (self.log_softmax(logits) + self.log_softmax(biased_logits))
        else:
            output = logits
        return {'logits': output, "loss": self.loss_fcn(output, labels)}
    
def train_bias():
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)
    model = Hypo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    #model = BiasModel()
    model.zero_grad()
    model.train()
    snli = load_dataset("snli")
    #anli = load_dataset("facebook/anli")
    dataset = snli['train'].select(range(512))
    #dataset = anli['train_r1'].select(range(16000))
    #dataset = dataset.map(getFeatures)
    #dataset = dataset.map(prependCorrectLabel)
    dataset = dataset.map(adversarial)
    prepare_train_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, 128)
    train_dataset_featurized = dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=2,
            remove_columns=dataset.column_names
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(5):
        print(i)
        train_dataset_featurized = train_dataset_featurized.shuffle(seed=i)
        start = 0
        for batch in range(0, len(train_dataset_featurized), 128):
            """if batch == 0:
                continue"""
            set = train_dataset_featurized[batch:batch+128]
            input_ids = torch.tensor(set['input_ids'])
            attention_mask = torch.tensor(set['attention_mask'])
            token_type_ids = torch.tensor(set['token_type_ids'])
            labels = torch.tensor(set['label'])
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }
            output = model.forward(**model_inputs)
            loss = loss_fcn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    train_bias()