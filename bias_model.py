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
from torch.mps import device_count
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer)

from helpers import prepare_dataset_nli

fasttext.util.download_model('en')
nltk.download('punkt_tab')

class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5,3)
        self.wordVectorModel = fasttext.load_model('cc.en.300.bin')
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def get_feature(self, premise:str, hypothesis:str): # create feature vector based on bias
        features = []
        min_distances = []
        premise_words = word_tokenize(premise)
        hypothesis_words = word_tokenize(hypothesis)
        count = 0
        #print(premise_words)
        #print(hypothesis_words)
        for word in hypothesis_words:
            distances = []
            hypo_word_vector = self.wordVectorModel.get_word_vector(word)
            for p in premise_words:
                if word == p:
                    count += 1
                premise_wordVector = self.wordVectorModel.get_word_vector(p)
                dist = 1 - nn.functional.cosine_similarity(torch.tensor(hypo_word_vector, device=self.device), torch.tensor(premise_wordVector, device=self.device), dim=0)
                #print(dist)
                distances.append(dist.item())
            min_distances.append(min(distances))
            
        if premise.find(hypothesis) != -1:
            features.append(1)
        else:
            features.append(0)        
        if count == len(hypothesis) and len(hypothesis) > 0:
            features.append(1)
        else:
            features.append(0)
        features.append(count / len(hypothesis))
        features.append(sum(min_distances)/len(min_distances))
        #print(sum(min_distances)/len(min_distances))
        features.append(max(min_distances))
        print(features)
        return (features)
    
    def forward(self, input):
        biased_input = []
        #print(input)
        for feature in input:
            biased_input.append(self.get_feature(feature['premise'], feature['hypothesis']))
        #print(biased_input)
        # return self.linear(torch.tensor(input[0], device=self.linear.weight.device))
        return self.linear(torch.tensor(biased_input, device=self.linear.weight.device))  
    
class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
        self.biasModel = train_bias()
        for parameter in self.biasModel.parameters():
          parameter.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids, labels)
        logits = elektra.logits
        biased_input = []
        for input_id in input_ids:
            input = self.tokenizer.decode(input_id, skip_special_tokens=True)
            feature = {'premise':input[0:input.find('.') + 1], 'hypothesis':input[input.find('.') + 1:] }
            biased_input.append({'premise':feature['premise'], 'hypothesis': feature['hypothesis']})
        biased_logits = self.biasModel(biased_input)
        #biased_logits = self.biasModel(input_ids)
        output = self.softmax(self.log_softmax(logits) + self.log_softmax(biased_logits))
        return {'logits': output, "loss": self.loss_fcn(output, labels)}
    
def train_bias():
    #tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)
    model = BiasModel()
    model.zero_grad()
    model.train()
    snli = load_dataset("snli")
    anli = load_dataset("facebook/anli")
    dataset = snli['train'].select(range(2048))
    #dataset = anli['train_r1'].select(range(1024))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(5):
        model.zero_grad()
        dataset = dataset.shuffle(seed=i)
        #for set in dataset:
        start = 0
        for batch in range(128, len(dataset), 128):
            if batch == 0:
                continue
            set = []
            labels = []
            #print(batch)
            for i in range(start, batch):
                set.append({'premise': dataset[i]['premise'], 'hypothesis':dataset[i]['hypothesis'] })
                #set.append(prepare_dataset_nli({'premise': dataset[i]['premise'], 'hypothesis':dataset[i]['hypothesis'] }, tokenizer, 128))
                labels.append(dataset[i]['label'])
                #print(set)
                #print(labels)
            start = batch
            output = model.forward(set)
            #output = model.predict(synthetic(set["hypothesis"], set["label"]))
            loss = loss_fcn(output, torch.tensor(labels))
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    train_bias()