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
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer)

fasttext.util.download_model('en')

class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5,3)
        self.wordVectorModel = fasttext.load_model('cc.en.300.bin')
        self.device = 'cuda' if torch.cuda.is_available() else 'mps:0'
        self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    
    def get_feature(self, premise:str, hypothesis:str): # create feature vector based on bias
        features = []
        min_distances = []
        modified_premise = Counter(premise)
        modified_hypo = Counter(hypothesis)
        count = 0
        for word in modified_hypo.keys():
            distances = []
            hypo_word_vector = self.wordVectorModel.get_word_vector(word)
            for p in modified_premise.keys():
                if word == p:
                    count += 1
                premise_wordVector = self.wordVectorModel.get_word_vector(p)
                dist = 1 - nn.functional.cosine_similarity(torch.tensor(hypo_word_vector), torch.tensor(premise_wordVector), dim=0)
                #print(dist)
                distances.append(dist.item())
            min_distances.append(min(distances))
            
        if premise.find(hypothesis) != 0:
            features.append(1)        
        if count == len(hypothesis):
            features.append(1)
        else:
            features.append(0)
        features.append(count / len(hypothesis))
        features.append(sum(min_distances)/len(min_distances))
        #print(sum(min_distances)/len(min_distances))
        features.append(max(min_distances))
        return (features)
    
    def forward(self, input):
        biased_input = []
        for feature in input:
            biased_input.append(self.get_feature(feature['premise'], feature['hypothesis']))
        return self.linear(torch.tensor(biased_input, device=self.device))  
    
class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
        self.biasModel = train_bias()
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids, labels)
        logits = elektra.logits
        biased_input = []
        for input_id in input_ids:
            input = self.tokenizer.decode(input_id, skip_special_tokens=True)
            feature = {'premise':input[0:input.find('.') + 1], 'hypothesis':input[input.find('.') + 1:] }
            biased_input.append((feature['premise'], feature['hypothesis']))
        biased_logits = self.biasModel(biased_input)
        output = self.softmax(self.log_softmax(logits) + self.log_softmax(biased_logits))
        return {'logits': output, "loss": self.loss_fcn(output, labels)}
    
def train_bias():
    model = BiasModel()
    model.zero_grad()
    model.train()
    snli = load_dataset("snli")
    dataset = snli['train'].select(range(2048))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(5):
        model.zero_grad()
        dataset = dataset.shuffle(seed=i)
        #for set in dataset:
        for batch in range(0, len(dataset), 128):
            if batch == 0:
                continue
            set = []
            labels = []
            print(batch)
            for i in range(batch):
                set.append({'premise': dataset[i]['premise'], 'hypothesis':dataset[i]['hypothesis'] })
                labels.append(dataset[i]['label'])
                print(set)
                print(labels)
            output = model.forward(set)
            #output = model.predict(synthetic(set["hypothesis"], set["label"]))
            loss = loss_fcn(output, torch.tensor(labels))
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    train_bias()