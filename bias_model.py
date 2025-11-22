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

fasttext.util.download_model('en')


class BiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5,3)
        self.wordVectorModel = fasttext.load_model('cc.en.300.bin')
        
    def get_feature(self, premise:str, hypothesis:str): # create feature vector based on bias
        features = []
        min_distances = []
        #max_distances = []
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
                distances.append(np.dot(hypo_word_vector, premise_wordVector) / (np.linalg.norm(hypo_word_vector) * np.linalg.norm(premise_wordVector)))
            min_distances.append(min(distances))
            #max_distances.append(max(distances))
                
        if count == len(hypothesis):
            features.append(1)
        else:
            features.append(0)
        features.append(count / len(hypothesis))
        features.append(np.mean(np.array(min_distances)))
        features.append(max(min_distances))
        return torch.Tensor(features)
    
    def predict(self, hypothesis): # synthetic data
        return hypothesis[0]
    
    def forward(self, premise, hypothesis): # word similarity
        x = self.get_feature(premise, hypothesis)
        return self.linear(x)

def synthetic(self, hypothesis, label:int):
        data = ""
        data += str(label) + hypothesis
        return data
    
def train():
    model = BiasModel()
    model.zero_grad()
    model.train()
    dataset = load_dataset("snli")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fcn = nn.CrossEntropyLoss()
    
    for i in range(5):
        model.zero_grad()
        random.seed(i)
        random.shuffle(dataset['train'])
        for set in dataset['train']:
            output = model.forward(set["premise"], set["hypothesis"])
            #output = model.predict(synthetic(set["hypothesis"], set["label"]))
            index = max(output)
            for j in range(len(output)):
                if output[j] == index:
                    index = j
            loss = loss_fcn(output, index)
            loss.backward()
            optimizer.step()
    
    return model
      