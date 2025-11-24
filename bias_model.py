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
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)

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
                #dist = np.dot(hypo_word_vector, premise_wordVector) / (np.linalg.norm(hypo_word_vector) * np.linalg.norm(premise_wordVector))
                dist = 1 - nn.functional.cosine_similarity(torch.tensor(hypo_word_vector), torch.tensor(premise_wordVector), dim=0)
                #print(dist)
                distances.append(dist.item())
            min_distances.append(min(distances))
            #max_distances.append(max(distances))
            
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
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids, labels)
        logits = elektra.logits
        #print(logits)
        print(labels)
        biased_input = []
        for input_id in input_ids:
            input = self.tokenizer.decode(input_id, skip_special_tokens=True)
            print(input)
            feature = {'premise':input[0:input.find('.') + 1], 'hypothesis':input[input.find('.') + 1:] }
            #print(feature)
            biased_input.append(self.get_feature(feature['premise'], feature['hypothesis']))
        biased_logits = self.linear(torch.tensor(biased_input, device='mps:0'))
        #print(logits)
        #print(biased_logits)
        output = self.softmax(self.log_softmax(logits) + self.log_softmax(biased_logits))
        #print(output)
        return {'logits': output, "loss": self.loss_fcn(output, labels)  }
        #return (nn.Softmax(torch.log(logits[0]+biased_logits[0])), nn.Softmax(torch.log(logits[1]+biased_logits[1])), nn.Softmax(torch.log(logits[2]+biased_logits[2])))
    
def train_bias():
    model = BiasModel()
    model.zero_grad()
    model.train()
    snli = load_dataset("snli")
    dataset = snli['train'].select(range(500))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    
    for i in range(5):
        model.zero_grad()
        #random.shuffle(dataset['train'])
        dataset = dataset.shuffle(seed=i)
        for set in dataset:
        #for batch in range(0, len(dataset), 64):
            #set = dataset[batch:batch+64]
            output = model.forward(set)
            #output = model.predict(synthetic(set["hypothesis"], set["label"]))
            loss = loss_fcn(output, torch.tensor(set['label']))
            loss.backward()
            optimizer.step()
    return model

if __name__ == "__main__":
    train_bias()