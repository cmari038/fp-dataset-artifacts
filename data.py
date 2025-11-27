import json
import random
from collections import Counter

import fasttext
import fasttext.util
import nltk
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk import RegexpTokenizer, sent_tokenize, word_tokenize

fasttext.util.download_model('en')
nltk.download('punkt_tab')

#snli = load_dataset("snli")
#dataset = snli['train'].select(range(15500))
#for i in range(15500):

def adversarial(dataset):
    #print(dataset['premise'])
    premise_words = word_tokenize(dataset['premise'])
    hypothesis_words = word_tokenize(dataset['hypothesis'])
    holder = ''
    #print(premise_words)
    for word in hypothesis_words:
        if word in premise_words:
            holder += word
            holder += ' '
    #dataset['hypothesis'] = holder
    dataset['premise'] = ''
    #print(holder)
    return dataset

def addPeriods(dataset):
    if dataset['premise'][len(dataset['premise'])-1] != '.':
        holder = dataset['premise']
        holder += '.'
        dataset['premise'] = holder
        #print(dataset['premise'])
    if dataset['hypothesis'][len(dataset['hypothesis'])-1] != '.':
        holder = dataset['hypothesis']
        holder += '.'
        dataset['hypothesis'] = holder
        #print(dataset['hypothesis'])
    return dataset

def prependLabel(dataset):
    correct_label_chance = random.randint(1,10)
    if correct_label_chance < 9:
      holder = str(dataset['label'])
      holder += ' '
      holder += dataset['hypothesis']
      #holder += '0'
      dataset['hypothesis'] = holder
      return dataset
    else:
      holder = str(random.randint(0,2))
      holder += ' '
      holder += dataset['hypothesis']
      #holder += '1'
      dataset['hypothesis'] = holder
      return dataset

def getFeatures(dataset):
        def isSubsequence(premise_tokens, hypothesis_tokens):
            htoken = 0
            ptoken = 0
            while htoken < len(hypothesis_tokens) and ptoken < len(premise_tokens):
                if hypothesis_tokens[htoken] == premise_tokens[ptoken]:
                    htoken += 1
                ptoken += 1
            if htoken == len(hypothesis_tokens):
                return True
            else:
                return False
        
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        wordVectorModel = fasttext.load_model('cc.en.300.bin')
        premise = dataset['premise']
        hypothesis = dataset['hypothesis']
        features = []
        min_distances = []
        premise_tokens = word_tokenize(premise.lower())
        hypothesis_tokens = word_tokenize(hypothesis.lower())
        premise_words = Counter(premise_tokens)
        hypothesis_words = Counter(hypothesis_tokens)
        
        count = 0
        for word in hypothesis_words.keys():
            distances = []
            hypo_word_vector = wordVectorModel.get_word_vector(word)
            for p in premise_words.keys():
                if word == p:
                    count += 1
                premise_wordVector = wordVectorModel.get_word_vector(p)
                dist = 1 - nn.functional.cosine_similarity(torch.tensor(hypo_word_vector, device=device), torch.tensor(premise_wordVector, device=device), dim=0)
                #print(dist)
                distances.append(dist.item())
            min_distances.append(min(distances))
            
        if len(hypothesis) > 0:
            if isSubsequence(premise_tokens, hypothesis_tokens):
                features.append(1)
            else:
                features.append(0)
        else:
            features.append(0)
        if count == len(hypothesis_words) and len(hypothesis) > 0:
            features.append(1)
        else:
            features.append(0)
        if len(hypothesis) > 0:
            features.append(count / len(hypothesis_words))
        else:
            features.append(0)
        if len(min_distances) > 0:    
            features.append(sum(min_distances)/len(min_distances))
            features.append(max(min_distances))
        else:
            features.append(0)
            features.append(0)
            
        #print(features)
        dataset['features'] = features
        return (dataset)

errors = []
success = []
with open("eval_predictions.jsonl", "r") as file:
    for prediction in file:
        pred = json.loads(prediction)
        if pred["label"] != pred["predicted_label"]:
            errors.append(pred)
        else:
            success.append(prediction)
for error in errors:
    print(error)
#print(success)



