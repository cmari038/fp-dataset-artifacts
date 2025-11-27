import json
import random
from collections import Counter

import fasttext
import fasttext.util
import nltk
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk import RegexpTokenizer, sent_tokenize, word_tokenize

fasttext.util.download_model('en')
nltk.download('punkt_tab')
wordVectorModel = fasttext.load_model('cc.en.300.bin')
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

def prependLabel(dataset):
    label_vals = {"0":"entailment", "1":"neutral", "2": "contradiction"}
    correct_label_chance = random.randint(1,10)
    if correct_label_chance < 9:
      holder = label_vals[str(dataset['label'])]
      holder += ' '
      holder += dataset['hypothesis']
      #holder += '0'
      dataset['hypothesis'] = holder
      return dataset
    else:
      holder = label_vals[str(random.randint(0,2))]
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
        
        #wordVectorModel = fasttext.load_model('cc.en.300.bin')
        premise = dataset['premise']
        hypothesis = dataset['hypothesis']
        features = []
        unique_tokens = 0
        unique_hypo_word_vector = np.zeros(300)
        similar_hypo_word_vector = np.zeros(300)
        premise_tokens = word_tokenize(premise.lower())
        hypothesis_tokens = word_tokenize(hypothesis.lower())
        premise_words = Counter(premise_tokens)
        hypothesis_words = Counter(hypothesis_tokens)
        
        count = 0
        for word in hypothesis_words.keys():
            if word in premise_words.keys():
                count += 1
                similar_hypo_word_vector += wordVectorModel.get_word_vector(word)
            else:
                unique_tokens += 1
                unique_hypo_word_vector += wordVectorModel.get_word_vector(word)
        
        #print(similar_hypo_word_vector)
        #print(unique_hypo_word_vector)
            
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
            features.append(count/(unique_tokens + len(premise_words)))
            features.append(len(premise_tokens) - len(hypothesis_tokens)/len(premise_tokens) + len(hypothesis_tokens))
            features.extend(unique_hypo_word_vector.tolist())
            features.extend(similar_hypo_word_vector.tolist())
        else:
            features.append(0)
            features.append(0)
            features.append(0)
            features.append(0)
            features.append(0)

        if "not" in hypothesis_words.keys() or "no" in hypothesis_words.keys() or "n't" in hypothesis_words.keys():
          features.append(1)
        else:
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



