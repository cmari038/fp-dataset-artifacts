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

#fasttext.util.download_model('en')
nltk.download('punkt_tab')
#wordVectorModel = fasttext.load_model('cc.en.300.bin')
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

def prependCorrectLabel(dataset):
    label_vals = {"0":"entailment", "1":"neutral", "2": "contradiction"}
    correct_label_chance = random.randint(1,10)
    if correct_label_chance < 9:
      if dataset['label'] == -1:
          return dataset
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

def prependRandomLabel(dataset):
    label_vals = {"0":"entailment", "1":"neutral", "2": "contradiction"}
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
        
        premise = dataset['premise']
        hypothesis = dataset['hypothesis']
        features = []
        unique_tokens = 0
        #unique_hypo_word_vector = np.zeros(300)
        #similar_hypo_word_vector = np.zeros(300)
        premise_tokens = word_tokenize(premise.lower())
        hypothesis_tokens = word_tokenize(hypothesis.lower())
        premise_words = Counter(premise_tokens)
        hypothesis_words = Counter(hypothesis_tokens)
        
        count = 0
        for word in hypothesis_words.keys():
            if word in premise_words.keys():
                count += 1
                #similar_hypo_word_vector += wordVectorModel.get_word_vector(word)
            else:
                unique_tokens += 1
                #unique_hypo_word_vector += wordVectorModel.get_word_vector(word)
        
            """
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            for word in hypothesis_words.keys():
            distances = []
            hypo_word_vector = self.wordVectorModel.get_word_vector(word)
            for p in premise_words.keys():
                if word == p:
                    count += 1
                premise_wordVector = self.wordVectorModel.get_word_vector(p)
                dist = 1 - nn.functional.cosine_similarity(torch.tensor(hypo_word_vector, device=device), torch.tensor(premise_wordVector, device=device), dim=0)
                #print(dist)
                distances.append(dist.item())
            min_distances.append(min(distances))
            """
        
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
            #features.append(count/(unique_tokens + len(premise_words)))
            features.append(len(premise_tokens) - len(hypothesis_tokens)/len(premise_tokens) + len(hypothesis_tokens))
            #features.extend(unique_hypo_word_vector.tolist())
            #features.extend(similar_hypo_word_vector.tolist())
            #features.append(sum(min_distances)/len(min_distances))
            #features.append(max(min_distances))

        else:
            features.extend([0,0,0,0,0])

        """if "not" in hypothesis_words.keys() or "no" in hypothesis_words.keys() or "n't" in hypothesis_words.keys():
          features.append(1)
        else:
          features.append(0)"""
            
        #print(features)
        dataset['features'] = features
        return (dataset)

errors = []
success = []
stats = {'0':{'0':0, '1':0, '2':0}, '1':{'0':0, '1':0, '2':0}, '2':{'0':0, '1':0, '2':0}}
with open("eval_predictions.jsonl", "r") as file:
    for prediction in file:
        pred = json.loads(prediction)
        if pred["label"] != pred["predicted_label"]:
            errors.append(pred)
            if pred["label"] == 0:
                stats['0'][str(pred['predicted_label'])] += 1
            elif pred["label"] == 1:
                stats['1'][str(pred['predicted_label'])] += 1
            else:
                stats['2'][str(pred['predicted_label'])] += 1
                
                
        else:
            success.append(prediction)
for error in errors:
    print(error)
print(stats)


