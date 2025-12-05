import json
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
from nltk import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')

def adversarial(dataset):
    dataset['premise'] = ''
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
        stop_words = set(stopwords.words("english"))
        regex_tokenizer = RegexpTokenizer(r'\w+')
        premise_tokens = regex_tokenizer.tokenize(premise.lower())
        hypothesis_tokens = regex_tokenizer.tokenize(hypothesis.lower())
        #unique_hypo_word_vector = np.zeros(300)
        #similar_hypo_word_vector = np.zeros(300)
        #premise_tokens = word_tokenize(premise.lower())
        #hypothesis_tokens = word_tokenize(hypothesis.lower())
        premise_words = Counter(premise_tokens)
        hypothesis_words = Counter(hypothesis_tokens)
        
        count = 0
        for word in hypothesis_words.keys():
            #if word in stop_words:
                #continue
            if word in premise_words.keys():
                count += 1
            else:
                unique_tokens += 1
            
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
          

        else:
            features.extend([0,0,0])

        if "not" in hypothesis_words.keys() or "no" in hypothesis_words.keys() or "n't" in hypothesis_words.keys():
          features.append(1)
        else:
          features.append(0)
            
        #print(features)
        dataset['features'] = features
        return (dataset)

def error_analysis(output_file):
    errors = []
    success = []
    stats = {'0':{'0':0, '1':0, '2':0}, '1':{'0':0, '1':0, '2':0}, '2':{'0':0, '1':0, '2':0}}
    with open(output_file, "r") as file:
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



