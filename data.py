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