import json
import random
from collections import Counter

from datasets import load_dataset
from nltk import RegexpTokenizer, sent_tokenize, word_tokenize

"""errors = []
success = []
with open("eval_predictions_copy.jsonl", "r") as file:
    for prediction in file:
        pred = json.loads(prediction)
        if pred["label"] != pred["predicted_label"]:
            errors.append(pred)
        else:
            success.append(prediction)
print(errors)
print(success)"""

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

def addCorrectLabel(dataset):
    holder = dataset['label']
    holder += dataset['hypothesis']
    #holder += '0'
    dataset['hypothesis'] = holder
    return dataset

def addRandLabel(dataset):
    holder = random.randint(0,2)
    holder += dataset['hypothesis']
    #holder += '1'
    dataset['hypothesis'] = holder
    return dataset
        



