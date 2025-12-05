import random
import string
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer)

from data import adversarial, getFeatures, prependCorrectLabel
from helpers import prepare_dataset_nli


class Hypo(nn.Module):
    def __init__(self): # accuracy for hypothesis only nli: 0.6158854365348816
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        elektra = self.unbiasedModel(input_ids, attention_mask)
        output = elektra.logits
        return {'logits': output, "loss": self.loss_fcn(output, labels)}
        #return elektra.logits
    
class HypoEnsemble(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.unbiasedModel = AutoModelForSequenceClassification.from_pretrained('google/electra-small-discriminator', use_safetensors=True, num_labels=3)
        self.eval_model = False
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=-1)
        self.biasModel = model
        for parameter in self.biasModel.parameters():
          parameter.requires_grad = False
        #self.tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels, features=None):
        elektra = self.unbiasedModel(input_ids, attention_mask, token_type_ids)
        logits = elektra.logits
        if self.eval_model == False:
            #biased_logits = self.biasModel(features)
            hypo_tokens = token_type_ids == 1
            input_ids_hypo = input_ids * hypo_tokens
            hypo_attention_mask = attention_mask * hypo_tokens
            biased_logits = self.biasModel(input_ids_hypo, hypo_attention_mask, token_type_ids, labels)
            biased_logits = biased_logits['logits']
            output = biased_logits + logits
            #output = (self.log_softmax(logits) + self.log_softmax(biased_logits))
        else:
            output = logits
        return {'logits': output, "loss": self.loss_fcn(output, labels)}