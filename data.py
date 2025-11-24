import json
import random

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
#ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
#ds = ds["train"].train_test_split(0.30)
# model = AutoModel.from_pretrained("/path/to/model_name", use_safetensors=True)
#ds["train"].to_json("train.json")

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
        



